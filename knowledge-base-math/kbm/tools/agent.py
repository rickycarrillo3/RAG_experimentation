"""
kbm/tools/agent.py - the native tool-calling protocol, and nothing else.

This is OpenAI-style function calling over Ollama's own `tools` field: a JSON schema
goes out with the request, the model answers with a structured `tool_calls` array, and
each call comes back as a `ToolMessage` in a *new* turn. It is emphatically NOT kbm/tools/tir.py's
text protocol, and the two are mutually exclusive by construction (kbm/config.py) — a model
handed both a `tools` array and a ```output stop word has two ways to do one thing and
will interleave them mid-answer, which is unreadable for the student and un-attributable
for the eval.

Which protocol a model gets is a property of the model, not a policy:

    deepseek-math-7b-rl   `ollama show` -> Capabilities: completion        -> neither
    Qwen2.5-Math-7B       fine-tuned on the ```python/```output shape      -> kbm/tools/tir.py
    qwen3:8b              Capabilities: completion, tools                  -> this file

Measured on qwen2:7b before any of this was written (the ERRORS.md 2026-08-20 lesson —
when behaviour depends on the model, a second model is the only test):

  - `args` arrive as a COMPLETE parsed dict on a single chunk. Ollama does not stream
    argument fragments the way the OpenAI delta protocol does, so a caller must not try
    to accumulate AIMessageChunks to assemble a call. langchain_ollama mints a fresh
    uuid4 per parse, so the call `id` is a runtime artefact and NOT a dedupe key.
  - done_reason is "stop" for a tool call, exactly as it is for EOS. A caller must
    decide "was that a tool call?" from `.tool_calls`, never from done_reason — the same
    warning kbm/tools/tir.py:40-49 carries, for the same reason.
  - **A tool call very often arrives with EMPTY content.** Two of three probe questions
    produced zero text alongside the call. Any loop that treats "the model produced no
    text" as a reason to stop will therefore swallow most tool calls and emit nothing.

Pure — no I/O, no model handles, no execution — for the same reason kbm/tools/tir.py and
api/chat.py are: the pieces that decide what the transcript looks like should be testable
without a GPU, and the server and evaluation/self_consistency.py must drive one protocol
rather than two implementations of it.

⚠️ THE ONE RULE THAT IS NOT NEGOTIABLE: no tool schema here has a `user` parameter, and
none may ever gain one. Multi-user isolation in this project is per-username directory
naming, not auth (CLAUDE.md), so a `user` the model can write is a `user` the model can
change. The caller closes `user` over from the request. This matters more here than
anywhere else in the repo because search_documents reads text the family UPLOADED: a PDF
carrying instructions aimed at the model is a prompt-injection route (DEPLOYMENT.md §8),
and the closure is what stops it reaching another family member's index.
"""

from __future__ import annotations

import json

# ── Tool names ────────────────────────────────────────────────────────────────
# Constants rather than string literals at the dispatch site: a typo in a name is a tool
# that silently never runs, and the model is not able to tell you it asked.
TOOL_SEARCH = "search_documents"
TOOL_PYTHON = "run_python"
TOOL_LIST = "list_documents"


# ── Schemas ───────────────────────────────────────────────────────────────────
# Raw OpenAI-shaped dicts, not @tool-decorated callables and not pydantic models.
#
# Two reasons, both structural. A dict imports nothing, so this module stays as pure and
# as cheap to import as kbm/tools/tir.py — evaluation/self_consistency.py picks it up without
# dragging in LangChain's tool machinery. And a schema cannot close over anything, so
# there is no place for `user` to hide: the absence above is enforced by the shape of the
# thing rather than remembered by whoever edits it next.
#
# langchain_core.utils.function_calling.convert_to_openai_tool passes a dict of exactly
# this shape through untouched, so what is written here is what Ollama receives.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH,
            "description": (
                "Search the student's own uploaded documents (their textbooks and notes) "
                "for passages relevant to a query. Use this when the question is about "
                "their material, or when the context already provided does not cover it. "
                "Search with the mathematical terms you expect the book to use, not with "
                "the student's exact wording."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search terms. A few keywords work better than a sentence — "
                            "'chain rule proof composite derivative', not 'can you tell "
                            "me why the chain rule works'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_PYTHON,
            "description": (
                "Run a short Python program and get back what it prints. Use it for "
                "arithmetic, algebra and anything numeric you are not certain of by "
                "hand — a modular exponent, a determinant, an integral. sympy, numpy and "
                "the math module are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "The program. It must print what you want to see — a program "
                            "whose last line is a bare expression returns nothing."
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_LIST,
            "description": (
                "List the documents the student has uploaded. Use it to check whether "
                "they have material on a topic before searching, or to tell them what "
                "you can see."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = frozenset({TOOL_SEARCH, TOOL_PYTHON, TOOL_LIST})


def known(name: str) -> bool:
    """Is this a tool we actually have? Models invent names, especially small ones."""
    return name in TOOL_NAMES


# ── Budgets ───────────────────────────────────────────────────────────────────
# Separate budgets per tool, for the reason api/routes.py already gives for keeping tool
# rounds and continuations apart: they mean different things, and one shared number lets
# whichever the model reaches for first starve the other. Three sympy calls must not eat
# the search the answer actually needed.
#
# The binding constraint is CONTEXT, not cost. qwen3:8b runs at num_ctx=8192 and every
# round re-sends the whole transcript — system prompt, history, the upfront retrieved
# context, the question, and every previous call and result. Ollama answers an overflow
# by shifting the window from the LEFT, so the first casualty is the system prompt and
# the model quietly stops being a tutor mid-answer. These numbers are that budget.
MAX_SEARCH_ROUNDS = 2   # a second search is a rewrite; a third is thrashing
MAX_PYTHON_ROUNDS = 3   # tir.MAX_TOOL_ROUNDS' value, named separately so they can diverge
MAX_CALLS_PER_PASS = 4  # one message asking for eight things at once
MAX_PASSES = MAX_SEARCH_ROUNDS + MAX_PYTHON_ROUNDS + 1

# The retrieval counterpart of sandbox.MAX_OUTPUT_CHARS, and needed far more badly: a
# sandbox result is a number, while five retrieved chunks are a few thousand characters
# of textbook that the transcript then carries for every remaining round.
SEARCH_RESULT_CHARS = 1200


# ── Prompt ────────────────────────────────────────────────────────────────────
# Added to the static head of the system prompt, where tir.TIR_RULES goes, so `history`
# still comes last and the cacheable prefix keeps its shape (see api/chat.py, LATENCY.md).
#
# A tools-trained model already knows the *mechanics* — the schema tells it those. What
# it cannot infer is the three things below, and the measured failure mode is over-eager
# calling: asked "why is the derivative of a constant zero?", qwen2 called run_python
# with code that printed nothing rather than simply answering. Hence rule 1 and the
# explicit "print" note that mirrors TIR_RULES.
#
# The last line resolves a real contradiction rather than adding polish. api/chat.py's
# general-mode rules open with "No relevant material was found in the student's uploaded
# documents" — a sentence frozen into the prompt before generation, which a mid-answer
# search can now falsify. Without an override the model is asked to trust a statement the
# transcript has already disproved.
#
# NO BRACES. This string goes through ChatPromptTemplate, where { and } are variable
# syntax and a literal brace raises at format time (kbm/tools/tir.py:130 learned this the hard way).
AGENT_RULES = """- You have tools. Use them when they help and answer directly when they do not — a conceptual "why" question usually needs an explanation, not a computation.
- Run Python for arithmetic or algebra you are not certain of by hand. Print what you want to see; a program that prints nothing returns nothing.
- Search the student's documents when the question is about their own material, or when the context below does not cover it. Search with the terms the textbook would use, not the student's phrasing.
- The student reads your answer, not your tool calls. Explain what came back in your own words.
- If a tool returns nothing useful or refuses, say so and answer from what you have. Never repeat a call with the same arguments.
- If a search does find something the context below did not, use it and say which document it came from.
"""


# ── Argument handling ─────────────────────────────────────────────────────────
def coerce_args(name: str, args) -> tuple[dict | None, str | None]:
    """(args, None) on success, (None, message) on failure. Never raises.

    Same contract as sandbox.run_python, and for the same reason: "the argument you sent
    was the wrong shape" is something a generator fixes on the next round, and an
    exception inside the SSE stream is not — it ends the answer with an error frame and
    tells the student nothing.

    Small models send an int where a string belongs, wrap the arguments in another dict,
    or omit a required key entirely. All three are the model's mistake to correct, so all
    three come back as text addressed to the model.
    """
    if not known(name):
        return None, unknown_tool(name)

    if isinstance(args, str):
        # Some templates hand back the JSON blob unparsed rather than an object.
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return None, arg_error(name, "the arguments were not valid JSON")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, arg_error(name, "the arguments must be an object")

    if name == TOOL_LIST:
        return {}, None

    key = "query" if name == TOOL_SEARCH else "code"
    value = args.get(key)
    if value is None:
        return None, arg_error(name, f'it needs a "{key}" argument')
    if not isinstance(value, str):
        value = str(value)
    if not value.strip():
        return None, arg_error(name, f'the "{key}" argument was empty')
    return {key: value}, None


# ── Text handed back to the model ─────────────────────────────────────────────
# Every one of these becomes a ToolMessage. They are written for the MODEL to read and
# act on, which is why they say what to do next rather than merely reporting a state —
# a tool result that just says "error" ends the answer, and one that says "answer from
# what you have" finishes it. The student-facing half of each event lives in
# api/chat.py's markers, the same Job.detail / Job.diagnostic split CLAUDE.md describes.

def unknown_tool(name: str) -> str:
    return (f'There is no tool called "{name}". The tools available are '
            f'{", ".join(sorted(TOOL_NAMES))}. Use one of those or answer directly.')


def arg_error(name: str, why: str) -> str:
    return f"The call to {name} could not be run because {why}. Try again or answer directly."


def budget_notice(name: str) -> str:
    """Out of budget for one tool. Spliced ONCE per tool.

    A dangling tool call breaks the chat template's rendering, so a denied call still
    needs a result — the native analogue of api/routes.py splicing an output block rather
    than leaving a ```python fence open. Once only: this arm consumes no budget, so a
    model that answers the notice with another call would spin for as long as the pod
    stayed up. Telling a model twice that it is out of calls is a loop with no exit.
    """
    if name == TOOL_SEARCH:
        return ("No more searches available — answer from the passages you already have, "
                "and say so if they do not cover the question.")
    if name == TOOL_PYTHON:
        return "No more calculations available — finish from what you have."
    return "That tool is not available again on this answer — finish from what you have."


def format_search_result(passages: list[tuple[str, str]], n_total: int) -> str:
    """Retrieved passages, in the shape retrieval.format_context uses for the prompt.

    Capped at SEARCH_RESULT_CHARS in total. The cap is per-result rather than per-passage
    so that one long chunk cannot crowd the others out entirely, and it is applied here
    rather than at the call site because "how much of a tool result may enter the
    transcript" is a property of the protocol's context budget.
    """
    if not passages:
        return no_results_text()

    budget = SEARCH_RESULT_CHARS // max(1, len(passages))
    blocks = []
    for source, text in passages:
        body = text.strip()
        if len(body) > budget:
            body = body[:budget].rsplit(" ", 1)[0] + " …"
        blocks.append(f"[Source: {source}]\n{body}")

    head = f"{n_total} passage{'s' if n_total != 1 else ''} found:"
    return head + "\n\n" + "\n\n".join(blocks)


def no_results_text() -> str:
    return ("Nothing in the student's documents matched that. Either search with different "
            "terms or answer from your own knowledge — and tell the student their "
            "documents do not cover it.")


def format_documents(names: list[str], n_chunks: int) -> str:
    if not names:
        return ("The student has not uploaded any documents yet, so there is nothing to "
                "search. Answer from your own knowledge.")
    listed = ", ".join(names)
    return f"The student has uploaded: {listed} ({n_chunks} passages indexed)."
