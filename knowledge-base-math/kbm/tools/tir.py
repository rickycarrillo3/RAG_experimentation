"""
kbm/tools/tir.py - the tool-integrated-reasoning protocol, and nothing else.

Qwen2.5-Math's TIR is not OpenAI-style function calling. There is no JSON tool schema,
no <tool_call> tag and no `bind_tools()` — it is a *text* protocol the model was trained
on, and the whole of it is this:

    ... natural-language reasoning ...
    ```python
    print(pow(7, 100, 13))
    ```
    ```output          <- the model stops here; the runtime fills the block in
    9
    ```
    ... reasoning continues, now knowing the answer is 9 ...

So the runtime's job is three things: stop generation when the model opens an `output`
block, execute the program above it (kbm/tools/sandbox.py), and splice the result back into the
same assistant turn. That is what this module describes. It is pure — no I/O, no model
handles, no execution — for the same reason api/chat.py is: the pieces that decide what
the prompt and the transcript look like should be testable without a GPU.

Sources for every constant below: QwenLM/Qwen2.5-Math `evaluation/math_eval.py`
(stop words, `max_func_call`, the `exec_result` splice) and the Qwen2.5-Math-7B-Instruct
model card (the system prompt). They are Qwen's values, not ours, deliberately — the
model was trained against this exact shape and drifts off-distribution if we improvise.

The two loops that use this (api/routes.py, streaming and async;
evaluation/self_consistency.py, batch and sync) are necessarily different code. The
protocol they implement is not, and lives here — same rule kbm/retrieval.py exists for.
"""

from __future__ import annotations

import re

# Generation must stop the moment the model opens an output block, or it will happily
# hallucinate the interpreter's answer and reason on from a made-up number — which is
# strictly worse than not having a tool at all.
#
# Ollama strips the stop text from what it returns and reports done_reason == "stop",
# which is the SAME value it reports for a natural EOS (verified locally on ollama with
# both qwen2 and deepseek). So a caller must NOT decide "was that a tool call?" from
# done_reason. That is what pending_program() is for.
#
# "\n\n---" is Qwen's second stop word for their few-shot base-model prompt, where it
# separates demonstrations. It has no meaning in a chat-template turn and would truncate
# any answer containing a horizontal rule, so it is deliberately not included here.
STOP_WORDS = ["```output"]

# Qwen's `max_func_call = 4` counts generation passes; this counts executions, which is
# the same budget expressed the way the loop needs it.
#
# It is also a context guard, and that is the binding constraint here rather than cost:
# Qwen2.5-Math-7B-Instruct has a 4096-token window, and every round re-sends the entire
# transcript — system prompt, history, retrieved context, question, and all previous
# code and output blocks. Ollama responds to an overflow by shifting the window from the
# left, so the first thing an over-long TIR trace destroys is the system prompt, and the
# model quietly stops being a tutor mid-answer. Three rounds plus
# sandbox.MAX_OUTPUT_CHARS keeps the trace bounded.
MAX_TOOL_ROUNDS = 3

# Non-greedy, so `finditer` yields each block separately rather than one match spanning
# from the first fence to the last. The opening fence tolerates the whitespace variants
# models actually emit (```python, ``` python, ```Python).
_PY_BLOCK = re.compile(r"```[ \t]*[Pp]ython[ \t]*\r?\n(.*?)```", re.DOTALL)


def pending_program(text: str) -> str | None:
    """The program the model is waiting on a result for, or None if it is not waiting.

    `text` is the whole assistant turn so far — reasoning, code blocks, and any output
    blocks already spliced in — because "is this a tool call?" is a question about the
    transcript, not about the last chunk.

    The model is waiting exactly when the transcript ENDS with a closed ```python block.
    Anything after that fence means it moved on: prose, a final answer, or an ```output
    block the runtime already filled in. Qwen's own harness asks a looser version of
    this (`"boxed" not in output and output.endswith("```")`); the tail check subsumes
    their \\boxed test, since a boxed answer written after the code is exactly a
    non-empty tail.

    An UNCLOSED ```python block returns None on purpose. That is not a tool call, it is
    a generation cut off mid-program by the token cap, and it belongs to the
    continuation arm — running half a program would feed the model a SyntaxError for a
    mistake it did not make.
    """
    matches = list(_PY_BLOCK.finditer(text))
    if not matches:
        return None

    last = matches[-1]
    if text[last.end():].strip():
        return None

    program = last.group(1).strip()
    return program or None


def format_result(output: str) -> str:
    """The output block to splice into the assistant turn, byte-for-byte as Qwen's
    harness writes it:

        exec_result = f"\\n```output\\n{exec_result}\\n```\\n"
        query += exec_result

    The leading and trailing newlines are part of the format the model saw in training,
    not cosmetic. Appending — never inserting — is also what keeps Ollama's prompt-prefix
    KV cache alive across rounds (LATENCY.md): everything before the splice point is
    unchanged, so only the new tokens are prefilled.
    """
    return f"\n```output\n{output}\n```\n"


# Added to the static head of the system prompt, so `history` still comes last and the
# cacheable prefix keeps its shape (see api/chat.py).
#
# Wording tracks Qwen's official TIR system prompt ("Please integrate natural language
# reasoning with programs to solve the problem above, and put your final answer within
# \boxed{}") with two deliberate changes:
#
#   - The \boxed{} clause is dropped. It belongs in evaluation/self_consistency.py,
#     where a boxed answer is what the scorer parses; in a tutoring bubble it renders
#     as literal LaTeX noise around a number the student can already see.
#   - The protocol is spelled out. Qwen's one-liner works because the model is
#     fine-tuned on the format; saying "stop after the block, the result comes back to
#     you" costs a handful of cached tokens and makes the same prompt survive being
#     pointed at a general instruct model, which is one of the bake-off arms.
#
# NO BRACES. This string is fed through ChatPromptTemplate, where { and } are variable
# syntax — a literal brace here has to be doubled or it raises at format time.
TIR_RULES = """- You can run Python to compute anything you are not certain of by hand. Write the program in a ```python code block and stop; the result comes back to you in an ```output block, and you carry on from there.
- Use it for arithmetic, algebra, and anything numeric — a modular exponent, a determinant, an integral. sympy, numpy and the math module are available. Print what you want to see.
- Then explain the result in your own words. The student is here to understand the method, not to read code, so the program is a tool you used and not the answer you give.
"""
