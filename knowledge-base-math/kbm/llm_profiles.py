"""
kbm/llm_profiles.py - what each generator model needs, in one table.

The generator used to be a single literal in kbm/retrieval.py, which was fine while there
was exactly one and it needed nothing special. Tool-integrated reasoning ends that:
whether a model can drive the Python sandbox, how big its context window is, and how
many tokens a full answer needs are all properties OF THE MODEL, and they arrive
together or not at all. Scattering them across deps.py, routes.py and eval.sh is how
you end up running Qwen with deepseek's 350-token cap and concluding that TIR does not
work.

So: name a model, get its settings. Environment variables still win over everything
here — the table supplies the right default for each model, not a policy.

Matching is by substring against the lowercased Ollama tag rather than by exact key,
because the same weights ship under several names (`hf.co/bartowski/Qwen2.5-Math-7B-
Instruct-GGUF:Q4_K_M`, `mightykatun/qwen2.5-math:7b`) and pinning a quantization should
not silently drop a model back to the fallback profile.

This module imports nothing from the rest of the repo, so kbm/config.py can import it
without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """Everything the serving code needs to know about a generator."""

    name: str
    tir: bool          # can it drive the sandbox? (see kbm/tools/tir.py)
    num_ctx: int       # the model's real window — NOT Ollama's default
    num_predict: int   # decode cap for one pass
    # Can it be handed a JSON tool schema? (see kbm/tools/agent.py) `tir` and `tools` are both
    # CAPABILITIES — two protocols a model may have been trained for, and qwen3 has
    # both. Which one it actually gets is a policy decision, and it is made in exactly
    # one line in kbm/config.py so no combination of environment variables can turn both on
    # at once. Keeping them independent here is what lets EVALUATION.md §12's arm D
    # (qwen3 + TIR) still be measured against arm E (qwen3 + tools).
    tools: bool = False
    think: bool | None = None   # Qwen3-style thinking mode; None = model has none
    notes: str = ""


# Ordered: first match wins, so specific patterns precede general ones.
_RULES: list[tuple[tuple[str, ...], Profile]] = [
    (
        ("deepseek-math",),
        Profile(
            name="deepseek-math",
            tir=False,
            num_ctx=4096,
            num_predict=350,
            notes="The incumbent. RL-trained on chain-of-thought with \\boxed answers, and "
                  "a solver rather than an instruction-follower (ERRORS.md:247) — which is "
                  "why the server writes the Sources line itself. No TIR training: asked to "
                  "emit a ```python block it writes prose about Python instead, so tir=False "
                  "is a statement about the weights, not a policy choice. tools=False is the "
                  "same kind of statement, and this one Ollama will tell you outright: "
                  "`ollama show` reports Capabilities: completion, with no tools template.",
        ),
    ),
    (
        ("qwen2.5-math", "qwen25-math", "qwen2_5-math"),
        Profile(
            name="qwen2.5-math",
            tir=True,
            # No tools template: its tool use is kbm/tools/tir.py's text protocol, which it was
            # fine-tuned on, and that is the whole reason kbm/tools/tir.py exists as a separate
            # mechanism rather than being replaced by kbm/tools/agent.py.
            tools=False,
            # config.json says max_position_embeddings 4096. Ollama's own default is
            # larger, and setting num_ctx above the model's trained window degrades it
            # silently rather than erroring.
            num_ctx=4096,
            # 350 is deepseek's number and far too small here: a TIR trace spends tokens
            # on reasoning, a program, and then reasoning again about the result.
            num_predict=1024,
            notes="Apache-2.0. MATH 85.3 with TIR vs 83.6 with CoT — the tool is most of the "
                  "reason to run it. Qwen are explicit that it is for math QA in English and "
                  "Chinese and 'do not recommend using this series of models for other "
                  "tasks', so watch the tutoring and grounding behaviour, not just accuracy.",
        ),
    ),
    (
        ("qwen3",),
        Profile(
            name="qwen3",
            tir=True,
            # `ollama show qwen3:8b` reports Capabilities: completion, tools. This is the
            # only model here with both, so it is the only one where the exclusivity rule
            # in kbm/config.py has anything to decide — and it resolves to tools.
            tools=True,
            num_ctx=8192,
            num_predict=1024,
            # Thinking mode emits a <think> block before the answer. It would land in the
            # student's bubble, and it makes traces incomparable with the other arms.
            think=False,
            notes="The general-instruct arm of the bake-off (EVALUATION.md §12). Not a math "
                  "specialist, but it follows instructions, and it is not capped at 4096 "
                  "tokens — which is the constraint most likely to bite Qwen2.5-Math in the "
                  "RAG path, where retrieved context competes with the TIR trace.",
        ),
    ),
    (
        ("qwen2",),
        Profile(
            name="qwen2",
            tir=False,
            # The one row where this field is policy rather than capability: `ollama show
            # qwen2:7b` DOES report a tools template, and it was the spike model kbm/tools/agent.py's
            # docstring records. But this model is the eval's question-writer and judge, it
            # is not a math model, and it must never quietly become the generator. Force it
            # with KBM_TOOLS=1 when you want it — the capability probe in api/deps.py will
            # agree, because the capability is real.
            tools=False,
            num_ctx=8192,
            num_predict=512,
            notes="Eval-only: the instruct question-writer for make_evalset.py and the judge "
                  "for eval.py. Never the generator. Reports the tools capability and was "
                  "the second model the kbm/tools/agent.py protocol was verified on.",
        ),
    ),
]

# What an unrecognised model gets. Conservative on every axis: neither tool protocol for
# a model we have no evidence was trained for one, and the smallest window any 7B here
# has, so a wrong guess truncates rather than corrupting.
FALLBACK = Profile(
    name="unknown",
    tir=False,
    tools=False,
    num_ctx=4096,
    num_predict=350,
    notes="No profile matched this tag. Add one to llm_profiles._RULES before trusting it.",
)


def profile_for(model: str) -> Profile:
    """The Profile for an Ollama tag. Always returns something; never raises."""
    tag = (model or "").lower()
    for patterns, profile in _RULES:
        if any(p in tag for p in patterns):
            return profile
    return FALLBACK
