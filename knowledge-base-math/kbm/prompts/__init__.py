"""kbm.prompts - the production LLM prompt *text*, in one place.

Every string here is a prompt fragment the deployed service sends to the generator.
The *composition* logic that orders and splices them — which is latency-sensitive
(KV-cache prefix reuse, LATENCY.md) — stays in `api/chat.py:system_prompt()`; this
package holds only the operands.

Out of scope on purpose: `query.py`'s CLI prompts and the `evaluation/` harness
prompts, which are deliberately framed differently from what ships.

This package imports **nothing** from `kbm.*` (pure string modules), so it is safe
to import from `kbm/tools/*` without a cycle.
"""

from kbm.prompts.persona import TEACHING_STYLE, SAFETY_RULES
from kbm.prompts.grounding import GROUNDED_RULES, GENERAL_RULES
from kbm.prompts.tools import TIR_RULES, AGENT_RULES

# `context` carries its own trailing blank line (see api/chat.py:build_context) rather
# than the template hard-coding one. In `general` mode the context is empty, and a
# template with the blank line baked in handed the model a human turn that opened with
# two blank lines before "Question:" — a continuation prompt with nothing above it.
HUMAN_PROMPT = "{context}Question: {input}"

__all__ = [
    "TEACHING_STYLE",
    "SAFETY_RULES",
    "GROUNDED_RULES",
    "GENERAL_RULES",
    "TIR_RULES",
    "AGENT_RULES",
    "HUMAN_PROMPT",
]
