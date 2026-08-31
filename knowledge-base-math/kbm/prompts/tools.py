"""The tool-protocol rule blocks, spliced into the static head of the system prompt.

Pure strings, no logic. `kbm/tools/tir.py` and `kbm/tools/agent.py` re-export these
for back-compat; `api/chat.py:system_prompt()` splices the right one in where
`history` still ends up last (see api/chat.py, LATENCY.md).

NO BRACES. Both strings are fed through ChatPromptTemplate, where `{` and `}` are
variable syntax — a literal brace has to be doubled or it raises at format time
(kbm/tools/tir.py learned this the hard way).
"""

# ── TIR (tool-integrated reasoning; Qwen2.5-Math's text protocol) ──────────────
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
TIR_RULES = """- You can run Python to compute anything you are not certain of by hand. Write the program in a ```python code block and stop; the result comes back to you in an ```output block, and you carry on from there.
- Use it for arithmetic, algebra, and anything numeric — a modular exponent, a determinant, an integral. sympy, numpy and the math module are available. Print what you want to see.
- Then explain the result in your own words. The student is here to understand the method, not to read code, so the program is a tool you used and not the answer you give.
"""

# ── Native tool calling (agent mode; OpenAI-style function calling) ────────────
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
AGENT_RULES = """- You have tools. Use them when they help and answer directly when they do not — a conceptual "why" question usually needs an explanation, not a computation.
<tool_calling_rules>
- Run Python (use run_python) for arithmetic or algebra you are not certain of by hand. Print what you want to see; a program that prints nothing returns nothing.
- Search the student's documents (search_documents) when the question is about their own material, or when the context below does not cover it. Search with the terms the textbook would use, not the student's phrasing.
- The student reads your answer, not your tool calls. Explain what came back in your own words.
- If a tool returns nothing useful or refuses, say so and answer from what you have (say you could not find a proper answer). Never repeat a call with the same arguments.
- If a search does find something the context below did not, use it and state which document it came from.
"""
