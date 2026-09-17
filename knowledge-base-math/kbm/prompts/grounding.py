"""The two mode blocks — grounded vs. general — and the trailer they share.

Pure strings, no logic. `api/chat.py:_MODE_RULES` maps the `Mode` enum onto these,
and `system_prompt()` appends the chosen one LAST so `{history}` stays at the tail of
the static prefix (two modes share every token before this block, so alternating
grounded/general mid-conversation reuses the KV cache — LATENCY.md).

NO BRACES except the `{memory}` and `{history}` template variables: these go through
ChatPromptTemplate. `{memory}` (api/chat.py:format_memory) sits just before
`{history}` because it is stable within a conversation and only varies by user, so
it belongs in the cacheable prefix ahead of history.
"""

# The two modes differ only in this trailing block, and it is deliberately the *last*
# part of the static prefix: two prompts that share a prefix also share the KV cache
# for that prefix, so alternating modes mid-conversation costs less than a full reload.
# `{memory}` is the curated per-user memory block (empty string when there is none —
# see api/chat.py:format_memory) and stays ahead of `{history}` for the same reason.
_HISTORY_TRAILER = "\n\n{memory}Conversation so far:\n{history}"

_GROUNDED_BODY = """- Context from the student's uploaded documents is provided below. Answer from it.
- If the context does not cover part of the question, say so explicitly (say which part it does not cover) rather than filling the gap silently.
- Do not write a source list or citation of your own: the server appends the exact one below your answer."""

_GENERAL_BODY = """- No relevant material was found in the student's uploaded documents, so answer from your own expertise.
- Do not claim or imply that any uploaded document supports what you say, and do not cite sources."""

GROUNDED_RULES = _GROUNDED_BODY + _HISTORY_TRAILER
GENERAL_RULES = _GENERAL_BODY + _HISTORY_TRAILER
