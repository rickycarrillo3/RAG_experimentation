# Agent mode — the generator as a tool user

*2026-08-20.*

With `KBM_TOOLS` on, the generator stops being a text producer that a pipeline feeds and
becomes something that asks. It can run Python, and — this is the new part — it can decide
that the retrieval the server already did was not the retrieval the question needed, and go
back for more.

This file is the decision record: what the arm is, what it deliberately does *not* take
from the model, and why it is hand-rolled rather than a LangGraph agent.

---

## 1. Two protocols, one of them per model

There are two ways a model in this repo reaches a tool, they are different mechanisms, and
**a model is never given both**:

| | `kbm/tools/tir.py` | `kbm/tools/agent.py` |
|---|---|---|
| shape | text: a ` ```python ` block, stop word ` ```output `, result spliced into the same turn | native: a JSON schema goes out, a structured `tool_calls` array comes back, results as `ToolMessage`s in new turns |
| needs | nothing — it is just text | a `tools` template in the model |
| reaches | the sandbox | the sandbox **and retrieval** |
| deepseek-math | ✗ no TIR training | ✗ `ollama show` → `Capabilities: completion` |
| Qwen2.5-Math | ✓ fine-tuned on this exact shape | ✗ no tools template |
| qwen3:8b | ✓ | ✓ ← gets this one |

`kbm/llm_profiles.py` carries `tir` and `tools` as independent **capabilities**, because qwen3
genuinely has both. Which one it actually gets is a **policy**, and it is decided in one
line in `kbm/config.py`:

```python
TOOLS_ENABLED = _flag("KBM_TOOLS", PROFILE.tools)
TIR_ENABLED   = False if TOOLS_ENABLED else _flag("KBM_TIR", PROFILE.tir)
```

Downstream of every environment variable, so no combination can hand a model a `tools`
array *and* a ` ```output ` stop word. That model would interleave the two mid-answer:
unreadable for the student, and un-attributable for the eval, which could no longer say
which protocol produced an answer. Keeping the capabilities independent is also what lets
`EVALUATION.md §12`'s arm D (qwen3 + TIR) still be measured against arm E (qwen3 + tools) —
`KBM_TOOLS=0` falls back to TIR rather than to nothing.

An environment variable can ask for a protocol the weights do not have, so `api/deps.py`
asks Ollama at startup (`/api/show` → `capabilities`) and **downgrades loudly** rather than
letting every answer fail with a 400. `/healthz` reports the effective arm as `protocol`,
and `startup.sh` prints it at stage 5.

⚠️ **The downgrade lands on `none`, not on TIR.** By the time the probe runs, the exclusivity
line has already set `TIR_ENABLED = False` because `KBM_TOOLS` was on — so forcing
`KBM_TOOLS=1` onto a TIR-capable model whose probe fails loses *both* protocols. That is
the right default (silence beats a protocol nobody asked for), but it is a footgun when
switching between bake-off arms D and E: use `KBM_TOOLS=0` to fall back to TIR, not an
unset variable. A 404 from `/api/show` — the model is not pulled — is treated as a hard
refusal rather than "cannot tell", because it is the most actionable signal on a fresh pod.

## 2. The three tools

| tool | wraps | budget |
|---|---|---|
| `search_documents(query)` | `retrieval.retrieve_detailed` | `MAX_SEARCH_ROUNDS = 2` |
| `run_python(code)` | `sandbox.run_python` | `MAX_PYTHON_ROUNDS = 3` |
| `list_documents()` | `deps.index_summary` | pass ceiling only |

Two more bounds sit on top: **`MAX_CALLS_PER_PASS = 4`** — a model that asks for eight
things in one message gets the first four executed and a budget notice for the rest, which
is user-visible and easy to miss — and **`MAX_PASSES`**, which feeds the loop's overall
pass ceiling in `api/routes.py`.

Budgeted apart for the reason `api/routes.py` already gives for keeping tool rounds and
continuations apart: they mean different things, and one shared number lets whichever the
model reaches for first starve the other. The binding constraint is **context, not cost** —
qwen3 runs at `num_ctx=8192`, every round re-sends the whole transcript, and Ollama answers
an overflow by shifting the window from the left, which eats the system prompt first.
`agent.SEARCH_RESULT_CHARS = 1200` is `sandbox.MAX_OUTPUT_CHARS`' counterpart and matters
more, because five retrieved chunks are a few thousand characters and a sandbox result is a
number.

### `user` is never a tool parameter

No schema in `agent.TOOL_SCHEMAS` has a `user` field and none may ever gain one. The caller
closes `user` over from the request.

Multi-user isolation in this project is per-username directory naming, not auth — so a
`user` the model can write is a `user` the model can change. This matters more here than
anywhere else in the repo, because `search_documents` returns text the family **uploaded**:
a PDF carrying instructions aimed at the model is a prompt-injection route
(`DEPLOYMENT.md §8`), and the closure is the control that stops it reaching another family
member's index. Using static JSON dicts rather than `@tool`-decorated callables is what
makes this structural instead of remembered — there is no parameter to fill in.

## 3. Provenance did not move

`mode` is still the server's decision, made once, up front, by the calibrated 0.15
relevance floor on the question **as the student asked it**. A mid-answer search does not
revise it.

That is deliberate. `EVALUATION.md §10.8` depends on the label to make faithfulness
measurable at all, and a label that quietly means something different on answers where the
model happened to search again is not a label. So:

- `mode` — the upfront decision. Unchanged, and comparable across every answer ever logged.
- `done.sources` — the union: upfront plus anything a search found that cleared the **same**
  floor.
- `done.late_sources` — how many of those arrived after generation started. This is how a
  client sees an answer became better grounded than its label, without the label moving.
- the `Sources:` footer — names documents whenever there are documents to name, including
  the case where the answer started `general` and a search found something.

`sources` stays a **once-only** SSE frame. Its job is to fill the dead time before the first
token; by the time a mid-answer search runs, that job is done, and a second frame would
double-render in any client written to the documented contract.

## 4. Why hand-rolled, and when to move to LangGraph

**Not LangGraph, for now.** `langgraph` is in neither `requirements.txt` nor the venv, and
adding it would buy nothing this arm needs today. The loop in `api/routes.py` already
exists and already carries, for the other two arms, everything a graph rewrite would have
to reproduce byte-for-byte:

- SSE frame emission, interleaved with generation
- `PrefillEcho` / `QuestionEcho` / `CodeFenceFilter` — three filters whose contract is
  *filter what is shown, never what the model is fed back*
- the truncation/continuation budget, which needs `done_reason` off `response_metadata` —
  the reason there is no LCEL chain here at all (`StrOutputParser` discards it, and
  truncated answers shipped silently for as long as it was in the chain)
- LATENCY.md's prompt-prefix rule, which every arm preserves by only ever *appending*

Adding a third arm to that loop was a state change (`turns`), a break-condition fix, and a
dispatch block. A graph would have been a rewrite of the streaming contract.

**What would make the move worth it**, and the point at which this decision should be
revisited:

1. **Parallel tool calls.** The current arm executes a pass's calls in order. LangGraph's
   `ToolNode` does this properly, and search + compute in one pass is a real latency win.
2. **A second tool-using surface.** One loop is fine; two hand-rolled loops that must agree
   is the thing `kbm/retrieval.py` exists to prevent.
3. **Interrupts / human-in-the-loop.** "The model wants to run this — approve?" is
   checkpointing, and hand-rolling checkpointing is a bad trade.
4. **Branching or multi-agent.** A retrieval critic, a separate solver — anything where the
   control flow stops being a loop.

None of those is true today. When one becomes true, the migration cost is concentrated in
`_chat_stream`: the SSE emission and the three display filters would have to move to the
edge of the graph rather than living inside the generation loop.

## 5. Measured before it was built

Against `qwen2:7b` (`ollama show` → `completion, tools`), before the loop was written —
the `ERRORS.md` 2026-08-20 lesson that when behaviour depends on the model, a second model
is the only test:

- **A tool call very often arrives with empty content.** Two of three probe questions
  produced *zero* text alongside the call. Any loop treating "no text" as a reason to stop
  swallows most tool calls; `routes.py`'s break condition carries `and not calls` for
  exactly this, and the failure it prevents would have been silent.
- **`args` arrive complete, as a parsed dict, on one chunk.** Ollama does not stream
  argument fragments, so accumulating `AIMessageChunk`s is unnecessary.
- **`done_reason` is `"stop"` for a tool call** — the same value as EOS, the same ambiguity
  `kbm/tools/tir.py:40-49` warns about. Detect a call from `.tool_calls`, never from `done_reason`.
- **Call ids are freshly minted `uuid4()`s** (`langchain_ollama/chat_models.py:218`), not
  the model's. Dedupe on `(name, args)`; an id-keyed dedupe would double-execute if Ollama
  ever repeated a `tool_calls` block.
- **The `AIMessage` + `ToolMessage` round-trip resumes cleanly** — no restating, no
  duplication, on both qwen2 and qwen3.
