# Errors and fixes

A log of failures this project actually hit, what caused them, and what fixed them.

**Why keep it.** Most of these looked like something other than what they were — a wrong
CUDA build that reports "no GPU", a merge that passes review and breaks every entry point,
an upload path that silently triples the index. The cause is rarely near the symptom, and
the second person to hit one of these (including a future agent) should not have to
re-derive it.

**Format.** Newest first. Each entry: what you see, what it actually is, the fix, and —
where there is one — the general lesson. If a fix lives in another doc, link there rather
than duplicating it.

**Add to this file when a bug takes more than a few minutes to diagnose**, especially when
the symptom pointed somewhere misleading. Routine typos don't belong here.

---

## 2026-08-21 · The model searched once in six, because another tool's description told it not to

**Symptom.** Reported from the running app as "the model is not searching properly", after
an unrelated edit to the system prompt (XML-ish section tags, and a strengthened
`_GROUNDED_RULES` line about naming the part of the question the context does not cover).
The obvious reading was that the prompt edit had suppressed the tool — the tags were
unclosed, so `<tool_calling_rules>` swallowed the mode block and `{history}`, and the last
instruction before the history now told the model to *announce* a gap where `AGENT_RULES`
told it to *search* one.

That reading was wrong. The model was calling a tool the whole time. It was calling
`list_documents`.

**What it actually is.** `list_documents`' schema description read:

> "List the documents the student has uploaded. Use it to check whether they have material
> on a topic **before searching**, or to tell them what you can see."

That last clause is an instruction to list *first*. It is also the only zero-argument tool
in `TOOL_SCHEMAS`, so it is the cheapest call a model can emit — nothing to get wrong.
Measured on `qwen3:8b` at temperature 0, six questions that should search plus two controls
that should not, counting `search_documents` on the **first** round:

```
your prompt as-is                          1/6      0/2 false positives
list_documents description fixed           4/6      0/2
+ an AGENT_RULES line saying the same       4/6      0/2   (byte-identical outputs)
tags closed, description fixed             3/6      0/2
```

Two things fall out of that table besides the fix.

**The prose rules do not drive tool choice.** Adding "if the student refers to their own
notes, search before you answer, and do not list file names instead" to `AGENT_RULES`
changed nothing at all — the same calls, on the same questions. A tools-trained model picks
from the *schemas*. `AGENT_RULES` exists for the three things a schema cannot say
(over-eager calling, print-or-nothing, the general-mode override); it is not where tool
selection is tuned, and lengthening it to fix a selection problem buys nothing but tokens.

**The tags were not the cause, and closing them did not help** (3/6 against 4/6; stripping
them entirely measured worse still). They are worth tidying for readability. They are not a
behaviour fix, and treating the most recent edit as the cause would have burned the
investigation on them.

**Why it stayed invisible.** It was recoverable. Handed the listing, the model went on to
call `search_documents` on the next round, 2 for 2 — so the student still got a grounded,
correct answer. The cost was one pass out of `MAX_PASSES` (6) and one full transcript
re-prefill per question, which shows up as latency and as a shorter tool budget, never as a
wrong answer. A failure that only makes things slower is one nobody reports for weeks.

**Fix.** `kbm/tools/agent.py` — the description now says what the tool returns and names the
other tool for what it does not: *"It returns names, NOT content — to find out what a
document SAYS, use search_documents."* One string; no code path changed.

**Lesson.** A tool description is a prompt, and it is the prompt the model actually reads
when choosing between tools. Every schema in `TOOL_SCHEMAS` is competing with every other
one, so a sentence that mentions a sibling tool ("before searching") is a routing
instruction whether or not it was written as one — and the cheapest-to-call tool wins ties.
Read the descriptions **together**, as the model receives them, not one at a time as they
are edited. And when a symptom appears right after an unrelated edit, measure the arms:
the edit was the obvious suspect here, it was the wrong one, and only holding it fixed while
varying the schemas showed that.

---

## 2026-08-20 · A tool call with no text would have been thrown away

**Symptom.** None, on any shipped model — found by the spike run *before* the native
tool-calling loop was written, in the same way and for the same reason as the PrefillEcho
entry below.

The assumption under test was "a model that calls a tool still says something first".
Measured on `qwen2:7b`, against three probe questions:

```
"What is 7**100 mod 13?"                          -> 0 content chars + run_python call
"What does my textbook say about the chain rule?" -> 0 content chars + search_documents call
"Why is the derivative of a constant zero?"       -> 913 content chars + a call
```

**What it actually is.** `api/routes.py` ended each pass with

```python
if echo.mismatch or not produced:
    truncated = done_reason == "length"
    break
```

`not produced` is there so a model that immediately emits EOS-at-length cannot spin to the
pass ceiling emitting nothing — correct, and load-bearing, for both existing arms. But a
native tool call arrives in `AIMessageChunk.tool_calls`, not in the text, so for two of the
three questions above `produced` is `""` and the loop breaks **before the tool arm is
reached**. The call is discarded, nothing is executed, and `shown` is empty — so the server
emits `chat.NO_ANSWER_TEXT` and the student is told the model returned nothing.

It would have looked like the model failing to answer, on some questions and not others,
with nothing in the log. The tool arm would have appeared to work whenever the model
happened to narrate first.

**Fix.** `if echo.mismatch or (not produced and not calls)`. A no-op on every model without
native tools, since `calls` is then always empty.

**Lesson.** The same one as the entry below, one level up: a stop condition is a claim about
what "the model produced nothing" means, and that meaning changed when a second channel
(structured tool calls) was added beside the text. When you add a new way for a pass to
carry information, re-read every condition that tests whether a pass carried any.

Two smaller findings from the same spike, both now relied on in `kbm/tools/agent.py`:
`done_reason` is `"stop"` for a tool call exactly as it is for EOS (so detect from
`.tool_calls`, never from `done_reason` — `kbm/tools/tir.py:40-49` says the same about stop words),
and `langchain_ollama` mints a fresh `uuid4()` per parse (`chat_models.py:218`), so a call
`id` is a runtime artefact and dedupe must key on `(name, args)`.

---

## 2026-08-20 · Every qwen3 answer in the eval scored "unparseable"

**Symptom.** `evaluation/self_consistency.py --model qwen3:8b --tools` returned
`greedy_pred=None` for every question and `unparseable samples: 50%`, while the TOOL USE
block showed the sandbox *had* run. A single question took **over 7 minutes** and still
produced no `\boxed{}`. It reads exactly like a model that cannot do the questions.

**What it actually is.** `Generator.__init__` built its `ChatOllama` without `reasoning=`.
`api/deps.py` passes `reasoning=THINK`; the eval never did, so qwen3 ran with **thinking
mode on** — its default. Two things follow, and neither is visible in the output:

1. The `<think>` block does not appear in `.text` at all. `langchain-ollama` puts it in
   `additional_kwargs["reasoning_content"]`, so grepping the answer for `<think>` finds
   nothing and the leak looks like it is not happening.
2. It consumes the entire `num_predict` budget. The answer is never reached, so there is no
   `\boxed{}` to parse — and the scorer's only vocabulary for that is "unparseable".

The same omission applies to the `--tir` arm, i.e. to **arm D of the §12 bake-off**, which
had not been run yet. Had it been run first, qwen3 would have been recorded as far worse
than it is.

**Fix.** `reasoning=profile_for(model).think` in `Generator.__init__`, matching `deps.py`.
The same question afterwards: **9.0 seconds**, one sandbox round, `\boxed{391}`.

**Lesson.** `kbm/llm_profiles.py` exists so that naming a model configures it — and that only
holds where every consumer actually reads the profile. The server read `think`; the eval
read `num_ctx` and `num_predict` from the same object and silently skipped the third field.
A profile with a consumer that uses *some* of it is worse than no profile, because the
divergence is invisible: both call sites look like they are configured from one table.
When adding a field to `Profile`, grep for every `profile_for(` and check what each one
ignores.

---

## 2026-08-20 · Continuation silently produces nothing on a Qwen model

**Symptom.** None, on the shipped model — which is why this is written up before it ever
reached anyone. Found by a spike run *before* implementing tool-integrated reasoning, on
the assumption that "Ollama echoes the assistant prefill" was a property of Ollama.

It is a property of the **chat template**, and the two models in this repo disagree:

```
prefill sent:  'The first three prime numbers are 2, 3'

deepseek-math-7b-rl  -> 'The first three prime numbers are 2, 3 and 5 .'   (echoes, then continues)
qwen2:7b  (ChatML)   -> ', and 5, followed by 7 and 11.'                    (continues, no echo)
```

**What it actually is.** `chat.PrefillEcho` was written against the first behaviour and
treated the second as a fault: anything that was not a byte-identical echo set `mismatch`,
and `mismatch` makes `feed()` return `""` for the rest of the pass. On a Qwen generator
every continuation pass would therefore emit **nothing at all** — `produced` comes back
empty, `routes` breaks out of the loop, and the answer just stops at `KBM_NUM_PREDICT`
with no error anywhere. The same code path carries every TIR tool round, so the sandbox
would have appeared to do nothing either.

The reason the old code was wrong rather than merely conservative: a model that *restates*
the answer instead of continuing produces text that IS a prefix of the prefill, so it
matches for a long way before diverging. Divergence at character 0 is therefore positive
evidence that no echo was attempted — the opposite of the reading it was given.

**Fix.** `PrefillEcho` now distinguishes three cases instead of two: full echo (strip it),
diverges within `ECHO_PROBE_CHARS` (there was no echo — emit everything), diverges deep in
(unexpected — abandon, as before). Verified on both models: deepseek resumes seamlessly,
qwen2 produces a single coherent answer across 3 passes with no duplication.

**Lesson.** "Verified on ollama with deepseek-math" was written in that docstring as an
observation rather than a guarantee, and it was right to be. When behaviour depends on the
model, a second model is the only test — and swapping generators is exactly what
`KBM_LLM_MODEL` now makes easy, so this class of bug gets cheaper to find and likelier to
appear.

---

## 2026-08-20 · `think=` on ChatOllama is accepted and never sent

**Symptom.** None visible. `ChatOllama(model=..., think=False)` constructs without error
and reads back as if unset — `models.llm.think` raises `AttributeError`.

**What it actually is.** In `langchain-ollama` 1.1.0 the field is **`reasoning`**, which
`_chat_params` maps to Ollama's wire field `think`. `ChatOllama` accepts unknown kwargs
without complaint, so `think=` lands in pydantic extras, is never serialised, and Qwen3
keeps emitting `<think>` blocks into the answer.

**Fix.** `api/deps.py` passes `reasoning=THINK`. Confirmed by inspecting the assembled
request rather than the object: `llm._chat_params([...])["think"]`.

**Lesson.** A constructor that tolerates unknown keyword arguments turns every typo into a
silent default. When a kwarg controls something you cannot see in the output, assert on
the request that goes out, not on the client object.

---

## 2026-08-19 · The answer came back as the student's own question — not reproduced

**Symptom.** Reported from the running app, with no documents ingested yet: the tutor's
reply was the question that had just been asked, and nothing else. No answer under it, no
error frame.

**Status: open.** No reproduction. The shipped `general`-mode path was driven through
`t1c/deepseek-math-7b-rl:Q4` on ~15 prompts chosen to provoke it — bare greetings,
one-word turns, statements with a wrong premise, multi-turn histories, word problems — and
every one answered normally. `app.handle_chat` was driven against a live API for a user
with no index, and the displayed history was correct. So the cause is **not** in prompt
construction or the Gradio client as they stand here, and may be environment-specific
(a different Ollama or `langchain-core` build on the pod).

**What was done about it anyway.** Two guards, neither of which can damage a good answer:

- `chat.QuestionEcho` (wired in `routes._chat_stream`) holds back the head of the stream
  only while it is still a prefix of the question, and drops it only if it turns out to be
  a verbatim copy. A real answer diverges within a character or two and streams through
  untouched. When it fires it logs a warning naming the mode — **if this symptom recurs,
  that line in the server log is the first thing to look for.**
- An answer that comes back empty — or that was nothing but the question — now says so
  (`chat.NO_ANSWER_TEXT`) instead of rendering an empty bubble with a sources footer
  under it.

**What would settle it.** The `question` and `n_completion_chars` of the offending event
in `$DATA_DIR/telemetry/events.jsonl` (a completion length within a few characters of the
question length is the fingerprint), plus `ollama --version` and `pip show langchain-core`
from the machine it happened on.

**Lesson.** A guard on the symptom is not a diagnosis, and saying so in writing is what
keeps it from being mistaken for one later.

---

## 2026-08-19 · `prefetch_models.py` failed on a second pod: hf_transfer, then a "missing config.json"

**Symptom.** Stage 4 of `startup.sh` on a freshly built pod:

```
↓ embedder BAAI/bge-small-en-v1.5
  FAILED: ValueError: Fast download using 'hf_transfer' is enabled
  (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available
↓ reranker BAAI/bge-reranker-v2-m3
  FAILED: OSError: Can't load the configuration of 'BAAI/bge-reranker-v2-m3' ...
  make sure ... is the correct path to a directory containing a config.json file
```

**Cause.** One cause, two faces. RunPod's images export `HF_HUB_ENABLE_HF_TRANSFER=1` to
speed up model downloads; `huggingface_hub` honours it by raising on *every* download
when the `hf_transfer` package is absent. It was absent because nothing in
`requirements.txt` asked for it — our code never imports it.

**What made it confusing.** The second error names the wrong problem. Nothing is wrong
with the model id or the cache: the download never ran, so there is no `config.json` on
disk, and `transformers` reports the empty cache as if the repo were bad. Chasing the
reranker message leads to checking model names and clearing caches, none of which is the
issue. **When several models fail in sequence, fix the first failure and re-run before
reading the rest** — later entries in a prefetch list are usually echoes of the first.

**Fix.** `hf_transfer` added to `requirements.txt`, with a comment saying why a package
we never import is there. On a pod that already exists, `pip install hf_transfer` inside
the venv. Unsetting `HF_HUB_ENABLE_HF_TRANSFER` also works but only for that shell — the
variable comes from the image, so the failure returns next session.

**General lesson.** The pod image's environment is part of the dependency set. A variable
someone else exported can make a correct `requirements.txt` incomplete, and the resulting
error names our config rather than theirs.

---

## 2026-08-18 · Gradio UI crashed on startup; `/healthz` returned 401

Two unrelated bugs surfaced by the first full `startup.sh` run on the pod.

### `Chatbot.__init__() got an unexpected keyword argument 'type'`

**Symptom.** Stages 1–5 passed, the API came up, then stage 6 died. The UI never
started; the API was left running until the trap killed it.

**Cause.** `requirements.txt` pins `gradio==6.18.0`, but `app.py` was written against
Gradio 5.x. Gradio 6 moved `theme` from `Blocks` to `launch()` and removed `Chatbot`'s
`type` argument. The Mac had 5.x installed, so the file worked there and could not work
on the pod — **the same code could only run on one of the two machines.**

**What made it confusing.** `SETUP.md §7` claimed the pin moved gradio *to 5.50*, and §8
said this exact error meant "environment has 6.x, code targets 5.50" — the precise
inverse of the truth. Following the docs would have led to downgrading a correct
environment.

**Fix.** `app.py` now targets Gradio 6 (theme on `launch()`, no `type=` on `Chatbot`),
matching the pin. `SETUP.md §7`/`§8` corrected. Local venvs on 5.x need
`pip install -r requirements.txt`.

**Lesson.** A version pin is a contract with the code, and a doc that describes a
different pin than the file contains is worse than no doc — it argues against the fix.

### `GET /healthz 401 Unauthorized`

**Symptom.** `startup.sh` logged a 401 for its own readiness probe and then printed
`✓ Ready.` anyway.

**Cause — not the one it looks like.** The 401 was correct: `/healthz` is behind
`KBM_API_TOKEN` like every other endpoint, and the probe was not sending it. The *bug* was
that the probe tested `curl`'s exit status rather than the HTTP status code, so any
response at all counted as ready — a 401, and equally a 500. The check would have passed
against an API that answered nothing but errors.

**The wrong fix, briefly applied.** `/healthz` was first moved to an unauthenticated
router on the reasoning that liveness probes precede credentials. That argument does not
hold here: every caller today (`startup.sh`, the Gradio client, SSH) already has the
token, port 8000 is not publicly exposed, and the only caller that would need an
unauthenticated probe — a family-facing wake page — does not exist and is listed as a
known gap. Reverted.

**Fix.** Probe sends the token and switches on the status code: 200 ready, 401 a distinct
and fatal "token mismatch" message, 000 keep waiting, anything else logged and retried.

**Lesson.** A readiness check that ignores the status code is not a readiness check. And
when a fix requires relaxing a security boundary, check whether the caller that needs it
actually exists before relaxing anything.

---

## 2026-08-17 · `ollama: command not found` on a fresh pod

**Symptom.** `startup.sh` stage 3, or `ollama pull` in setup, failed — Ollama is not in
the RunPod PyTorch image, and `SETUP.md` assumed it was already there.

**The trap, not the error.** The obvious fix is Ollama's own installer:
```bash
curl -fsSL https://ollama.com/install.sh | sh      # ← don't
```
It writes to `/usr/local/bin`, which is **container filesystem** — wiped on every pod
stop. It works, then silently stops existing on the next wake, and `startup.sh` fails
again with no indication that anything was ever installed.

**Fix.** Install to the volume, preserving the tarball's `bin/` + `lib/` layout (the
binary locates its runners relative to itself):
```bash
mkdir -p /workspace/ollama
curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
  | tar --zstd -x -C /workspace/ollama
export PATH=/workspace/ollama/bin:$PATH
```
`startup.sh` now adds `$WORKSPACE/ollama/bin` to `PATH` when it exists, and fails with
these instructions when it doesn't — so no future session needs the export.

**It kept happening anyway, and that sentence is why.** "No future session needs the
export" is true of `startup.sh` and false of everything else: the export dies with the
shell, so every later SSH session gets `ollama: command not found` the moment someone runs
`ollama pull` or `ollama list` by hand. The script was fixed and the *shell* was not.
`SETUP.md §1` now appends the export to `~/.bashrc`, which is the part that actually stops
the recurrence:

```bash
grep -q '/ollama/bin' ~/.bashrc || echo 'export PATH=/workspace/ollama/bin:$PATH' >> ~/.bashrc
```

Two related traps worth naming, because both have been hit: any instruction that begins
with a bare `ollama …` assumes a PATH the pod does not have by default — and most of the
time it is not needed at all, because `startup.sh` pulls whatever `KBM_LLM_MODEL` names.

**Lesson.** The same rule that governs `DATA_DIR`, `HF_HOME` and `OLLAMA_MODELS` governs
*binaries*: on a pod, anything installed outside `/workspace` is temporary. A vendor
install script that assumes a normal machine will put it in the wrong place.

---

## 2026-08-17 · `torch.cuda.is_available()` is False on a working A5000

**Symptom.** `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
printed `False 13.0`, while `nvidia-smi` showed a healthy idle RTX A5000 with 24 GB free.

**Cause.** The pod's driver was `550.127.05`, which caps at **CUDA 12.4**. Installed torch
was a **cu130** build. CUDA guarantees compatibility within a major version (any 12.x build
runs on a driver supporting 12.0+), but not across one — a 13.x build on a 12.x driver
finds no usable device. torch does not raise; `is_available()` just returns False.

**What made it slow to find.** RunPod's pod listing advertised "CUDA 14.2", and `SETUP.md`
had been written to say a 14.x pod runs cu130 fine. Both were wrong: that number is not the
driver's CUDA version. Also, `torch.version.cuda` printing `13.0` rather than `None` means
the wheel is *not* CPU-only, so the documented "reinstall the CUDA wheel" fix was a no-op.

**Fix.**
```bash
nvidia-smi | head -3        # read the driver's real CUDA version
pip install --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Lesson.** `nvidia-smi` on the running pod is the only authority on the driver's CUDA
version. Pick the highest published index sharing that **major** version. And distinguish
`False None` (CPU wheel) from `False <version>` (CUDA build too new) — they have opposite
fixes. See `SETUP.md §1` and `§8`.

---

## 2026-08-17 · `operator torchvision::nms does not exist`

**Symptom.** Any `import torchvision` died in `torchvision/_meta_registrations.py` at
`@torch.library.register_fake("torchvision::nms")`. Would have surfaced to a user as
`Ingestion failed: ...` on the first PDF upload, since Marker imports torchvision.

**Cause.** `pip list` showed `torch 2.11.0+cu128` next to `torchvision 0.28.0` — note the
missing `+cu128`. torchvision ships compiled C++ ops linked against one exact torch build;
mismatched, the ops never register.

**Why it kept coming back.** `requirements.txt` listed a bare `torch` and no torchvision,
but `marker-pdf` depends on torchvision — so `pip install -r requirements.txt` pulled it
from **PyPI**, on top of the CUDA torch installed moments earlier. `SETUP.md`'s documented
order (torch first, then requirements) therefore *caused* the breakage every single time.

**Fix.** Uninstall both, then install as a pinned pair from one index —
`--force-reinstall` alone left stale compiled objects behind:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip check
```
`requirements.txt` now lists `torchvision` explicitly with a warning comment, and
`SETUP.md §1` installs requirements **first** so the CUDA pair is the last write.

**Lesson.** When a package's compiled extensions link against another package, they must
come from the same index, and the CUDA install has to be the last thing that runs.

---

## 2026-08-19 · `Blocks.launch() got an unexpected keyword argument 'show_api'` — and the silent half of the same change

**Symptom.** `python app.py` died at launch on gradio 6.17.3 with
`TypeError: Blocks.launch() got an unexpected keyword argument 'show_api'`. The local venv
had been on 5.50, where the argument exists; installing the pinned 6.17.3 surfaced it.

**Cause.** Gradio 6 reworked how an app declares what it publishes. `launch(show_api=...)`
is gone, replaced by `footer_links` — a list of the links to *keep* (`"api"`, `"gradio"`,
`"settings"`) rather than a flag for the one to remove.

**The part that did not raise anything.** The same release narrowed event bindings'
`api_name` to `str | None` and moved visibility to a new `api_visibility` argument
(`"public"` / `"private"` / `"undocumented"`). It does **not** reject the old value.
`api_name=False` — the 5.x spelling of "do not publish this handler", used on all six
bindings in `app.py` — was accepted, stringified, and published each handler as a
**public endpoint literally named `/False`**. Verified on the unfixed file:

```
$ python -c "...; print(json.dumps(app.get_api_info()))"
{"named_endpoints": {"/False": {...,"api_visibility": "public"}}, "unnamed_endpoints": {}}
```

So the loud failure was the harmless one. The argument protecting the backend surface
inverted its meaning in silence, and only the crash on the *neighbouring* line led here.

**Fix.** `footer_links=["gradio", "settings"]` on `launch()`, and
`api_visibility="private"` on every binding. `get_api_info()` is empty again and
`/gradio_api/info` serves `{"named_endpoints":{},"unnamed_endpoints":{}}` on a real launch.

**Lesson.** A security-relevant argument is not verified by reading the call site — it is
verified by asking the framework what it ended up exposing (`Blocks.get_api_info()`, or
the `/gradio_api/info` route). And when a major version renames one argument in a pair,
check the other: the one that still *accepts* the old value is more dangerous than the one
that raises. Related: the 2026-08-18 gradio-6 entry above — same version boundary, and
this file's own summary line for it says `theme=` moved the other way.

**A second lesson, from re-finding this independently on a branch that predated the fix.**
The first two readings of `/gradio_api/info` taken during that re-discovery were served by
a **stale `app.py` process** that had held port 7860 for over an hour, while the freshly
launched one had already died with `OSError: Cannot find empty port`. Both readings looked
like evidence about the code under test and were about something else entirely; the
conclusion happened to survive, which is worse, not better. When a server is launched to
test a change, confirm the process answering is the one under test — read the launch log,
or bind a port nothing else could be holding.

---

## Earlier · fixed, documented elsewhere

Kept short — each links to where the reasoning lives.

| What broke | Cause | Where it's written up |
|---|---|---|
| Chroma grew 66 → 136 → 210 on uploads of 66/4/4 chunks | `build_chroma` re-added the user's whole accumulated corpus; no explicit ids | `DEPLOYMENT.md §7` |
| Re-uploading a PDF raised `DuplicateIDError` | surfaced by the ids fix above; previously silent duplication | `ingest.merge_chunks` |
| Uploaded PDFs and `.mmd` vanished after ingest | `shutil.rmtree(tmp_dir)` deleted the only copy | fixed by persisting them, then **deliberately reverted** on 2026-08-19 — see the entry above |
| Every entry point raised `NameError` after a merge | `resolve_device()` called but never imported — a textually clean, semantically broken merge | found by importing, not reading |
| `startup.sh` could not run at all | `ALLOW_CPU`/`DO_PULL`/`DO_PREFETCH` read but never assigned; fatal under `set -u`. `bash -n` passed | recovered from commit `eb9044c` |
| Indexes written where retrieval never read | `CHROMA_DIR`/`BM25_DIR` declared in both `kbm/retrieval.py` and `ingest.py` | `CLAUDE.md` — define a path once, import it |
| Abstention was structurally impossible | `KBM_RELEVANCE_FLOOR` documented as a raw logit, default `0.0`; the reranker applies a Sigmoid, so every score passed | `api/settings.py`, `CLAUDE.md` |
| Continuation emitted nothing on a Qwen generator | `PrefillEcho` assumed every model echoes the assistant prefill; ChatML models do not | entry above, 2026-08-20 |
| `think=False` never reached Ollama | langchain-ollama calls the field `reasoning`; `ChatOllama` swallows unknown kwargs | entry above, 2026-08-20 |
| Answers claimed document grounding they didn't have | deepseek-math ignores the "say this isn't from your documents" instruction — it is a solver, not an instruction-follower | server appends the `Sources:` line itself; `api/chat.py:sources_footer` |
| Gradio UI crashed on startup | `theme=` passed to `launch()` instead of `Blocks()` on gradio 6.18 | `app.py` |
| Prompt re-prefilled on every turn (~6 s/turn by turn 5) | sliding history window shifted the *start* of the prompt, which a KV prefix cache cannot survive | `LATENCY.md` |
| Every question paid a 4.4 s cold model load | `keep_alive` unset, so Ollama unloaded after 5 min idle | `LATENCY.md`, `DEPLOYMENT.md §5` |
| Math PDFs routed to the wrong extractor | math detected by grepping LaTeX tokens in `pymupdf4llm` output, which contains **no LaTeX at all** | `CLAUDE.md` — the repeat offender |
| Gradio published every handler as a public `/False` endpoint | `api_name=False` (the gradio 5 spelling) is silently accepted by gradio 6, which wants `api_visibility="private"` | entry above; found only because the neighbouring `show_api` raised |
| Gradio died with `Cannot find empty port in range: 7860-7860` | `APP_HOST` was `0.0.0.0.` — a trailing dot, so `getaddrinfo` failed (Errno -2); gradio's port loop swallows the bind error and blames the port | `DEPLOYMENT.md §4`, `SETUP.md` troubleshooting |
