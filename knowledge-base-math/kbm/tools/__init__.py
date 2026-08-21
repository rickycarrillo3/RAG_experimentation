"""kbm.tools - the two tool protocols, and the one gate they both go through.

    kbm/tools/sandbox.py   executes model-written Python. An AST allow-list, a short-lived process
                 with a scrubbed environment, rlimits. A gate, not a jail — DEPLOYMENT.md §8.
    kbm/tools/tir.py       the TEXT protocol: a ```python block, the ```output stop word, a splice
                 back into the same assistant turn. Qwen2.5-Math was fine-tuned on it.
    kbm/tools/agent.py     NATIVE tool calling: a JSON schema out, a structured tool_calls array
                 back, results as ToolMessages in new turns. Also reaches retrieval.

They live together because they answer one question — how does the model ask for
something — and a model gets exactly one of them, decided in a single line in
kbm/config.py. See AGENT.md. Both modules are pure: no I/O, no model handles, no
execution, so the server and the eval harness drive one protocol rather than two
implementations of it.
"""
