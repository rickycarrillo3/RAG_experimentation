"""
kbm/tools/sandbox.py - the only place in this repo where model-written code is executed.

Tool-integrated reasoning (kbm/tools/tir.py) works by letting the generator write a Python
program and handing it back the result. That is the whole point — EVALUATION.md §11.6
traced deepseek's r07 failure to arithmetic *execution*, not missing knowledge: it
reduced 7^100 mod 13 correctly and then computed 2401 mod 13 as 3 instead of 9. A
model that can call Python does not make that mistake.

It also means a language model now decides what code this process runs, so the gate
below is load-bearing.

WHAT THIS IS
    A gate, not a jail. Three independent layers:
      1. an AST allow-list, checked before anything runs (no filesystem, no network,
         no dynamic import, no dunder access);
      2. a separate short-lived process — never a thread — in its own session and its
         own empty temp cwd, with a scrubbed environment;
      3. rlimits (CPU, address space, file size) plus a wall-clock kill of the whole
         process group.

WHAT THIS IS NOT
    A security boundary against a determined attacker. There is no namespace, no
    seccomp filter and no container. The threat it is actually sized for is the real
    one: a 7B math model emitting a runaway loop, a typo, or — via text in a PDF the
    family uploaded — a prompt injection that tries something obvious. See
    DEPLOYMENT.md before exposing this to anyone outside the household.

Qwen's own executor (evaluation/python_executor.py in QwenLM/Qwen2.5-Math) does less
than this: a regex against `input(`, a pebble worker, and a TODO saying real sandboxing
is unimplemented. The 400-character output cap and the error format below are theirs,
kept deliberately — the model was trained on feedback shaped that way, and here the cap
does double duty as the context-budget guard (see kbm/tools/tir.py).

Nothing in here raises. Every failure becomes a string the model can read and react to,
because "your program had a SyntaxError on line 2" is something a generator can fix on
the next round, and an exception in the server is not.
"""

from __future__ import annotations

import ast
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

# ── Policy ────────────────────────────────────────────────────────────────────

# Everything a math program legitimately needs, and nothing that reaches the network,
# the filesystem, or another process. sympy is the one that matters: it is what Qwen
# reaches for by default, and it is already a transitive dependency (torch requires it).
ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "sympy", "numpy", "np", "fractions", "decimal", "statistics",
    "itertools", "functools", "collections", "operator", "random", "re", "string",
    "heapq", "bisect", "array", "copy", "abc", "numbers", "typing", "dataclasses",
})

# Names that defeat the import allow-list or reach outside the process.
BANNED_NAMES = frozenset({
    "__import__", "open", "eval", "exec", "compile", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview", "exit", "quit",
})

# Qwen's number, and it is also the context guard: Qwen2.5-Math-7B-Instruct has a
# 4096-token window, a TIR trace re-sends the whole transcript every round, and an
# unbounded `print(list_of_10000_items)` would evict the system prompt.
MAX_OUTPUT_CHARS = 400

DEFAULT_TIMEOUT_S = 10.0

# Address space cap for the child. Generous because importing numpy+sympy reserves a
# lot of *virtual* address space that is never resident.
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
# Not zero. RLIMIT_FSIZE=0 raises SIGXFSZ — a kill, not an error — the first time
# CPython tries to write a .pyc, which turns "imported sympy" into "sandbox died for
# no visible reason". PYTHONDONTWRITEBYTECODE below stops the attempt; this bounds
# what a program could write if it found a way to try.
FILE_LIMIT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SandboxResult:
    """The outcome of one execution.

    `output` is always safe to paste straight back into the model's transcript: it is
    the program's stdout, or the error text, already truncated. `ok` and `reason` are
    for the log and for telemetry — the model only ever sees `output`.
    """

    output: str
    ok: bool
    reason: str | None = None      # "refused" | "syntax" | "timeout" | "error" | None
    duration_s: float = 0.0
    truncated: bool = False


# ── Layer 1: the AST gate ─────────────────────────────────────────────────────

def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def check_code(code: str) -> str | None:
    """Reason to refuse `code`, or None if it may run.

    Deliberately a *whitelist* on imports and a blacklist on names. A blacklist of
    dangerous modules is unwinnable — `os` is obvious, `pathlib`, `shutil`, `ctypes`,
    `pickle`, `webbrowser`, `subprocess`, `socket`, `urllib`, `http`, `importlib`,
    `pty`, `code`, `runpy` are all equally fatal and the list never ends. Naming what
    is *permitted* is a list that stays short and stays correct.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Not a refusal — a fact about the program, and one the model can fix.
        return f"SyntaxError: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root not in ALLOWED_IMPORTS:
                    return f"ImportError: module {root!r} is not available in this sandbox"
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has module=None; there is no package here to import from.
            root = _root_module(node.module or "")
            if root not in ALLOWED_IMPORTS:
                return f"ImportError: module {root or '.'!r} is not available in this sandbox"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return f"NameError: {node.id!r} is not available in this sandbox"
            if node.id.startswith("__"):
                return f"NameError: {node.id!r} is not available in this sandbox"
        elif isinstance(node, ast.Attribute):
            # ().__class__.__bases__[0].__subclasses__() and friends: the standard
            # route from "no imports" back to arbitrary code. Blocking dunder access
            # closes it without having to enumerate the escapes.
            if node.attr.startswith("__"):
                return f"AttributeError: {node.attr!r} is not available in this sandbox"

    return None


# ── Layer 2/3: the child process ──────────────────────────────────────────────

# Runs inside the sandbox process. Kept as a file rather than `python -c` so the
# model's own code never has to survive shell or argv quoting.
_RUNNER = '''\
import ast, io, sys, contextlib, traceback

src = open(sys.argv[1]).read()
tree = ast.parse(src)

# Qwen's models usually print(); when they do not, the answer is the value of the last
# expression. Their harness evaluates code[-1] for the same reason.
tail = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    tail = ast.Expression(tree.body.pop().value)
    ast.fix_missing_locations(tail)

buf = io.StringIO()
scope = {"__name__": "__main__"}
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<sandbox>", "exec"), scope)
        if tail is not None:
            value = eval(compile(tail, "<sandbox>", "eval"), scope)
            if value is not None and not buf.getvalue():
                print(repr(value))
except BaseException:
    sys.stdout.write(buf.getvalue())
    # Last line only: "ZeroDivisionError: division by zero". Same shape Qwen fed its
    # model during training, and a full traceback would be mostly sandbox internals.
    lines = [ln for ln in traceback.format_exc().rstrip().split("\\n") if ln.strip()]
    sys.stderr.write(lines[-1] if lines else "Error")
    sys.exit(1)

sys.stdout.write(buf.getvalue())
'''


def _apply_limits(timeout_s: float):
    """preexec_fn for the child: detach, then cap what it can consume.

    Runs between fork and exec, in the child. Each limit is set independently because
    they are not uniformly supported — RLIMIT_AS in particular is unreliable on macOS
    and enforced properly on the Linux pod, and a failure to set one must not prevent
    the others.
    """

    def _setup():
        # Own session, so a timeout can kill the child *and* anything it spawned.
        os.setsid()

        def _limit(what, soft, hard=None):
            try:
                resource.setrlimit(what, (soft, hard if hard is not None else soft))
            except (ValueError, OSError):
                pass

        # SIGXCPU backstop for the wall-clock kill: a program that ignores signals or
        # blocks in C still burns a bounded number of CPU seconds.
        cpu = max(1, int(timeout_s) + 1)
        _limit(resource.RLIMIT_CPU, cpu, cpu + 1)
        _limit(resource.RLIMIT_FSIZE, FILE_LIMIT_BYTES)
        _limit(resource.RLIMIT_CORE, 0)
        if sys.platform != "darwin":
            # macOS accepts RLIMIT_AS and then breaks unrelated allocations; the pod is
            # Linux, which is where this actually has to hold.
            _limit(resource.RLIMIT_AS, MEMORY_LIMIT_BYTES)
            _limit(resource.RLIMIT_NPROC, 64)

    return _setup


def _child_env(home: str) -> dict[str, str]:
    """A minimal environment. Everything not named here is dropped.

    The pod's real environment carries HF tokens, a RunPod API key with power to stop
    the pod, and the paths to the family's indexes. None of it has any business being
    visible to generated code.
    """
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": home,
        "TMPDIR": home,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",   # see FILE_LIMIT_BYTES
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",            # deterministic runs, so a wrong answer reproduces
        "OMP_NUM_THREADS": "1",           # numpy/OpenMP would otherwise fan out across the pod
        "MPLBACKEND": "Agg",
    }


def _truncate(text: str) -> tuple[str, bool]:
    """Middle-ellipsis to MAX_OUTPUT_CHARS — the head and the tail both carry signal."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n...\n" + text[-half:], True


def run_python(code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Execute `code` and return what the model should be told about it.

    Never raises. A refusal, a syntax error, a timeout and a crash all come back as a
    SandboxResult whose `output` is short, plain, and useful as the next thing in the
    model's transcript.
    """
    started = time.perf_counter()

    refusal = check_code(code)
    if refusal is not None:
        return SandboxResult(
            output=refusal,
            ok=False,
            reason="syntax" if refusal.startswith("SyntaxError") else "refused",
            duration_s=time.perf_counter() - started,
        )

    with tempfile.TemporaryDirectory(prefix="kbm-sandbox-") as workdir:
        program = os.path.join(workdir, "program.py")
        runner = os.path.join(workdir, "_runner.py")
        with open(program, "w", encoding="utf-8") as f:
            f.write(code)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)

        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", runner, program],   # -I: no cwd/user site-packages, no PYTHON* vars
                cwd=workdir,
                env=_child_env(workdir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                preexec_fn=_apply_limits(timeout_s),
            )
        except OSError as e:
            return SandboxResult(
                output=f"Error: could not start the Python sandbox ({e})",
                ok=False, reason="error", duration_s=time.perf_counter() - started,
            )

        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, err = proc.communicate()
            return SandboxResult(
                output=f"TimeoutError: the program did not finish within {timeout_s:.0f} seconds",
                ok=False, reason="timeout", duration_s=time.perf_counter() - started,
            )
        finally:
            if proc.poll() is None:
                _kill_group(proc)

    duration = time.perf_counter() - started

    if proc.returncode != 0:
        # SIGXCPU/SIGKILL/SIGXFSZ leave no traceback: report the limit, not the signal.
        message = err.strip() or _signal_message(proc.returncode, timeout_s)
        text, cut = _truncate(message)
        return SandboxResult(output=text, ok=False, reason="error",
                             duration_s=duration, truncated=cut)

    body = out.strip()
    if not body:
        # A program that computes and prints nothing is a dead end for the model; say
        # so rather than handing back an empty block it will try to interpret.
        return SandboxResult(output="(no output — the program printed nothing)",
                             ok=True, duration_s=duration)

    text, cut = _truncate(body)
    return SandboxResult(output=text, ok=True, duration_s=duration, truncated=cut)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _signal_message(returncode: int, timeout_s: float) -> str:
    if returncode == -signal.SIGXCPU:
        return f"TimeoutError: the program used more than {timeout_s:.0f} seconds of CPU"
    if returncode == -signal.SIGKILL:
        return "MemoryError: the program was stopped for using too much memory or time"
    if returncode == -signal.SIGXFSZ:
        return "OSError: the program tried to write too much data"
    return f"Error: the program exited with status {returncode}"
