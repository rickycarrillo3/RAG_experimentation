"""
verify_reasoning_set.py - Recompute every gold answer independently, with sympy.

A benchmark is worthless if its answer key is wrong, and a wrong key fails in the most
expensive way possible: the model is marked wrong for being right, you go chasing a model
problem that does not exist, and every downstream number (accuracy, self-consistency gain,
the model bake-off) is quietly corrupted.

So the answers in reasoning_set_*.jsonl are not hand-checked — they are *derived here*, by
a second computation that does not look at the JSONL, and compared. If a question is edited,
this must be edited too, and the two must still agree.

    (run from knowledge-base-math/)
    python evaluation/verify_reasoning_set.py

Exits non-zero on any mismatch or any question missing a verification.
"""

import json
import os
import sys

try:
    import sympy as sp
except ImportError:                          # sympy ships with torch, but don't hard-fail
    print("sympy not installed — cannot verify the answer key. "
          "pip install sympy (or run inside the project venv).")
    sys.exit(2)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

x, y, z, t, n = sp.symbols("x y z t n")


def easy_answers() -> dict[str, sp.Expr]:
    """The easy tier, recomputed."""
    k = sp.symbols("k", integer=True)
    return {
        "r01": sp.Integer(17 * 23 - 15 * 19),
        "r02": sp.diff(x**3 * sp.log(x), x).subs(x, 1),
        "r03": sum(i for i in range(1, 101) if i % 3 == 0 or i % 5 == 0),
        "r04": sp.integrate(3 * x**2 + 2 * x, (x, 0, 1)),
        "r05": sp.binomial(5, 2) / sp.Integer(2)**5,
        "r06": (lambda s: s[x] + s[y])(sp.solve([2 * x + 3 * y - 12, x - y - 1], [x, y])),
        "r07": sp.Integer(pow(7, 100, 13)),
        "r08": sp.factorial(6) / (sp.factorial(3) * sp.factorial(2)),
        "r09": sp.limit(sp.sin(3 * x) / (5 * x), x, 0),
        "r10": sp.Integer(120) / (sp.Rational(60, 30) + sp.Rational(60, 60)),
        "r11": sp.Matrix([[2, 3], [4, 5]]).det(),
        "r12": sum(sp.solve(2 * x**2 - 8 * x + 6, x)),
        "r13": sp.log(8, 2) + sp.log(81, 3),
        "r14": sp.divisor_count(360),
        "r15": ((x**2 + 1).subs(x, (2 * x - 3).subs(x, 2))),
        "r16": sp.integrate(x - x**2, (x, 0, 1)),
        "r17": 1000 * sp.Rational(11, 10)**3,
        "r18": sp.Rational(len([(a, b) for a in range(1, 7) for b in range(1, 7) if a + b == 8]), 36),
        "r19": sp.solve(sp.Eq(3**(2 * x), 81), x)[0],
        # bat + ball: b + (b + 1) = 1.10
        "r20": sp.solve(sp.Eq(2 * sp.Symbol("b") + 1, sp.Rational(11, 10)), sp.Symbol("b"))[0],
    }


def college_answers() -> dict[str, sp.Expr]:
    """The college tier, recomputed."""
    r, th = sp.symbols("r theta", positive=True)
    A = sp.Matrix([[1, 2], [3, 4]])

    # c01 directional derivative
    f = x**2 * y
    grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)]).subs({x: 1, y: 2})
    c01 = (grad.T * sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)]))[0]

    # c02 double integral over the unit disk, in polar
    c02 = sp.integrate(sp.integrate(r**2 * r, (r, 0, 1)), (th, 0, 2 * sp.pi))

    # c03 divergence
    F = (x**2 * y, y * z, z * x)
    c03 = sum(sp.diff(Fi, v) for Fi, v in zip(F, (x, y, z))).subs({x: 1, y: 1, z: 1})

    # c04 Lagrange: maximize xy on x+y=10
    c04 = max(sp.Rational(a) * (10 - a) for a in range(0, 11))

    # c05 Green: ∮(-y dx + x dy) = 2 * area of the unit disk
    c05 = 2 * sp.pi * 1**2

    # c06 volume of revolution, disks
    c06 = sp.pi * sp.integrate((x**2)**2, (x, 0, 1))

    c07 = max(sp.Matrix([[4, 1], [2, 3]]).eigenvals().keys())
    c08 = sp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det()
    c09 = sp.Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]]).rank()
    c10 = (A * A).trace()
    c11 = 7 - 3                      # rank-nullity: domain dim 7 (5x7 acts on R^7), rank 3
    c12 = sp.Matrix([[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]).det()

    # c13 y'' - 5y' + 6y = 0, y(0)=1, y'(0)=0  ->  A e^{2t} + B e^{3t}
    Ac, Bc = sp.symbols("A B")
    sol = sp.solve([Ac + Bc - 1, 2 * Ac + 3 * Bc], [Ac, Bc])
    c13 = sol[Ac]

    # c14 y' + 2y = e^x with y_p = A e^x
    a = sp.Symbol("a")
    c14 = sp.solve(sp.Eq(sp.diff(a * sp.exp(x), x) + 2 * a * sp.exp(x), sp.exp(x)), a)[0]

    # c15 dy/dx = y^2, y(0)=1  ->  y = 1/(1-x)
    fn = sp.Function("f")
    gen = sp.dsolve(sp.Eq(sp.diff(fn(x), x), fn(x)**2), fn(x), ics={fn(0): 1})
    c15 = gen.rhs.subs(x, sp.Rational(1, 2))

    # c16 Wronskian of e^{2t}, e^{3t} at 0
    f1, f2 = sp.exp(2 * t), sp.exp(3 * t)
    c16 = sp.simplify(f1 * sp.diff(f2, t) - f2 * sp.diff(f1, t)).subs(t, 0)

    # c17 radius of convergence of sum n!/n^n x^n
    an = sp.factorial(n) / n**n
    c17 = sp.limit(an / (an.subs(n, n + 1)), n, sp.oo)

    c18 = sp.summation(1 / (n * (n + 1)), (n, 1, sp.oo))
    c19 = sp.summation(n / 2**n, (n, 0, sp.oo))
    c20 = sp.limit((sp.exp(x) - 1 - x) / x**2, x, 0)
    c21 = sp.series(sp.log(1 + x**2), x, 0, 6).removeO().coeff(x, 4)
    c22 = sp.integrate(1 / (1 + x**2), (x, 0, sp.oo))

    c23 = sp.Rational(10) * sp.Rational(3, 10) * (1 - sp.Rational(3, 10))
    c24 = sp.exp(-2)
    # c25 Bayes
    prev, sens, spec = sp.Rational(1, 100), sp.Rational(99, 100), sp.Rational(95, 100)
    c25 = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    c26 = 1 / sp.Rational(1, 6)                    # geometric mean of trials
    c27 = sp.integrate((x - sp.Rational(1, 2))**2, (x, 0, 1))

    c28 = sp.Abs(1 + sp.I)**8
    # c29 elements of order exactly 2 in Z/12Z
    c29 = len([i for i in range(1, 12) if (2 * i) % 12 == 0])
    c30 = sp.divisor_count(12)

    return {
        "c01": c01, "c02": c02, "c03": c03, "c04": c04, "c05": c05, "c06": c06,
        "c07": c07, "c08": c08, "c09": c09, "c10": c10, "c11": c11, "c12": c12,
        "c13": c13, "c14": c14, "c15": c15, "c16": c16, "c17": c17, "c18": c18,
        "c19": c19, "c20": c20, "c21": c21, "c22": c22, "c23": c23, "c24": c24,
        "c25": c25, "c26": c26, "c27": c27, "c28": c28, "c29": c29, "c30": c30,
    }


def check(path: str, computed: dict[str, sp.Expr]) -> list[str]:
    """Compare the file's answers against the recomputed ones. Returns failure strings."""
    problems = []
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    seen = set()
    for item in items:
        qid = item["id"]
        seen.add(qid)
        if qid not in computed:
            problems.append(f"{qid}: no independent computation — answer key unverified")
            continue
        want = sp.N(computed[qid])
        got = sp.N(sp.sympify(item["answer"]))
        # The stated answer is often a 4dp rounding of an irrational, so compare at that
        # precision rather than exactly.
        if abs(float(want) - float(got)) > 5e-4:
            problems.append(f"{qid}: file says {item['answer']}, computed {float(want):.6f}")
            continue
        for alias in item.get("aliases", []):
            try:
                av = float(sp.N(sp.sympify(alias)))
            except (sp.SympifyError, TypeError, ValueError):
                problems.append(f"{qid}: alias {alias!r} does not parse")
                continue
            if abs(av - float(want)) > 5e-4:
                problems.append(f"{qid}: alias {alias!r} = {av:.6f} != {float(want):.6f}")

    for qid in computed:
        if qid not in seen:
            problems.append(f"{qid}: computed but absent from {os.path.basename(path)}")
    return problems, len(items)


def main():
    total_problems = []
    for fname, computed in (
        ("reasoning_set_easy.jsonl", easy_answers()),
        ("reasoning_set_college.jsonl", college_answers()),
    ):
        path = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(path):
            print(f"skip {fname} (not present)")
            continue
        problems, count = check(path, computed)
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"{fname}: {count} questions — {status}")
        for p in problems:
            print(f"   ✗ {p}")
        total_problems += problems

    if total_problems:
        print(f"\n{len(total_problems)} problem(s). The answer key is not trustworthy yet.")
        sys.exit(1)
    print("\nEvery gold answer reproduced by an independent computation.")


if __name__ == "__main__":
    main()
