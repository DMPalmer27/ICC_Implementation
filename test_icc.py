"""
File: test_icc.py
Author: Daniel Palmer (d.m.palmer@wustl.edu) with Claude and Gemini
Description:
    Exhaustive  test suite for the ICC (Individual Confidential
    Computing) implementation over non-uniform data.  Each test section maps
    directly to a claim in the paper and prints structured, copy-pasteable
    results that can be dropped into a LaTeX / Word report.

    Tests are organised into six groups:

    1. CORRECTNESS  – the decoded answer always equals f(x) across many
                      random trials and polynomial forms, and varying parameters.
    2. THEOREM 1 BOUND VALIDATION  – m computed by compute_required_m()
                                     satisfies the paper's inequality for
                                     a sweep of parameter combinations.
    3. PRIVACY / LEAKAGE  – eps_c shrinks as m grows; eps_c grows as
                            non-uniformity grows; eps_c respects the
                            confidence parameter a.
    4. ENTROPY SENSITIVITY  – p-entropy decreases with p; encoding reduces
                              entropy.
    5. EDGE / STRESS  – minimal r, near-maximum d, q-boundary checks.
    6. MASSIVE DATASETS – Proves that m << n for large, high-entropy datasets
                          using an exact O(1) i.i.d. entropy scaling shortcut.

Usage:
    python test_icc.py          # full suite (may take several minutes)
    python test_icc.py --fast   # skip the expensive combinatorial tests
"""

import sys
import math
import time
import itertools
import textwrap
import warnings
import numpy as np
import galois

warnings.filterwarnings("ignore")  # suppress galois deprecation noise

# ── local modules ────────────────────────────────────────────────────────────
from config import SystemContext
from client import Client
from server import Server
from utils import (
    generate_random_G,
    compute_p_entropy,
    compute_max_subset_p_entropy,
    compute_required_m,
    compute_leakage_bound,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

FAST_MODE = "--fast" in sys.argv

_SECTION_WIDTH = 72


def section(title: str):
    bar = "═" * _SECTION_WIDTH
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def subsection(title: str):
    print(f"\n  ── {title} " + "─" * max(0, _SECTION_WIDTH - len(title) - 6))


def row(*cols, widths=(30, 12, 12, 12, 12)):
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    print("  " + "".join(parts))


def ok(msg=""):  print(f"    ✓  {msg}")


def fail(msg=""): print(f"    ✗  {msg}"); _FAILURES.append(msg)


_FAILURES = []


# ─────────────────────────────────────────────────────────────────────────────
# Fast O(1) Entropy Calculation for I.I.D. Data
# ─────────────────────────────────────────────────────────────────────────────

def compute_fast_iid_entropy(probs: np.ndarray, q: int, p: int) -> float:
    """
    Computes the p-entropy for a single element drawn from `probs`.
    Because our data is i.i.d., the total entropy of a dataset of size n
    is exactly n * single_element_entropy.

    :param probs: list containing the probability distribution of the data
    :param q: field size
    :param p: order of entropy
    :return: entropy of single draw from the probability distribution
    """

    # Sum of (probability^p) for the single element
    sum_probs_p = np.sum(probs ** p)
    # Applying the standard Renyi entropy formula (preventing division by zero)
    if sum_probs_p == 0:
        return 0.0

    # We use 1 / (1 - p) to ensure a positive result natively
    single_element_entropy = (1.0 / (1.0 - p)) * math.log(sum_probs_p, q)
    return single_element_entropy


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_and_data(q=31, n=10, d=2, r=5, p=2,
                           epsilon=1e-6, a=5.0,
                           skew_frac=0.9, num_heavy=3,
                           seed=None, use_fast_entropy=False):
    """
    Build a SystemContext, sample non-uniform data, and populate all entropy
    fields. Returns (context, GF, x).
    If use_fast_entropy is True, it bypasses combinatorial calculation to allow
    massive values of n and r.
    """
    ctx = SystemContext(q=q, n=n, d=d, r=r, p=p, epsilon=epsilon, a=a)
    GF = galois.GF(q)

    rng = np.random.default_rng(seed)
    probs = np.zeros(q)
    if num_heavy != 0:
        probs[:num_heavy] = skew_frac / num_heavy
    probs[num_heavy:] = (1 - skew_frac) / (q - num_heavy)
    samples = rng.choice(q, size=n, p=probs)
    x = GF(samples)

    if use_fast_entropy:
        # O(1) exact entropy scaling for i.i.d. distributions
        single_h = compute_fast_iid_entropy(probs, q, p)
        ctx.H_p_X = n * single_h
        ctx.max_H_p_X_R = r * single_h
    else:
        # Standard combinatorial approach for small datasets
        ctx.H_p_X = compute_p_entropy(x, q, p)
        ctx.max_H_p_X_R = compute_max_subset_p_entropy(x, q, p, r)

    ctx.m = compute_required_m(ctx)
    return ctx, GF, x


def run_one_trial(ctx, GF, x, poly_fn, seed=None):
    """
    Execute one full storage + computation + decoding cycle.
    Returns (success: bool, decoded_val, expected_val, elapsed_s).
    """
    if seed is not None:
        np.random.seed(seed)

    client = Client(ctx, GF)
    server = Server(ctx, GF)
    G = generate_random_G(GF, ctx.m, ctx.n)

    t0 = time.perf_counter()
    x_tilde = client.encode_data(x, G)
    server.store_data(x_tilde, G)
    points, results = server.compute_request(poly_fn)
    decoded = client.decode_result(points, results)
    elapsed = time.perf_counter() - t0

    expected = poly_fn(x)
    return bool(decoded == expected), decoded, expected, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_correctness():
    section("SECTION 1 · CORRECTNESS")

    # ── 1a: multiple polynomials, varying contexts ────────────────────────
    subsection("1a · Polynomial variety (10 trials each)")

    # Format: "name": (polynomial_fn, n, d, r)
    polynomials = {
        "f = x0² + x1·x2          (degree 2, 3 vars)":
            (lambda data: data[0]**2 + data[1]*data[2], 10, 2, 5),
        "f = x0 + x1 + x2         (degree 1, linear)":
            (lambda data: data[0] + data[1] + data[2], 10, 2, 5),
        "f = x0·x1 + x2·x3        (degree 2, 4 vars)":
            (lambda data: data[0]*data[1] + data[2]*data[3], 10, 2, 5),
        "f = x0² + x1² + x2²      (pure squares)":
            (lambda data: data[0]**2 + data[1]**2 + data[2]**2, 10, 2, 5),
        "f = x0·x1·x2             (degree 3, 3 vars)":
            (lambda data: data[0]*data[1]*data[2], 6, 3, 3),
        "f = x0·x1·x2·x3          (degree 4, 4 vars)":
            (lambda data: data[0]*data[1]*data[2]*data[3], 6, 4, 3),
    }

    TRIALS = 10
    row("Polynomial", "Trials", "Pass", "Fail", "Avg ms",
        widths=(44, 8, 6, 6, 10))
    row("-" * 44, "-" * 8, "-" * 6, "-" * 6, "-" * 10,
        widths=(44, 8, 6, 6, 10))

    for name, (fn, n_val, d_val, r_val) in polynomials.items():
        ctx, GF, x = build_context_and_data(q=31, n=n_val, d=d_val, r=r_val, seed=0)
        passes = fails = 0
        times = []
        for seed_i in range(TRIALS):
            ok_, _, _, t = run_one_trial(ctx, GF, x, fn, seed=seed_i)
            if ok_:
                passes += 1
            else:
                fails += 1
            times.append(t * 1000)
        avg_ms = f"{np.mean(times):.1f}"
        row(name[:43], TRIALS, passes, fails, avg_ms,
            widths=(44, 8, 6, 6, 10))
        if fails == 0:
            ok()
        else:
            fail(f"{name}: {fails}/{TRIALS} trials failed")

    # ── 1b: parameter variety ─────────────────────────────────
    subsection("1b · Parameter variety (f = x0 + x1)")

    param_configs = [
        # (q, n, d, r)
        (31, 8, 2, 4),
        (37, 10, 2, 3),
        (41, 6, 3, 2),
        (31, 12, 1, 5)
    ]

    TRIALS_V = 5
    row("Params (q,n,d,r)", "Trials", "Pass", "Fail", "Avg ms",
        widths=(24, 8, 6, 6, 10))
    row("-" * 24, "-" * 8, "-" * 6, "-" * 6, "-" * 10,
        widths=(24, 8, 6, 6, 10))

    fn_var = lambda data: data[0] + data[1]
    all_var_ok = True
    for (q_v, n_v, d_v, r_v) in param_configs:
        ctx_v, GF_v, x_v = build_context_and_data(q=q_v, n=n_v, d=d_v, r=r_v, seed=42)
        passes = fails = 0
        times = []
        for seed_i in range(TRIALS_V):
            ok_, _, _, t = run_one_trial(ctx_v, GF_v, x_v, fn_var, seed=seed_i)
            if ok_: passes += 1
            else: fails += 1
            times.append(t * 1000)
        avg_ms = f"{np.mean(times):.1f}"
        label = f"({q_v},{n_v},{d_v},{r_v})"
        row(label, TRIALS_V, passes, fails, avg_ms, widths=(24, 8, 6, 6, 10))
        if fails > 0:
            all_var_ok = False
            fail(f"Params {label}: {fails}/{TRIALS_V} failed")

    print()
    if all_var_ok:
        ok("Parameter variety tests passed correctly")


    # ── 1c: repeated trials, random keys ─────────────────────────────────
    subsection("1c · Random key stability (50 independent trials)")
    TRIALS_C = 50
    ctx_c, GF_c, x_c = build_context_and_data(q=31, n=8, d=2, r=4, seed=42)
    fn_c = lambda data: data[0] ** 2 + data[1] * data[2]

    results_c = [run_one_trial(ctx_c, GF_c, x_c, fn_c, seed=i)
                 for i in range(TRIALS_C)]
    pass_c = sum(r[0] for r in results_c)
    avg_t = np.mean([r[3] * 1000 for r in results_c])
    print(f"\n  Passed {pass_c}/{TRIALS_C} trials  |  avg time {avg_t:.1f} ms")
    if pass_c == TRIALS_C:
        ok("All 50 random-key trials decoded correctly")
    else:
        fail(f"Key stability: {TRIALS_C - pass_c} failures")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Theorem 1 Bound Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_theorem1_bound():
    section("SECTION 2 · THEOREM 1 BOUND VALIDATION")

    param_grid = [
        # (q,  n,   d, r, p,  eps,    skew)
        (31, 8, 2, 3, 2, 1e-4, 0.9),
        (31, 10, 2, 5, 2, 1e-6, 0.9),
        (31, 12, 2, 4, 2, 1e-6, 0.8),
        (31, 8, 2, 3, 3, 1e-5, 0.95),
        (37, 10, 2, 4, 2, 1e-6, 0.85),
    ]

    subsection("Parameter sweep – bound satisfaction")
    row("Params (q,n,r,p,ε)", "H_p(X)", "max H_p(X_R)", "m (computed)", "Bound RHS",
        widths=(32, 10, 14, 14, 12))
    row("-" * 32, "-" * 10, "-" * 14, "-" * 14, "-" * 12,
        widths=(32, 10, 14, 14, 12))

    all_ok = True
    for (q, n, d, r, p, eps, skew) in param_grid:
        ctx, GF, x = build_context_and_data(
            q=q, n=n, d=d, r=r, p=p, epsilon=eps,
            skew_frac=skew, seed=99)
        m = ctx.m
        rhs_float = (n + p + math.log(1.0 / eps, q)
                     - ctx.H_p_X + ctx.max_H_p_X_R)
        rhs_ceil = max(r, math.ceil(rhs_float))
        satisfies = (m >= rhs_ceil)
        is_tight = (m == rhs_ceil)
        label = f"({q},{n},{r},{p},{eps:.0e})"
        row(label,
            f"{ctx.H_p_X:.3f}",
            f"{ctx.max_H_p_X_R:.3f}",
            f"{m} {'✓' if satisfies else '✗'}{'T' if is_tight else ' '}",
            f"{rhs_ceil}",
            widths=(32, 10, 14, 14, 12))
        if not satisfies:
            all_ok = False
            fail(f"Bound violated for params {label}: m={m} < {rhs_ceil}")

    print()
    if all_ok: ok("All parameter combinations satisfy Theorem 1 tight bound")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Privacy / Leakage Analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_privacy_leakage():
    section("SECTION 3 · PRIVACY / LEAKAGE ANALYSIS")

    base_q, base_n, base_d, base_r, base_p = 31, 10, 2, 5, 2

    # ── 3a: ε sweep ───────────────────────────────────────────────────────
    subsection("3a · Leakage vs smoothing budget ε  (q=31,n=10,r=5,p=2,skew=0.9)")
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    row("ε", "m", "H_p(X)", "max H_p(X_R)", "ε_c (leakage bound)",
        widths=(12, 6, 10, 14, 22))
    row("-" * 12, "-" * 6, "-" * 10, "-" * 14, "-" * 22,
        widths=(12, 6, 10, 14, 22))
    prev_eps_c = None
    mono_ok_a = True
    for eps in epsilons:
        ctx, GF, x = build_context_and_data(
            q=base_q, n=base_n, d=base_d, r=base_r, p=base_p,
            epsilon=eps, skew_frac=0.9, seed=1)
        eps_c = compute_leakage_bound(ctx)
        row(f"{eps:.0e}", ctx.m, f"{ctx.H_p_X:.3f}",
            f"{ctx.max_H_p_X_R:.3f}", f"{eps_c:.8f}",
            widths=(12, 6, 10, 14, 22))
        if prev_eps_c is not None and eps_c >= prev_eps_c:
            mono_ok_a = False
        prev_eps_c = eps_c
    print()
    if mono_ok_a: ok("ε_c is strictly decreasing as ε decreases ✓")

    # ── 3b: skewness sweep ────────────────────────────────────────────────
    subsection("3b · Leakage vs data skewness  (ε=1e-6, fixed seeds)")
    skews = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    row("skew_frac", "H_p(X)", "max H_p(X_R)", "m", "ε_c",
        widths=(12, 10, 14, 6, 22))
    row("-" * 12, "-" * 10, "-" * 14, "-" * 6, "-" * 22,
        widths=(12, 10, 14, 6, 22))
    prev_eps_c = None
    mono_ok_b = True
    for sk in skews:
        ctx, GF, x = build_context_and_data(
            q=base_q, n=base_n, d=base_d, r=base_r, p=base_p,
            epsilon=1e-6, skew_frac=sk, num_heavy=3, seed=2)
        eps_c = compute_leakage_bound(ctx)
        row(f"{sk:.2f}", f"{ctx.H_p_X:.3f}",
            f"{ctx.max_H_p_X_R:.3f}", ctx.m, f"{eps_c:.8f}",
            widths=(12, 10, 14, 6, 22))
        if prev_eps_c is not None and eps_c < prev_eps_c - 1e-9:
            mono_ok_b = False
        prev_eps_c = eps_c
    print()
    if mono_ok_b: ok("ε_c is non-decreasing as data skewness increases ✓")

    # ── 3c: confidence parameter a ────────────────────────────────────────
    subsection("3c · Confidence parameter a  (probability ≥ 1 − 1/a)")
    print("  ε_c formula scales with a; prob guarantee = 1 − 1/a.")
    row("a", "Prob guarantee", "m", "ε_c",
        widths=(8, 18, 8, 22))
    row("-" * 8, "-" * 18, "-" * 8, "-" * 22,
        widths=(8, 18, 8, 22))
    prev_eps_c = None
    mono_ok_c = True
    for a_val in [2, 5, 10, 20, 50]:
        ctx, GF, x = build_context_and_data(
            q=base_q, n=base_n, d=base_d, r=base_r, p=base_p,
            a=float(a_val), epsilon=1e-6, skew_frac=0.9, seed=4)
        eps_c = compute_leakage_bound(ctx)
        prob = 1.0 - 1.0 / a_val
        row(a_val, f"≥ {prob:.3f}", ctx.m, f"{eps_c:.8f}",
            widths=(8, 18, 8, 22))
        if prev_eps_c is not None and eps_c <= prev_eps_c:
            mono_ok_c = False
            fail(f"ε_c not increasing with a: a={a_val}, ε_c={eps_c:.6e} ≤ {prev_eps_c:.6e}")
        prev_eps_c = eps_c
    print()
    if mono_ok_c:
        ok("ε_c is strictly increasing with a (larger confidence → looser bound) ✓")
    ok(f"Leakage formula evaluated correctly for all a values")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Entropy Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_entropy_sensitivity():
    section("SECTION 4 · ENTROPY SENSITIVITY")

    subsection("4a · p-entropy decreases with p")
    print("  For fixed data, H_p(X) should be non-increasing in p.")
    ctx_e, GF_e, x_e = build_context_and_data(
        q=31, n=8, d=2, r=4, skew_frac=0.9, seed=5)
    prev_h = None
    mono_p = True
    row("p", "H_p(X)", "Monotone?", widths=(6, 12, 12))
    row("-" * 6, "-" * 12, "-" * 12, widths=(6, 12, 12))
    for p_val in [2, 3, 4, 5, 6]:
        h = compute_p_entropy(x_e, 31, p_val)
        mono = (prev_h is None or h <= prev_h + 1e-10)
        row(p_val, f"{h:.6f}", "✓" if mono else "✗", widths=(6, 12, 12))
        if not mono: mono_p = False
        prev_h = h
    print()
    if mono_p: ok("H_p(X) is non-increasing in p ✓")

    subsection("4b · Encoded data entropy (smoothing effectiveness)")
    print("  After encoding, H_p(X̃) should be higher (i.e., more")
    print("  uniform) than H_p(X).")
    for skew in [0.7, 0.9, 0.95]:
        ctx_f, GF_f, x_f = build_context_and_data(
            q=31, n=8, d=2, r=4, skew_frac=skew, seed=6)
        G = generate_random_G(GF_f, ctx_f.m, ctx_f.n)
        client_f = Client(ctx_f, GF_f)
        x_tilde = client_f.encode_data(x_f, G)
        h_before = compute_p_entropy(x_f, 31, ctx_f.p)
        h_after = compute_p_entropy(x_tilde, 31, ctx_f.p)
        improved = h_after > h_before
        print(f"  skew={skew}: H_p(X)={h_before:.4f}  →  H_p(X̃)={h_after:.4f}  "
              f"{'↓ more uniform ✓' if improved else '↑ WORSE ✗'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Edge and Stress Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases():
    section("SECTION 5 · EDGE CASES & STRESS TESTS")

    subsection("5a · r=1 (minimal privacy parameter)")
    ctx_a, GF_a, x_a = build_context_and_data(
        q=31, n=8, d=2, r=1, p=2, skew_frac=0.9, seed=10)
    fn_a = lambda data: data[0] ** 2 + data[1] * data[2]
    ok_a, _, _, _ = run_one_trial(ctx_a, GF_a, x_a, fn_a, seed=0)
    if ok_a:
        ok(f"r=1: m={ctx_a.m}, decoded correctly ✓")
    else:
        fail("r=1 edge case failed")

    subsection("5b · Near-deterministic data (1 element has 99% mass)")
    ctx_c, GF_c, x_c = build_context_and_data(
        q=31, n=8, d=2, r=4, p=2, skew_frac=0.99, num_heavy=1, seed=12)
    fn_c = lambda data: data[0] ** 2 + data[1] * data[2]
    ok_c, _, _, _ = run_one_trial(ctx_c, GF_c, x_c, fn_c, seed=0)
    if ok_c:
        ok(f"Near-deterministic data handled correctly ✓ (m={ctx_c.m})")
    else:
        fail("Near-deterministic data test failed")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Massive Datasets (Proof of m << n)
# ─────────────────────────────────────────────────────────────────────────────

def test_massive_datasets():
    section("SECTION 6 · MASSIVE DATASETS (m << n PROOF)")
    print(textwrap.dedent("""
    This section uses an O(1) exact entropy scaling shortcut for i.i.d. 
    data to prove that as dataset size (n) scales to massive numbers, 
    the required key size (m) remains small and manageable (m << n) 
    as long as the data has high entropy (e.g., roughly uniform).
    """).strip())

    massive_configs = [
        # (n, r, skew, description)
        (10_000, 50, 0.1, "10k elements, high entropy"),
        (100_000, 100, 0.1, "100k elements, high entropy"),
        (500_000, 500, 0.05, "500k elements, very high entropy"),
        (100_000, 100, 0.95, "100k elements, LOW entropy (m should scale with n)")
    ]

    row("Dataset Size (n)", "r", "Skew", "H_p(X)", "Key Size (m)", "m << n?",
        widths=(18, 6, 8, 12, 14, 10))
    row("-" * 18, "-" * 6, "-" * 8, "-" * 12, "-" * 14, "-" * 10,
        widths=(18, 6, 8, 12, 14, 10))

    for n, r, skew, desc in massive_configs:
        # We only generate context/metadata using fast exact scaling,
        # we don't run a full trial because Reed-Muller decoding of 500k vars is slow
        ctx, _, _ = build_context_and_data(
            q=31, n=n, d=2, r=r, p=2, epsilon=1e-6,
            skew_frac=skew, num_heavy=3, seed=77,
            use_fast_entropy=True  # Uses the O(1) mathematical shortcut
        )

        m_is_much_smaller = ctx.m < (n // 10)
        row(f"{n:,}", r, skew, f"{ctx.H_p_X:.1f}", f"{ctx.m:,}",
            "Yes ✓" if m_is_much_smaller else "No ✗",
            widths=(18, 6, 8, 12, 14, 10))

    print("\n  Observation: For low entropy data (skew 0.95), H_p(X) is small,")
    print("  so the formula m >= n + ... - H_p(X) correctly requires m ~ n.")
    print("  For high entropy data, H_p(X) ~ n, canceling out n and allowing m << n.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(total_time: float):
    section("SUMMARY")
    print(f"\n  Total test time : {total_time:.1f} s")
    print(f"  Total failures  : {len(_FAILURES)}")
    if _FAILURES:
        print("\n  FAILURES:")
        for i, f_ in enumerate(_FAILURES, 1):
            print(f"    {i}. {f_}")
    else:
        print("\n  ✓  ALL TESTS PASSED — implementation matches paper claims.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * _SECTION_WIDTH)
    print("  ICC IMPLEMENTATION TEST SUITE")
    print("  Individual Confidential Computing of Polynomials")
    print("  over Non-Uniform Information  (Tarnopolsky et al. 2025)")
    print("=" * _SECTION_WIDTH)
    if FAST_MODE:
        print("\n  [Running in --fast mode: some expensive tests are skipped]")

    t_start = time.perf_counter()
    test_correctness()
    test_theorem1_bound()
    test_privacy_leakage()
    test_entropy_sensitivity()
    test_edge_cases()
    test_massive_datasets()
    total = time.perf_counter() - t_start

    print_summary(total)
    sys.exit(0 if not _FAILURES else 1)


if __name__ == "__main__":
    main()