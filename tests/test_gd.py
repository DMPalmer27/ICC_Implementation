"""
File: test_gd.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Checks for Phase 1 of docs/GD_LINEAR_REGRESSION.md -- linear regression by
    gradient descent through the scheme on synthetic data. Section numbers refer to that
    document's section 7 test list; the leakage tests (3-5) belong to Phase 2.

      1. trajectory exactness: scheme GD == plaintext integer GD, bit for bit, with and
         without the intercept, and every decoded gradient == the integer oracle
      2. the gate: every step is admitted; an over-budget model raises with the reason
      6. exact entropies: convolution route == brute-force enumeration
      7. sanity: the exact entropy respects the rigorous bounds' directions; the correlated
         route equals quantize.entropy_bounds
      8. hygiene: the trainer takes no plaintext; the client keeps none after upload
      +  the intercept's query and bound; the hand-written linear algebra; the metrics;
         the source's samples match its exact pmf; the design schedule uses no sample

Usage:
    .venv/bin/python tests/test_gd.py
"""

import conftest  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from query import QueryStatus, check_query, least_squares_gradient  # noqa: E402
from quantize import (FixedPointSpec, QueryBudget, choose_prime, entropy_bounds,  # noqa: E402
                      to_integer)
from training import (GDSchedule, run_gd, scheme_gradient,  # noqa: E402
                      train_linear_regression)

from gd_reference import (condition_number, float_gd, integer_gradient,  # noqa: E402
                          integer_gradient_fn, jacobi_eigh, least_squares_optimum,
                          min_norm_solve, steps_to_tolerance, suboptimality)
from gd_setup import design_schedule, upload  # noqa: E402
from synthetic_source import (FAMILIES, SourceSpec, discretised_pmf,  # noqa: E402
                              exact_entropies, exact_entropies_bruteforce)

_PASSED, _FAILED = [], []


def check(name: str, condition: bool, detail: str = ""):
    """Records the outcome of one assertion."""
    (_PASSED if condition else _FAILED).append(name)
    print(f"  {'✓' if condition else '✗'} {name}" + (f"  ({detail})" if detail and not condition
                                                    else ""))


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 66 - len(title)))


# ─────────────────────────────────────────────────────────────────────────────

def test_intercept_query():
    section("Intercept: query polynomial and magnitude bound")
    xs, ws, bs = FixedPointSpec(7, 4.0), FixedPointSpec(4, 2.0), FixedPointSpec(4, 4.0)
    agree = True
    for n_s, P in ((25, 3), (50, 1), (10, 9), (6, 10), (10, 3)):
        B = QueryBudget(n_s, P, 2, xs, ws, b_spec=bs).magnitude_bound()
        worst = max(least_squares_gradient(n_s, P, [ws.max_int] * P, ws.f, j,
                                           bias_code=bs.max_int, f_X=xs.f)
                    .worst_case_bound(xs.max_int) for j in range(P + 1))
        agree &= B == worst == n_s * (P + 1) * 2 ** 23
    check("B = n 2^23 = max over components of the query's own worst case, five sizes", agree)

    rng = np.random.default_rng(1)
    n_s, P = 4, 2
    Xc, yc = rng.integers(-60, 61, (n_s, P)), rng.integers(-60, 61, n_s)
    w, b = rng.integers(-20, 21, P), int(rng.integers(-30, 31))
    codes = list(Xc.ravel()) + list(yc)
    oracle = integer_gradient(Xc, yc, w, b, 4, 3)
    poly = [least_squares_gradient(n_s, P, w, 4, j, bias_code=b, f_X=3).evaluate_integers(codes)
            for j in range(P + 1)]
    check("polynomial components == the integer oracle (features and bias)", poly == oracle)
    no_b = [least_squares_gradient(n_s, P, w, 4, j).evaluate_integers(codes) for j in range(P)]
    check("without an intercept the query is unchanged", no_b == integer_gradient(
        Xc, yc, w, None, 4, 3))
    check("the bias component has degree 1, the feature components degree 2",
          least_squares_gradient(n_s, P, w, 4, P, bias_code=b, f_X=3).degree == 1
          and least_squares_gradient(n_s, P, w, 4, 0, bias_code=b, f_X=3).degree == 2)
    raised = False
    try:
        least_squares_gradient(n_s, P, w, 4, P)
    except ValueError:
        raised = True
    check("the bias component without a bias code is rejected", raised)
    raised = False
    try:
        QueryBudget(n_s, P, 2, xs, ws, b_spec=FixedPointSpec(5, 4.0)).magnitude_bound()
    except ValueError:
        raised = True
    check("an intercept at a different f than the weights is rejected", raised)


def test_linear_algebra():
    section("Hand-written linear algebra and metrics")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((9, 9))
    A = A @ A.T
    vals, V = jacobi_eigh(A)
    check("Jacobi reconstructs A and matches eigvalsh to 1e-10",
          np.abs(V @ np.diag(vals) @ V.T - A).max() < 1e-10
          and np.abs(np.sort(np.linalg.eigvalsh(A)) - vals).max() < 1e-10)
    X = rng.standard_normal((30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.3 + 0.1 * rng.standard_normal(30)
    w = least_squares_optimum(X, y, True)
    ref = np.linalg.lstsq(np.hstack([X, np.ones((30, 1))]), y, rcond=None)[0]
    check("least-squares optimum == numpy lstsq (full rank)", np.abs(w - ref).max() < 1e-9)
    Xw = rng.standard_normal((5, 9))
    yw = rng.standard_normal(5)
    wm = least_squares_optimum(Xw, yw, True)
    refm = np.linalg.pinv(np.hstack([Xw, np.ones((5, 1))])) @ yw
    check("minimum-norm solution == pseudo-inverse (P >= n_s)", np.abs(wm - refm).max() < 1e-9)
    ex = np.array([1.0, 0.5, 0.05, 0.004, 0.02, 0.003, 0.002])
    check("steps_to_tolerance is the first step after which it STAYS below",
          steps_to_tolerance(ex, 1e-2) == 5 and steps_to_tolerance(ex, 1e-1) == 2
          and steps_to_tolerance(ex, 1e-3) is None)


def test_source():
    section("Synthetic source: pmfs, samples, design")
    ok_mass, ok_sample = True, True
    rng = np.random.default_rng(0)
    for fam, marginal in FAMILIES.items():
        spec = FixedPointSpec(3, 4.0)
        pmf = discretised_pmf(marginal, spec)
        ok_mass &= abs(pmf.sum() - 1) < 1e-12 and (pmf >= 0).all()
        draws = to_integer(marginal.sampler(rng, (200_000,)), spec)
        emp = np.bincount(draws + spec.max_int, minlength=spec.levels) / len(draws)
        tv = 0.5 * np.abs(emp - pmf).sum()
        ok_sample &= tv < 0.01
        if tv >= 0.01:
            print(f"    {fam}: total variation {tv:.4f}")
    check("every family's discretised pmf is a distribution", ok_mass)
    check("200k samples match the exact pmf (total variation < 0.01), every family", ok_sample)

    spec = SourceSpec(family="exponential", n_samples=25, n_features=3)
    w, b, sigma = spec.weights
    grid = 2 ** spec.grid_bits
    check("w* and b* lie on the dyadic grid", np.all(w * grid == np.round(w * grid))
          and b * grid == round(b * grid))
    s1 = spec.sample(np.random.default_rng(5))
    s2 = spec.sample(np.random.default_rng(5))
    check("sampling is reproducible from the rng",
          np.array_equal(s1["codes"], s2["codes"]))
    sched, info = design_schedule(spec)
    sched2, info2 = design_schedule(SourceSpec(family="exponential", n_samples=25, n_features=3))
    check("the schedule is a function of the design alone (same spec, same eta and T)",
          sched == sched2 and "kappa_pop" in info)
    check("exponential (mean ~1) has kappa_pop well above the centred families",
          info["kappa_pop"] > 5 * design_schedule(SourceSpec())[1]["kappa_pop"])


def test_exact_entropy():
    section("Exact entropies (test 6) and bound directions (test 7)")
    worst_block, worst_label, maxr_ok, exact_eq = 0.0, 0.0, True, True
    for fam in ("gauss", "exponential", "spiked", "discrete3", "student3", "uniform"):
        for noise in ("gauss", "laplace"):
            for P in (1, 2):
                spec = SourceSpec(family=fam, n_samples=2 if P == 2 else 3, n_features=P,
                                  f_X=1, clip_X=2.0, grid_bits=2, noise=noise)
                r = 3 if P == 2 else 2
                e = exact_entropies(spec, 2, r)
                bf = exact_entropies_bruteforce(spec, 2, r)
                worst_block = max(worst_block, abs(e.H_block_bits - bf["H_block"]))
                worst_label = max(worst_label, abs(e.h_label_bits - bf["h_y"]))
                maxr_ok &= e.max_R_bits >= bf["max_R"] - 1e-9
                if e.max_R_exact:
                    exact_eq &= abs(e.max_R_bits - bf["max_R"]) < 1e-9
    check("H_p(block) == brute force (24 tiny sources)", worst_block < 1e-10,
          f"{worst_block:.2e}")
    check("H_p(label) == brute force", worst_label < 1e-10, f"{worst_label:.2e}")
    check("max_R is never BELOW the brute-force maximum (upper-bound direction)", maxr_ok)
    check("when flagged exact, max_R equals the brute-force maximum", exact_eq)
    spec3 = SourceSpec(family="gauss", n_samples=2, n_features=2, f_X=1, clip_X=2.0,
                       grid_bits=2)
    e3 = exact_entropies(spec3, 3, 3)
    b3 = exact_entropies_bruteforce(spec3, 3, 3)
    check("order p = 3 agrees too", abs(e3.H_block_bits - b3["H_block"]) < 1e-10)

    spec = SourceSpec()
    e = spec.entropies()
    q = choose_prime(QueryBudget(25, 3, 2, spec.x_spec, FixedPointSpec(4, 2.0),
                                 b_spec=FixedPointSpec(4, 4.0)).magnitude_bound(),
                     warn_native=False)
    lo, hi = entropy_bounds(spec.n, 5, q, spec.x_spec)
    H, mR = e.in_log_q(q, spec.n)
    check("exact H_p(X) >= the entropy_bounds lower bound (case-study source)", H >= lo,
          f"{H:.3f} vs {lo:.3f}")
    check("exact max_R <= the assumption-free upper bound", mR <= hi, f"{mR:.3f} vs {hi:.3f}")
    corr = SourceSpec(rho=0.9)
    ec = corr.entropies()
    lo_c, hi_c = entropy_bounds(corr.n, 5, q, corr.x_spec,
                                joint_log2_sup_density=corr.joint_log2_sup_density())
    check("correlated route == quantize.entropy_bounds (joint form)",
          abs(ec.in_log_q(q, corr.n)[0] - lo_c) < 1e-9 and abs(ec.in_log_q(q, corr.n)[1] - hi_c) < 1e-9)
    check("correlation lowers the entropy bound",
          corr.entropies().H_X_bits < SourceSpec(rho=0.5).entropies().H_X_bits)


def test_trajectory_exact():
    section("Training through the scheme (tests 1, 2, 8)")
    for intercept in (True, False):
        spec = SourceSpec(family="exponential" if intercept else "gauss", n_samples=4,
                          n_features=2)
        sched, _ = design_schedule(spec, steps=15)
        if not intercept:
            sched = GDSchedule(eta=sched.eta, steps=15, w_spec=sched.w_spec, f_X=sched.f_X)
        up = upload(spec, [7, int(intercept)])
        enc = train_linear_regression(up.client, up.server, sched, 4, 2)
        ref = run_gd(sched, 4, 2, integer_gradient_fn(up.data["X_codes"], up.data["y_codes"],
                                                      sched))
        tag = "with" if intercept else "without"
        check(f"{tag} intercept: weight codes identical every step",
              np.array_equal(enc.w_codes, ref.w_codes))
        check(f"{tag} intercept: float iterates identical every step (bit for bit)",
              np.array_equal(enc.w, ref.w))
        check(f"{tag} intercept: decoded gradients == integer oracle every step",
              enc.grad == ref.grad)

    # Gate: a model outside the declared box must raise, not decode a wrapped value
    spec = SourceSpec(n_samples=4, n_features=2)
    sched, _ = design_schedule(spec, steps=3)
    up = upload(spec, [8])
    grad = scheme_gradient(up.client, up.server, 4, 2, sched)
    big = np.array([10 ** 7, -10 ** 7])
    raised = ""
    try:
        grad(big, 0)
    except RuntimeError as exc:
        raised = str(exc)
    check("an over-budget query raises with the server's reason", "may_overflow" in raised)
    worst = [least_squares_gradient(4, 2, [sched.w_spec.max_int] * 2, 4, j,
                                    bias_code=sched.b_spec.max_int, f_X=7) for j in range(3)]
    check("a model at the clip box corner is admitted by the gate",
          all(check_query(f, up.client.context.q, 2, spec.x_spec.max_int).ok for f in worst))

    params = list(inspect.signature(train_linear_regression).parameters)
    check("the trainer's only inputs are client, server, schedule, sizes and its rng",
          params == ["client", "server", "schedule", "n_samples", "n_features", "rng"])
    leaked = [k for k, v in vars(up.client).items()
              if isinstance(v, np.ndarray) and v.size == spec.n]
    check("the client holds no length-n array after upload (only k, of length m)",
          not leaked, str(leaked))


def test_float_reference():
    section("Unencoded baseline")
    spec = SourceSpec(n_samples=25, n_features=3)
    data = spec.sample(np.random.default_rng(11))
    sched, _ = design_schedule(spec)
    wf, times = float_gd(data["X_real"], data["y_real"], sched)
    ex = suboptimality(data["X_real"], data["y_real"], wf, True)
    check("float GD from the design schedule converges (suboptimality < 1e-8)", ex[-1] < 1e-8,
          f"{ex[-1]:.2e}")
    check("suboptimality starts at exactly 1", abs(ex[0] - 1) < 1e-12)
    check("float steps take microseconds, not seconds", np.median(times) < 1e-3)


def main():
    t0 = time.perf_counter()
    for test in (test_intercept_query, test_linear_algebra, test_source, test_exact_entropy,
                 test_trajectory_exact, test_float_reference):
        test()
    print(f"\n{'=' * 72}\n  {len(_PASSED)} passed, {len(_FAILED)} failed in "
          f"{time.perf_counter() - t0:.1f}s\n{'=' * 72}")
    for name in _FAILED:
        print("  FAILED:", name)
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
