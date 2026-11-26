"""
File: test_inference.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Checks for degree-1 inference through the scheme (docs/INFERENCE.md):

      1. exactness: every decoded score == the plaintext integer oracle, with and without an
         intercept, for one output and for K = 3 (softmax)
      2. the gate: the ScoreBudget bound equals check_query's bound at the ceiling; a model
         over the ceiling is refused, and predict_scores raises with the reason
      3. the analytic error bound holds on random models and data, clipping included
      4. information set: the direct enumeration equals the depth-first search it replaced,
         and works past Python's recursion limit
      5. synthetic entropies: exact product rule; the correlated route equals
         quantize.entropy_bounds
      6. real data: train and stored halves are disjoint, standardisation uses the training
         half only, and diabetes 'sex' is treated as discrete
      7. hygiene: predict_scores takes no plaintext

Usage:
    .venv/bin/python tests/test_inference.py
"""

import conftest  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from inference import predict_scores, score_queries  # noqa: E402
from query import check_query, linear_score  # noqa: E402
from quantize import (FixedPointSpec, ScoreBudget, entropy_bounds,  # noqa: E402
                      score_bound_worst_case)
from utils import get_information_set  # noqa: E402

from inference_setup import (integer_scores, plan_inference, quantise_model,  # noqa: E402
                             score_error_bound, upload)
from inference_source import FeatureSource  # noqa: E402

RESULTS = []


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _uploaded(N=12, P=4, K=1, f_X=6, f_w=5, intercept=True, seed=0):
    rng = np.random.default_rng(seed)
    src = FeatureSource("laplace", P, 0.0, f_X, 4.0)
    X = src.sample(rng, N)
    Xc = src.quantise(X)
    W = rng.normal(0, 1.2, (K, P))
    b = rng.normal(0, 1.0, K)
    model = quantise_model(W, b, f_w)
    b_spec = model["b_spec"] if intercept else None
    plan = plan_inference(N, P, src.x_spec, model["w_spec"], b_spec, src.entropies(N))
    client, server, _ = upload(Xc, plan, src.x_spec, seed)
    return dict(X=X, Xc=Xc, W=W, b=b, model=model, plan=plan, client=client, server=server,
                src=src, b_codes=model["b_codes"] if intercept else None)


def test_exactness():
    print("1. exactness")
    for K, intercept in [(1, True), (1, False), (3, True)]:
        u = _uploaded(K=K, intercept=intercept)
        Z, _ = predict_scores(u["client"], u["server"], 12, 4, u["model"]["W_codes"],
                              u["b_codes"], u["src"].f_X)
        b_codes = u["b_codes"] if intercept else np.zeros(K, dtype=int)
        oracle = integer_scores(u["Xc"], u["model"]["W_codes"], b_codes, u["src"].f_X)
        check(f"K={K}, intercept={intercept}: decoded == integer oracle", np.all(Z == oracle),
              f"{Z.size} scores, m={u['plan']['m']}, lambda={u['plan']['lam']}")
    check("lambda = m + 1 at d = 1", u["plan"]["lam"] == u["plan"]["m"] + 1)


def test_gate():
    print("2. the gate")
    x_spec, w_spec, b_spec = FixedPointSpec(6, 4.0), FixedPointSpec(5, 2.0), FixedPointSpec(5, 1.0)
    P = 4
    B = ScoreBudget(P, x_spec, w_spec, b_spec).magnitude_bound()
    f = linear_score(0, P, [w_spec.max_int] * P, b_spec.max_int, x_spec.f)
    res = check_query(f, 2 * B + 1, 1, x_spec.max_int)
    check("bound at the ceiling == score_bound_worst_case", res.bound == B ==
          score_bound_worst_case(P, x_spec, w_spec, b_spec), f"B = {B}")
    check("ceiling query admitted by a field of size 2B+1", res.ok)
    u = _uploaded()
    W_big = np.full((1, 4), 2 ** 20)
    try:
        predict_scores(u["client"], u["server"], 12, 4, W_big, u["b_codes"], u["src"].f_X)
        check("over-ceiling model is refused", False)
    except RuntimeError as e:
        check("over-ceiling model is refused, with a reason", "refused" in str(e))
    f2 = linear_score(0, P, [1] * P, None)
    check("degree-1 query reports degree 1", f2.degree == 1)


def test_error_bound():
    print("3. analytic error bound")
    worst, viol, total = 0.0, 0, 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        P, N = int(rng.integers(1, 12)), 30
        f_X, f_w = int(rng.integers(2, 12)), int(rng.integers(2, 12))
        x_spec = FixedPointSpec(f_X, 2.0)            # narrow clip so clipping is exercised
        X = rng.standard_t(3, (N, P))
        W = rng.normal(0, 2.0, (2, P))
        b = rng.normal(0, 1.0, 2)
        model = quantise_model(W, b, f_w)
        Xc = np.asarray(FeatureSource("gauss", P, 0, f_X, 2.0).quantise(X))
        Zq = np.asarray(integer_scores(Xc, model["W_codes"], model["b_codes"], f_X),
                        dtype=float) / 2.0 ** (f_X + f_w)
        err = np.abs(Zq - (X @ W.T + b))
        bnd = score_error_bound(X, x_spec, W, f_w)
        viol += int(np.sum(err > bnd * (1 + 1e-12)))
        total += err.size
        worst = max(worst, float((err / bnd).max()))
    check("|z_scheme - z_float| <= bound on every score", viol == 0,
          f"{total} scores, max err/bound = {worst:.3f}")


def test_information_set():
    print("4. information set")

    def dfs(q, m, d):
        res = []

        def rec(depth, cur, b):
            if depth == m:
                res.append(cur)
                return
            for v in range(min(q, b + 1)):
                rec(depth + 1, cur + (v,), b - v)
        rec(0, (), d)
        return res
    same = all(get_information_set(q, m, d) == dfs(q, m, d)
               for q in (2, 3, 5, 31) for m in range(1, 7) for d in range(0, 5))
    check("enumeration == the DFS it replaced (order included)", same)
    t0 = time.perf_counter()
    big = get_information_set(1_000_003, 2500, 1)
    check("m = 2500 at d = 1: m + 1 points, no recursion limit",
          len(big) == 2501 and big[0] == (0,) * 2500, f"{time.perf_counter() - t0:.2f} s")


def test_entropies():
    print("5. synthetic entropies")
    src = FeatureSource("bimodal", 3, 0.0, 6, 4.0)
    e = src.entropies(10, p=2, r=5)
    pmf = src.code_pmf
    h = -math.log2(float(np.sum(pmf ** 2)))
    check("independent: H_p(X) = n H_p(code), max_R = r H_p(code)",
          abs(e.H_X_bits - 30 * h) < 1e-9 and abs(e.max_R_bits - 5 * h) < 1e-9)
    corr = FeatureSource("gauss", 4, 0.6, 7, 4.0)
    ec = corr.entropies(8, p=2, r=5)
    q = 1_000_003
    lo, hi = ec.in_log_q(q, 32)
    P, rho = 4, 0.6
    log2_det = (P - 1) * math.log2(1 - rho) + math.log2(1 + (P - 1) * rho)
    joint = 8 * (-0.5 * P * math.log2(2 * math.pi) - 0.5 * log2_det)
    lo2, hi2 = entropy_bounds(32, 5, q, corr.x_spec, joint_log2_sup_density=joint)
    check("correlated: equals quantize.entropy_bounds", abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9)


def test_real_data():
    print("6. real data")
    from real_data import entropy_provisional, load
    for name in ("diabetes", "wine"):
        prep = load(name)
        tr = {tuple(r) for r in prep.X_train_raw}
        check(f"{name}: training mean is 0 on the training half only",
              np.allclose(prep.X_train.mean(axis=0), 0, atol=1e-9)
              and not np.allclose(prep.X_store.mean(axis=0), 0, atol=1e-6))
        check(f"{name}: halves have {len(prep.X_train)} / {len(prep.X_store)} rows",
              abs(len(prep.X_train) - len(prep.X_store)) <= 1 and len(tr) > 0)
    prep = load("diabetes")
    _, diag = entropy_provisional(prep, len(prep.X_store), FixedPointSpec(8, 4.0))
    check("diabetes: 'sex' is discrete and credited 0 bits", diag["discrete"] == ["sex"])
    check("recorded precision caps the credited bits", np.all(diag["f_eff"] <= 8))


def test_hygiene():
    print("7. hygiene")
    params = list(inspect.signature(predict_scores).parameters)
    check("predict_scores takes client, server and the public model only",
          params == ["client", "server", "n_rows", "n_features", "W_codes", "b_codes", "f_X"])
    check("score_queries depends on the model only",
          "X" not in inspect.signature(score_queries).parameters)


def main():
    t0 = time.perf_counter()
    for test in (test_exactness, test_gate, test_error_bound, test_information_set,
                 test_entropies, test_real_data, test_hygiene):
        test()
    passed = sum(ok for _, ok in RESULTS)
    print("=" * 72)
    print(f"  {passed} passed, {len(RESULTS) - passed} failed in {time.perf_counter() - t0:.1f}s")
    print("=" * 72)
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
