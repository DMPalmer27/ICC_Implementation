"""
File: test_regression.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Fast regression checks for the Week 1 optimisation pass. Unlike test_icc.py,
    which doubles as report output and takes minutes, this file is a safety net that runs in
    seconds and asserts the invariants the rewrite could plausibly break:

      1. build_monomial_matrix agrees elementwise with the original triple loop
      2. decode_result still recovers f(x) exactly, single and batched
      3. the Theorem 1 entropy quantities are mutually consistent
      4. compute_required_m rejects impossible inputs instead of returning a bad m
      5. the vectorised Server.store_data produces the same shares as the old loop

Usage:
    .venv/bin/python tests/test_regression.py
"""

import conftest  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import time
import galois
import numpy as np

from config import SystemContext
from client import Client
from server import Server
from utils import (
    build_monomial_matrix,
    compute_max_subset_p_entropy,
    compute_p_entropy,
    compute_required_m,
    compute_symbol_p_entropy,
    generate_random_G,
    get_information_set,
)

_PASSED = []
_FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    """
    Records the outcome of one assertion.

    :param name: Human readable description of the invariant
    :param condition: Whether the invariant held
    :param detail: Extra context printed on failure
    """
    if condition:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append(name)
        print(f"  ✗ {name}  {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Oracles: the pre-optimisation implementations, kept here so the checks survive
# deleting them from the library.
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_evaluate_monomials(GF, point, exponents, m: int) -> list:
    """
    Original Client._evaluate_monomials: scalar triple loop over the field.

    :param GF: Galois field object
    :param point: Single evaluation point
    :param exponents: Monomial exponent tuples
    :param m: Number of variables
    :return: List of monomial evaluations at the point
    """
    vals = []
    for exp in exponents:
        term_val = GF(1)
        for i in range(m):
            if exp[i] > 0:
                term_val *= point[i] ** exp[i]
        vals.append(term_val)
    return vals


def _legacy_shares(GF, x_tilde, G, points) -> list:
    """
    Original Server.store_data sharding: one matmul per evaluation point.

    :param GF: Galois field object
    :param x_tilde: Encoded data vector
    :param G: Generator matrix
    :param points: Evaluation points
    :return: List of per-worker shares
    """
    return [x_tilde - np.matmul(p, G) for p in points]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vectorised monomial matrix == legacy loop
# ─────────────────────────────────────────────────────────────────────────────

def test_monomial_matrix_matches_legacy():
    print("\n1. build_monomial_matrix vs legacy triple loop")
    # (q, m, d, rows_to_check). The legacy loop IS the 65 s bottleneck being removed, so at
    # large m only a sample of rows is compared against it; small cases are checked in full.
    cases = [
        (31, 10, 2, None), (31, 20, 2, None), (31, 12, 3, 24), (31, 10, 4, 16),
        (1073741827, 48, 2, 12),
    ]
    for (q, m, d, sample) in cases:
        GF = galois.GF(q)
        exponents = get_information_set(q, m, d)
        rng = np.random.default_rng(0)

        # Mix the real information-set points with random ones: the former are the actual
        # workload, the latter exercise exponents beyond the sparse structure.
        info_points = GF(np.asarray(exponents) % q)
        random_points = GF(rng.integers(0, q, size=(8, m)))

        for label, P in (("info-set", info_points), ("random", random_points)):
            fast = build_monomial_matrix(GF, P, exponents)
            rows = range(P.shape[0]) if sample is None else \
                rng.choice(P.shape[0], size=min(sample, P.shape[0]), replace=False)
            legacy = GF([_legacy_evaluate_monomials(GF, P[i], exponents, m) for i in rows])
            scope = "all rows" if sample is None else f"{len(list(rows))} sampled rows"
            check(f"q={q} m={m} d={d} ({label}, {scope}) identical",
                  np.array_equal(np.asarray(fast[list(rows)]), np.asarray(legacy)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-end decoding, single and batched
# ─────────────────────────────────────────────────────────────────────────────

def _build_ctx(q=31, n=10, d=2, r=5, p=2, seed=0):
    """
    Builds a context with skewed data and populated entropy fields.

    :param q: Field size
    :param n: Data length
    :param d: Max total degree
    :param r: Privacy parameter
    :param p: Entropy order
    :param seed: RNG seed
    :return: Tuple of (context, GF, data vector)
    """
    ctx = SystemContext(q=q, n=n, d=d, r=r, p=p)
    GF = galois.GF(q)
    rng = np.random.default_rng(seed)
    probs = np.zeros(q)
    probs[:3] = 0.9 / 3
    probs[3:] = 0.1 / (q - 3)
    x = GF(rng.choice(q, size=n, p=probs))
    ctx.H_p_X = compute_p_entropy(x, q, p)
    ctx.max_H_p_X_R = compute_max_subset_p_entropy(x, q, p, r)
    ctx.m = compute_required_m(ctx)
    return ctx, GF, x


def test_decode_exact():
    print("\n2. decode_result recovers f(x) exactly")
    ctx, GF, x = _build_ctx()
    fn = lambda data: data[0] ** 2 + data[1] * data[2]

    failures = 0
    for seed in range(20):
        np.random.seed(seed)
        client, server = Client(ctx, GF), Server(ctx, GF)
        G = generate_random_G(GF, ctx.m, ctx.n)
        server.store_data(client.encode_data(x, G), G)
        points, results = server.compute_request(fn)
        if client.decode_result(points, results) != fn(x):
            failures += 1
    check(f"20 seeded trials decode exactly", failures == 0, f"{failures} failed")

    # Batched right-hand side: two polynomials decoded in one elimination
    np.random.seed(123)
    client, server = Client(ctx, GF), Server(ctx, GF)
    G = generate_random_G(GF, ctx.m, ctx.n)
    server.store_data(client.encode_data(x, G), G)
    f1 = lambda data: data[0] ** 2 + data[1] * data[2]
    f2 = lambda data: data[3] * data[4] + data[0]
    points, r1 = server.compute_request(f1)
    _, r2 = server.compute_request(f2)

    batch = GF(np.stack([np.asarray(GF(r1)), np.asarray(GF(r2))], axis=1))
    decoded = client.decode_result(points, batch)
    check("batched (lambda, 2) RHS matches per-polynomial results",
          bool(decoded[0] == f1(x)) and bool(decoded[1] == f2(x)),
          f"got {decoded}, expected [{f1(x)}, {f2(x)}]")

    # The cache must not change the answer or grow per call
    before = len(client._M_cache)
    for _ in range(3):
        client.decode_result(points, r1)
    check("interpolation cache is reused, not regrown",
          len(client._M_cache) == before, f"{before} -> {len(client._M_cache)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Entropy consistency across the Section 2 parameter grid
# ─────────────────────────────────────────────────────────────────────────────

def test_entropy_consistency():
    print("\n3. Theorem 1 entropy quantities are consistent")
    grid = [
        (31, 8, 2, 3, 2), (31, 10, 2, 5, 2), (31, 12, 2, 4, 2),
        (31, 8, 2, 3, 3), (37, 10, 2, 4, 2),
    ]
    all_ok = True
    for (q, n, d, r, p) in grid:
        ctx, GF, x = _build_ctx(q=q, n=n, d=d, r=r, p=p, seed=99)
        h = compute_symbol_p_entropy(x, q, p)
        if not (ctx.max_H_p_X_R <= ctx.H_p_X + 1e-9
                and ctx.H_p_X <= n + 1e-9
                and abs(ctx.H_p_X - n * h) < 1e-9
                and abs(ctx.max_H_p_X_R - r * h) < 1e-9):
            all_ok = False
    check("max_R H_p(X_R) <= H_p(X) <= n, and both scale from h", all_ok)


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_required_m rejects impossible inputs
# ─────────────────────────────────────────────────────────────────────────────

def _raises(fn) -> bool:
    """
    Reports whether calling fn raises ValueError.

    :param fn: Zero-argument callable
    :return: True if ValueError was raised
    """
    try:
        fn()
        return False
    except ValueError:
        return True


def test_required_m_guards():
    print("\n4. compute_required_m guards")

    bad_subset = SystemContext(q=31, n=10, d=2, r=5, p=2)
    bad_subset.H_p_X, bad_subset.max_H_p_X_R = 0.5, 1.0
    check("rejects max_R H_p(X_R) > H_p(X)", _raises(lambda: compute_required_m(bad_subset)))

    bad_total = SystemContext(q=31, n=2, d=2, r=1, p=2)
    bad_total.H_p_X, bad_total.max_H_p_X_R = 5.0, 1.0
    check("rejects H_p(X) > n", _raises(lambda: compute_required_m(bad_total)))

    # q=2 makes m(q-1) = m, so a large d violates d < m(q-1)
    bad_degree = SystemContext(q=2, n=1, d=10, r=1, p=2, epsilon=0.5)
    bad_degree.H_p_X, bad_degree.max_H_p_X_R = 1.0, 1.0
    check("rejects d >= m(q-1)", _raises(lambda: compute_required_m(bad_degree)))

    good = SystemContext(q=31, n=10, d=2, r=5, p=2)
    good.H_p_X, good.max_H_p_X_R = 3.0, 1.5
    check("accepts a consistent context", not _raises(lambda: compute_required_m(good)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Vectorised sharding == legacy sharding
# ─────────────────────────────────────────────────────────────────────────────

def test_server_shares_match_legacy():
    print("\n5. Server.store_data sharding vs legacy loop")
    ctx, GF, x = _build_ctx()
    np.random.seed(7)
    client, server = Client(ctx, GF), Server(ctx, GF)
    G = generate_random_G(GF, ctx.m, ctx.n)
    x_tilde = client.encode_data(x, G)
    server.store_data(x_tilde, G)

    legacy = _legacy_shares(GF, x_tilde, G, server.points)
    same = all(np.array_equal(np.asarray(server.workers[i].data), np.asarray(legacy[i]))
               for i in range(len(legacy)))
    check(f"all {len(legacy)} worker shares identical", same)


def main():
    print("=" * 72)
    print("  ICC REGRESSION CHECKS  (Week 1 optimisation pass)")
    print("=" * 72)
    t0 = time.perf_counter()

    test_monomial_matrix_matches_legacy()
    test_decode_exact()
    test_entropy_consistency()
    test_required_m_guards()
    test_server_shares_match_legacy()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"  {len(_PASSED)} passed, {len(_FAILED)} failed in {elapsed:.1f}s")
    if _FAILED:
        for name in _FAILED:
            print(f"    FAILED: {name}")
    print("=" * 72)
    raise SystemExit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
