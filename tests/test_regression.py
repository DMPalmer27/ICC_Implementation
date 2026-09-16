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
      6-10. quantisation: fixed-point round trips, the balanced-lift sign convention, the
            two no-wraparound bounds, the storage-time query budget, the direction of the
            entropy bounds, and a quantised gradient decoding exactly through the scheme

Usage:
    .venv/bin/python tests/test_regression.py
"""

import conftest  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
import time
import galois
import numpy as np

from config import SystemContext
from client import Client
from server import Server
from quantize import (
    POWER_OF_TWO_GRID,
    FixedPointSpec,
    QueryBudget,
    assert_representable,
    choose_field_size,
    choose_prime,
    codes_to_field,
    entropy_bounds,
    from_field,
    from_integer,
    gradient_bound_data_dependent,
    gradient_bound_worst_case,
    leakage_budget_bits,
    lift_balanced,
    symbol_entropy_lower_bits,
    to_field,
    to_integer,
)
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

def _raises(fn, exc: type[Exception] = ValueError) -> bool:
    """
    Reports whether calling fn raises the expected exception type.

    :param fn: Zero-argument callable
    :param exc: Exception type to expect; the quantisation checks expect OverflowError
    :return: True if exc was raised
    """
    try:
        fn()
        return False
    except exc:
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quantisation: round trips, the sign convention, and the two hard bounds
# ─────────────────────────────────────────────────────────────────────────────

def test_quantization_round_trip():
    print("\n6. Fixed-point round trips and the balanced-lift sign convention")
    spec = FixedPointSpec(f=10, clip=4.0)
    rng = np.random.default_rng(11)
    v = rng.normal(size=5000)

    codes = to_integer(v, spec)
    back = from_integer(codes, spec)
    in_range = np.abs(v) <= spec.clip
    check("nearest rounding stays within half a step",
          bool(np.all(np.abs(back[in_range] - v[in_range]) <= spec.step / 2 + 1e-12)))
    check("clipping is enforced in both directions",
          int(np.abs(codes).max()) <= spec.max_int)

    q = int(galois.next_prime(2 ** 20))
    GF = galois.GF(q)
    z = to_field(GF, v, spec)
    check("to_field then lift_balanced recovers the integer codes exactly",
          bool(np.array_equal(np.asarray(lift_balanced(z, q)), codes)))
    check("from_field inverts a single-factor embedding",
          bool(np.allclose(from_field(z, q, spec.f), back)))

    # Signs are the part that silently breaks; check the window edges explicitly
    edges = np.array([-(q - 1) // 2, -1, 0, 1, (q - 1) // 2])
    check("balanced lift is the identity on the whole window",
          bool(np.array_equal(np.asarray(lift_balanced(GF(edges % q), q)), edges)))

    check("assert_representable accepts a value inside the window",
          not _raises(lambda: assert_representable((q - 1) // 2, q), OverflowError))
    check("assert_representable rejects a value one past the window",
          _raises(lambda: assert_representable((q - 1) // 2 + 1, q), OverflowError))
    check("to_field rejects an alphabet that cannot fit in F_q",
          _raises(lambda: to_field(galois.GF(31), v, spec), OverflowError))

    # Extension fields label their elements with integers too and some operations look
    # right by coincidence, so an unguarded run returns confident nonsense rather than
    # failing. GF(2^5): the image of Z is {0, 1} and 3 + 5 = 6.
    check("the embedding rejects extension fields",
          _raises(lambda: codes_to_field(galois.GF(2 ** 5), codes[:4])))
    check("the embedding accepts prime fields",
          not _raises(lambda: codes_to_field(GF, codes[:4])))
    check("integer addition survives the prime-field embedding but not GF(2^5)",
          int(GF(3) + GF(5)) == 8 and int(galois.GF(2 ** 5)(3) + galois.GF(2 ** 5)(5)) != 8)


def test_gradient_bounds_are_valid():
    print("\n7. The no-wraparound bounds actually bound the gradient")
    n_samples, n_features, trials = 6, 3, 40
    x_spec = FixedPointSpec(f=8, clip=4.0)
    w_spec = FixedPointSpec(f=4, clip=4.0)
    B_wc = gradient_bound_worst_case(n_samples, n_features, x_spec, w_spec)

    rng = np.random.default_rng(3)
    wc_ok = dd_ok = order_ok = True
    for _ in range(trials):
        # Heavy tails on purpose, so clipping is exercised rather than avoided
        X = rng.standard_t(2, size=(n_samples, n_features))
        y = rng.standard_t(2, size=n_samples)
        w = rng.standard_t(2, size=n_features)
        Xq, yq, wq = to_integer(X, x_spec), to_integer(y, x_spec), to_integer(w, w_spec)
        B_dd = gradient_bound_data_dependent(Xq, yq, w_spec)

        for j in range(n_features):
            g = int(sum((sum(int(Xq[i, k]) * int(wq[k]) for k in range(n_features))
                         - 2 ** w_spec.f * int(yq[i])) * int(Xq[i, j])
                        for i in range(n_samples)))
            wc_ok &= abs(g) <= B_wc
            dd_ok &= abs(g) <= B_dd
        order_ok &= B_dd <= B_wc

    check(f"worst-case bound holds over {trials} heavy-tailed trials", wc_ok)
    check(f"data-dependent bound holds over {trials} heavy-tailed trials", dd_ok)
    check("data-dependent bound is never looser than the worst case", order_ok)

    q = choose_prime(B_wc, warn_native=False)
    check("choose_prime returns a prime with q >= 2B + 1",
          q >= 2 * B_wc + 1 and galois.is_prime(q))

    # The field-size policy, including the leakage accounting that makes it a decision
    public = choose_field_size(B_wc, warn_native=False)
    check("public-format policy leaks nothing and cannot wrap",
          public.leakage_bits == 0.0 and public.q >= 2 * B_wc + 1,
          f"{public}")
    check("a data-derived bound cannot be published without a grid",
          _raises(lambda: choose_field_size(B_wc, data_dependent=True)))

    coarse = choose_field_size(B_wc, data_dependent=True, grid=POWER_OF_TWO_GRID,
                               warn_native=False)
    check("coarsened policy bounds its leakage by log2 of the grid size",
          math.isclose(coarse.leakage_bits, math.log2(len(POWER_OF_TWO_GRID)))
          and coarse.q >= 2 * B_wc + 1)
    check("coarsening rounds the bound up, never down",
          coarse.magnitude_bound >= B_wc)
    check("a bound past the end of the grid is refused",
          _raises(lambda: choose_field_size(2 ** 60, data_dependent=True,
                                            grid=POWER_OF_TWO_GRID)))
    # The comparison that decides the policy: the side channel dwarfs the scheme's budget
    check("the coarsened side channel exceeds the scheme's own leakage budget",
          coarse.leakage_bits > 50 * leakage_budget_bits(0.0016, public.q),
          f"{coarse.leakage_bits:.2f} vs {leakage_budget_bits(0.0016, public.q):.4f} bits")


def test_query_budget_ordering():
    print("\n8. Storage-time budget admits later queries, or refuses them")
    # q is frozen at upload, so it is sized against a declared ceiling on future queries
    budget = QueryBudget(n_samples=25, n_features=3, d=2,
                         x_spec=FixedPointSpec(f=8, clip=4.0),
                         w_spec=FixedPointSpec(f=0, clip=4.0))
    B = budget.magnitude_bound()
    check("budget reproduces the worst-case bound it is built from",
          B == gradient_bound_worst_case(25, 3, budget.x_spec, budget.w_spec))

    w_ok = to_integer(np.array([1.2, -3.1, 0.4]), budget.w_spec)
    check("a query inside the declared ceiling is admitted",
          not _raises(lambda: budget.assert_admissible(w_ok, 0)))
    check("raising f_w after upload is refused",
          _raises(lambda: budget.assert_admissible(w_ok, 4)))
    check("weights past the declared clip are refused",
          _raises(lambda: budget.assert_admissible(np.array([9, 1, 1]), 0)))

    # The one-sidedness is what makes a ceiling workable: anything smaller must stay safe
    loose = QueryBudget(n_samples=25, n_features=3, d=2,
                        x_spec=FixedPointSpec(f=8, clip=4.0),
                        w_spec=FixedPointSpec(f=8, clip=4.0))
    check("a looser ceiling costs field size but admits the tight query too",
          loose.magnitude_bound() > B
          and not _raises(lambda: loose.assert_admissible(w_ok, 0)))

    # Every admitted query must actually fit the window q was sized for
    q = choose_field_size(B, warn_native=False).q
    rng = np.random.default_rng(17)
    fits = True
    for _ in range(20):
        Xq = to_integer(rng.standard_t(2, size=(25, 3)), budget.x_spec)
        yq = to_integer(rng.standard_t(2, size=25), budget.x_spec)
        wq = to_integer(rng.standard_t(2, size=3), budget.w_spec)
        budget.assert_admissible(wq, budget.w_spec.f)
        for j in range(3):
            g = int(sum((sum(int(Xq[i, k]) * int(wq[k]) for k in range(3))
                         - 2 ** budget.w_spec.f * int(yq[i])) * int(Xq[i, j])
                        for i in range(25)))
            fits &= abs(g) <= (q - 1) // 2
    check("20 admitted queries all land inside the balanced window", fits)


def test_entropy_bounds_directions():
    print("\n9. Quantisation entropy bounds point the way Theorem 1 needs")
    spec = FixedPointSpec(f=8, clip=4.0)
    q = int(galois.next_prime(2 ** 30))
    n, r = 40, 5

    H_lower, max_upper = entropy_bounds(n, r, q, spec)
    check("H_p(X) lower bound and max_R upper bound are mutually consistent",
          max_upper <= H_lower <= n, f"{max_upper:.3f} <= {H_lower:.3f} <= {n}")

    ctx = SystemContext(q=q, n=n, d=2, r=r, p=2, epsilon=1e-6)
    ctx.H_p_X, ctx.max_H_p_X_R = H_lower, max_upper
    check("the bounds pass compute_required_m's guards",
          not _raises(lambda: compute_required_m(ctx)))

    # The saturation term must be able to win, or the f=16/clip=4 failure recurs
    wide = symbol_entropy_lower_bits(FixedPointSpec(f=16, clip=4.0), saturation_mass=3.2e-5)
    interior = symbol_entropy_lower_bits(FixedPointSpec(f=16, clip=4.0))
    check("a dominant saturation atom lowers the bound (not ignored)",
          wide < interior, f"{wide:.3f} vs {interior:.3f}")
    check("a negligible saturation atom leaves the interior term binding",
          math.isclose(symbol_entropy_lower_bits(spec, saturation_mass=1e-30),
                       symbol_entropy_lower_bits(spec)))

    # More precision must never lower the entropy bound
    monotone = all(symbol_entropy_lower_bits(FixedPointSpec(f=f, clip=6.0))
                   < symbol_entropy_lower_bits(FixedPointSpec(f=f + 1, clip=6.0))
                   for f in range(2, 20))
    check("the bound is increasing in f while the interior term binds", monotone)


def test_quantized_gradient_decodes_exactly():
    print("\n10. A quantised gradient survives the scheme exactly")
    n_samples, n_features, f_X, f_w = 4, 2, 6, 2
    x_spec = FixedPointSpec(f=f_X, clip=4.0)
    w_spec = FixedPointSpec(f=f_w, clip=4.0)
    n = n_samples * (n_features + 1)

    rng = np.random.default_rng(5)
    X, y, w = (rng.normal(size=(n_samples, n_features)), rng.normal(size=n_samples),
               rng.normal(size=n_features))
    Xq, yq, wq = to_integer(X, x_spec), to_integer(y, x_spec), to_integer(w, w_spec)

    q = choose_prime(gradient_bound_worst_case(n_samples, n_features, x_spec, w_spec),
                     warn_native=False)
    GF = galois.GF(q)
    ctx = SystemContext(q=q, n=n, d=2, r=3, p=2, epsilon=1e-3)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy_bounds(n, ctx.r, q, x_spec)
    ctx.m = compute_required_m(ctx)

    client, server = Client(ctx, GF), Server(ctx, GF)
    G = generate_random_G(GF, ctx.m, n)
    x_field = codes_to_field(GF, np.concatenate([Xq.ravel(), yq]), x_spec)
    server.store_data(client.encode_data(x_field, G), G)

    iX = lambda i, k: i * n_features + k
    iy = lambda i: n_samples * n_features + i
    w_gf = [GF(int(v) % q) for v in wq]
    two_fw = GF(2 ** f_w % q)

    exact = True
    for j in range(n_features):
        def f(z, j=j):
            acc = GF(0)
            for i in range(n_samples):
                residual = GF(0)
                for k in range(n_features):
                    residual = residual + z[iX(i, k)] * w_gf[k]
                acc = acc + (residual - two_fw * z[iy(i)]) * z[iX(i, j)]
            return acc

        points, results = server.compute_request(f)
        decoded = int(lift_balanced(client.decode_result(points, results), q))
        oracle = int(sum((sum(int(Xq[i, k]) * int(wq[k]) for k in range(n_features))
                          - 2 ** f_w * int(yq[i])) * int(Xq[i, j])
                         for i in range(n_samples)))
        exact &= decoded == oracle

    check(f"all {n_features} gradient components decode to the integer oracle "
          f"(m={ctx.m}, lambda={math.comb(ctx.m + 2, 2)})", exact)


def main():
    print("=" * 72)
    print("  ICC REGRESSION CHECKS  (Week 1 optimisation pass, Week 3 quantisation)")
    print("=" * 72)
    t0 = time.perf_counter()

    test_monomial_matrix_matches_legacy()
    test_decode_exact()
    test_entropy_consistency()
    test_required_m_guards()
    test_server_shares_match_legacy()
    test_quantization_round_trip()
    test_gradient_bounds_are_valid()
    test_query_budget_ordering()
    test_entropy_bounds_directions()
    test_quantized_gradient_decodes_exactly()

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
