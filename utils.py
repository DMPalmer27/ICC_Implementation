"""
File: utils.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains general utility functions for the ICC implementation
"""

import functools
import itertools
import galois
import math
import numpy as np
from config import SystemContext


def generate_vandermonde_G(GF: type[galois.FieldArray], m: int, n: int) -> galois.FieldArray:
    """
    This is not used in the ICC scheme and is relevant only when the user's data is uniformly distributed.
    It was phased out in favor of the random generator matrix for ICC.

    :param GF: Galois field object
    :param m: Number of rows which is the key length, must satisfy Theorem 1 bound
    :param n: Number of columns which is the data length
    :return: GF array that is an m x n Vandermonde matrix over the given Galois field.
    For to the 2024 paper, this guarantees the MDS property which achieves the optimal m = r bound
    """
    if GF.order < n + 1:
        raise ValueError(
            f"Field size q={GF.order} is too small! Must be >= {n + 1} to create an MDS matrix of length n.")

    # Select n distinct, non-zero elements from the field
    # (GF.elements returns all elements [0, 1, 2...]. We slice from 1 to avoid 0)
    alphas = GF.elements[1:n + 1]

    G = GF.Zeros((m, n))
    for i in range(m):
        for j in range(n):
            # In polynomial evaluation, the data 'x' is the constant term (power 0).
            # The key 'k' acts as the higher coefficients (power 1 to m).
            # So, row i corresponds to the power (i + 1).
            G[i, j] = alphas[j] ** (i + 1)

    return G

def generate_random_G(GF: type[galois.FieldArray], m: int, n: int) -> galois.FieldArray:
    """
    Generate a generator matrix for a random linear code. With sufficent size of m (given by Theorem 1 bound) this
    will successfully smooth with high probability.

    :param GF: Galois field object
    :param m: Number of rows which is the key length, must satisfy Theorem 1 bound
    :param n: Number of columns which is the data length
    :return: GF array of shape (m,n) with uniform random entries from GF
    """
    return GF.Random((m,n))

def compute_symbol_p_entropy(x, q: int, p: int) -> float:
    """
    Calculates the PER-SYMBOL p-entropy h of the data according to the formula given in
    ICC Appendix A. The alphabet distribution is estimated empirically from the n symbols
    of x, which treats them as n i.i.d. draws from one common source distribution.

    Units are log_q, so h = 0 for deterministic data and h = 1 for data uniform over F_q.
    This is the primitive estimator; Theorem 1 wants the vector quantity H_p(X), which is
    n * h under the i.i.d. model -- see compute_p_entropy.

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :return: Per-symbol p-entropy h of the data, in log_q units
    """
    # Histogram only the values that actually occur. A dense length-q histogram would be
    # 8.6 GB at the case-study field size q ~ 2^30, and symbols not present contribute
    # nothing to sum(prob^p) anyway.
    values = np.asarray(x).astype(np.int64, copy=False)
    if values.size == 0:
        return 0.0

    _, counts = np.unique(values, return_counts=True)
    probs = counts / values.size
    sum_pp = float(np.sum(probs ** p))
    res = (1.0 / (1-p)) * math.log(sum_pp, q)
    return res

def compute_p_entropy(x, q: int, p: int) -> float:
    """
    Calculates H_p(X), the p-entropy of the whole length-n data VECTOR, as required by the
    Theorem 1 bound. Renyi entropy is additive over independent coordinates, so under the
    i.i.d. source model H_p(X) = n * h where h is the per-symbol entropy.

    Note the units: log_q, so H_p(X) ranges over [0, n] and equals n exactly when X is
    uniform over F_q^n. Previously this function returned the per-symbol h, which made the
    -H_p(X) and +max_R H_p(X_R) terms of Theorem 1 nearly cancel and forced m > n in every
    test. See compute_symbol_p_entropy for the per-symbol quantity.

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :return: H_p(X) for the full data vector, in log_q units
    """
    return len(x) * compute_symbol_p_entropy(x, q, p)

def compute_max_subset_p_entropy(x, q: int, p: int, r: int) -> float:
    """
    Calculates max_R H_p(X_R) over all r-subsets R, the quantity Theorem 1 adds to the
    bound on m.

    Under the i.i.d. source model every r-subset of X has the same distribution, so the
    maximum is attained everywhere and equals r * h in closed form. This replaces a brute
    force over all C(n,r) subsets, which was superlinear in n and capped the project at
    n ~ 12 (C(60,5) = 5.4M). Estimating h from all n symbols is also a strictly better
    estimator than the old approach of estimating it from just r values.

    Because r <= n, this construction guarantees max_R H_p(X_R) <= H_p(X), which the old
    code could violate -- the tell that its scaling was wrong.

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :param r: Subset size (privacy parameter)
    :return: max_R H_p(X_R) over all r-subsets, in log_q units
    """
    return r * compute_symbol_p_entropy(x, q, p)

def _compute_max_subset_p_entropy_empirical(x, q: int, p: int, r: int) -> float:
    """
    DEPRECATED -- retained for reference only, not used to compute m.

    This is the original brute force over all C(n,r) subsets. It is NOT an oracle for
    compute_max_subset_p_entropy and will not agree with it: it estimates the empirical
    entropy of one r-element realization, whereas Theorem 1 wants the entropy of the
    r-subset under the source distribution. For small r the empirical estimate is badly
    biased (r values cannot resolve a q-ary distribution), which is why it could report
    max_R H_p(X_R) > H_p(X).

    Two defensible estimators exist here and they should not be silently mixed; this one
    is kept visible so the difference stays documented rather than lost.

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :param r: Subset size (privacy parameter)
    :return: Maximum empirical per-subset p-entropy, in log_q units
    """
    n = len(x)
    x_arr = np.array([int(v) for v in x])

    max_entropy = -9999999
    for indices in itertools.combinations(range(n), r):
        subset = x_arr[list(indices)]
        h = compute_symbol_p_entropy(subset, q, p)
        if h > max_entropy:
            max_entropy = h
    return max_entropy

def compute_required_m(context: SystemContext) -> int:
    """
    In the previous (2024) scheme m was a free parameter. Now, in order to ensure that the generator
    matrix is large enough to do smoothing, it must be calculated. Theorem 1 gives the formula
    m >= n + p + log_q(1/ε) - H_p(X) + max_R H_p(X_R) which is used to calculate m. For storage efficiency,
    only returns the minimum value satisfying this formula.

    Both entropy terms must be VECTOR quantities in log_q units (H_p(X) over all n symbols,
    max_R H_p(X_R) over r of them), not per-symbol values -- see compute_p_entropy.

    The max(r, ...) floor is justified by Remark 1 of the ICC paper: a key of size at least r
    is necessary for r-subset privacy. (The report previously justified it by the Singleton
    bound, which came from the MDS/uniform predecessor scheme; under ICC the code is random,
    so Singleton does not apply.)

    :param context: SystemContext with the system parameters calculated and set
    :return: The minimum value that m can be satisfying Theorem 1
    :raises: ValueError if the context has not been properly set or the entropies are inconsistent
    """
    if context.H_p_X is None or context.max_H_p_X_R is None:
        raise ValueError(
            "context.H_p_X and context.max_H_p_X_R must be computed before calling "
            "compute_required_m(). Call compute_p_entropy() and "
            "compute_max_subset_p_entropy() first and store the results on context."
        )

    # Guard the impossible cases. An r-subset cannot carry more entropy than the whole
    # vector, and n symbols over F_q cannot exceed n in log_q units. Either violation means
    # per-symbol and vector conventions have been mixed, which would silently under-estimate
    # m and break the privacy guarantee.
    if context.max_H_p_X_R > context.H_p_X + 1e-9:
        raise ValueError(
            f"max_R H_p(X_R) = {context.max_H_p_X_R:.6f} > H_p(X) = {context.H_p_X:.6f}, "
            "which is impossible for the true quantities. This usually means a per-symbol "
            "entropy was stored where a vector entropy H_p(X) was expected."
        )
    if context.H_p_X > context.n + 1e-9:
        raise ValueError(
            f"H_p(X) = {context.H_p_X:.6f} exceeds n = {context.n}. In log_q units the "
            "entropy of n symbols over F_q is at most n."
        )

    log_q_inv_eps = math.log(1.0 / context.epsilon, context.q)
    m_float = (context.n + context.p + log_q_inv_eps - context.H_p_X + context.max_H_p_X_R)
    # Take ceiling: m must be an integer and must satisfy the >= bound
    m = max(context.r, math.ceil(m_float))

    # The scheme requires d < m(q-1) for RM_q(d, m) to be well defined. Never asserted
    # anywhere before; it binds only at toy field sizes.
    if context.d >= m * (context.q - 1):
        raise ValueError(
            f"Scheme requires d < m(q-1), but d = {context.d} and m(q-1) = {m * (context.q - 1)} "
            f"(m = {m}, q = {context.q})."
        )

    return m

def compute_leakage_bound(context: SystemContext) -> float:
    """
    Computes the information leakage upper bound e_c given the scheme parameters from the formula
    given in Theorem 1 with probability at least 1 - 1/a:
    ε_c = (p/(p-1)) * log_q(1 + a * 2^{(2p-1)/p} * (1 + q^{-max_R H_p(X_R)}) * ε^{1/p})
    This is used to show the success of the scheme.

    :param context: SystemContext with the system parameters calculated and set
    :return: mutual information leakage upper bound over all subsets
    :raises: ValueError if the context has not been properly set
    """
    if context.max_H_p_X_R is None:
        raise ValueError(
            "context.max_H_p_X_R must be computed before calling"
        )
    p, q = context.p, context.q
    a, eps = context.a, context.epsilon
    max_H = context.max_H_p_X_R

    delta = a * (2 ** ((2 * p - 1) / p)) * (1 + q ** (-max_H)) * (eps ** (1.0 / p))
    eps_c = (p / (p - 1)) * math.log(1 + delta, q)
    return eps_c

def build_monomial_matrix(GF: type[galois.FieldArray], points, exponents) -> galois.FieldArray:
    """
    Builds the evaluation matrix M[i, j] = prod_v points[i][v] ** exponents[j][v] over GF.

    This is the interpolation matrix the Client solves to recover g, and it is also used to
    evaluate g at the secret key. It depends only on the evaluation points and the monomial
    exponents -- not on the data, the polynomial, or the key -- so a caller that reuses one
    information set can build it once and cache it.

    Implementation note: the obvious triple loop (points x exponents x variables) costs
    lambda^2 * m interpreted operations and dominated decoding by ~400x over the linear solve
    it feeds (64.9 s vs 0.15 s at m=48, d=2). This version instead makes one vectorised pass
    per VARIABLE: it builds the power table [1, t_v, t_v^2, ...] for column v across all
    points at once, gathers by exponent, and multiplies into the accumulator. Measured 0.18 s
    for the same case, a ~360x speedup, elementwise identical to the loop.

    :param GF: Galois field object
    :param points: N evaluation points, as an (N, m) array or a sequence of length-m vectors
    :param exponents: L monomial exponent tuples, as an (L, m) array or sequence of tuples
    :return: GF array of shape (N, L) of monomial evaluations
    """
    P = points if isinstance(points, GF) else GF(np.asarray(points))
    P = P.reshape(1, -1) if P.ndim == 1 else P
    E = np.asarray(exponents, dtype=np.int64)
    E = E.reshape(1, -1) if E.ndim == 1 else E

    n_points, m = P.shape
    if E.shape[1] != m:
        raise ValueError(f"points have {m} variables but exponents have {E.shape[1]}")

    M = GF.Ones((n_points, E.shape[0]))
    for v in range(m):
        max_exp = int(E[:, v].max())
        if max_exp == 0:
            # Every monomial ignores this variable, so it contributes a factor of 1
            continue
        # powers[e] holds points[:, v] ** e for every point at once
        powers = [GF.Ones(n_points)]
        for _ in range(max_exp):
            powers.append(powers[-1] * P[:, v])
        M = M * GF(np.stack(powers))[E[:, v]].T

    return M

@functools.lru_cache(maxsize=None)
def get_information_set(q: int, m: int, d: int) -> list[tuple[int, ...]]:
    """
    Generates the core combinatorial tuples for Reed-Muller RM_q(d, m).
    Used as evaluation points by the Server and as polynomial exponents by the Client.

    This gives explicit description of I d,m information set given in Section IV Definition 3 of 2024 paper
    using a dfs approach to prune so that not every option is visited

    Results are memoised because the set depends only on (q, m, d) and is rebuilt on every
    decode_result() and every store_data(). The returned list is shared between callers, so
    treat it as read-only.

    :param q: Field (alphabet) size
    :param m: Number of variables
    :param d: Max total degree
    :return: All m-tuples with entries in [0, q-1] such that sum(a_i) <= d
    """
    results = []

    def recurse(depth: int, current: tuple, remaining_budget: int):
        if depth == m:
            results.append(current)
            return
        # Never exceed remaining_budget — prunes entire subtrees immediately
        for val in range(min(q, remaining_budget + 1)):
            recurse(depth + 1, current + (val,), remaining_budget - val)

    recurse(0, (), d)
    return results

def get_information_superset(GF: type[galois.FieldArray], q: int, m: int, d: int, S: int) -> list[galois.FieldArray]:
    """
    This is a prototype for a function that would get an information superset which would allow resilience against
    S stragglers. It builds the superset from scratch by ensuring that no invariants are violated. Because of this
    construction, it is incredibly inefficent.

    Generates an Information Super-set of size lambda + S.
    Guarantees that ANY subset of size lambda yields a full-rank evaluation matrix.

    :param GF: Galois field object
    :param q: Field (alphabet) size
    :param m: Number of variables
    :param d: Max total degree
    :param S: Number of stragglers to have resilience against
    :return: Superset containing entries such that any subset of size lambda contain an information set
    """
    base_points = get_information_set(q, m, d)
    lambda_len = len(base_points)

    # Start the super-set with the base Information Set
    superset = [GF(list(p)) for p in base_points]

    all_points = itertools.product(range(q), repeat=m)

    for pt in all_points:
        if len(superset) == lambda_len + S:
            break

        gf_pt = GF(list(pt))

        # Skip if point is already in our superset
        if any(np.array_equal(gf_pt, sp) for sp in superset):
            continue

        candidate_set = superset + [gf_pt]

        # Ensure that ALL possible subsets of size lambda maintain full rank
        valid = True
        for subset in itertools.combinations(candidate_set, lambda_len):
            M_arr = []
            for p in subset:
                vals = []
                for exp in base_points:
                    term_val = GF(1)
                    for i in range(m):
                        if exp[i] > 0:
                            term_val *= p[i] ** exp[i]
                    vals.append(term_val)
                M_arr.append(vals)

            M = GF(M_arr)
            # If the matrix rank drops below lambda, this is not a valid super-set point
            if np.linalg.matrix_rank(M) < lambda_len:
                valid = False
                break

        if valid:
            superset.append(gf_pt)

    if len(superset) < lambda_len + S:
        raise ValueError("Field size too small to find a valid Information Super-set.")

    return superset