"""
File: utils.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains general utility functions for the ICC implementation
"""

import itertools
import galois
import math
import numpy as np
from config import SystemContext


def generate_vandermonde_G(GF: type[galois.FieldArray], m: int, n: int) -> galois.FieldArray:
    """
    This is not used in the ICC scheme and is relevant when the user's data is uniformly distributed.
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

def compute_p_entropy(x, q:int, p:int) -> float:
    """
    Calculates the p-entropy of the data according to the formula given in ICC Appendix A

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :return: p-entropy of the data
    """
    counts = np.zeros(q, dtype=float)
    for val in x:
        counts[int(val)] += 1
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    sum_pp = sum(prob ** p for prob in probs if prob > 0)
    res = (1.0 / (p - 1)) * math.log(sum_pp, q)
    return res

def compute_max_subset_p_entropy(x, q: int, p: int, r: int) -> float:
    """
    Calculates the maximum p entropy in any subset of the data in order to derive an accurate
    value for m which Theorem 1 requires. Because this is a toy example I am computing it
    directly, in the future it will shift to the further bound in Appendix B for efficiency

    :param x: Data within range [0,q-1]
    :param q: Field (alphabet) size
    :param p: Order of the entropy (p >= 2)
    :param r: Subset size (privacy parameter)
    :return: Maximum subset p-entropy over the data
    """
    n = len(x)
    x_arr = np.array([int(v) for v in x])

    max_entropy = 0.0
    for indices in itertools.combinations(range(n), r):
        subset = x_arr[list(indices)]
        h = compute_p_entropy(subset, q, p)
        if h > max_entropy:
            max_entropy = h
    return max_entropy

def compute_required_m(context: SystemContext) -> int:
    """
    In the previous (2024) scheme m was a free parameter. Now, in order to ensure that the generator
    matrix is large enough to do smoothing, it must be calculated. Theorem 1 gives the formula
    m >= n + p + log_q(1/ε) - H_p(X) + max_R H_p(X_R) which is used to calculate m. For storage efficiency,
    only returns the minimum value satisfying this formula.

    :param context: SystemContext with the system parameters calculated and set
    :return: The minimum value that m can be satisfying Theorem 1
    :raises: ValueError if the context has not been properly set
    """
    if context.H_p_X is None or context.max_H_p_X_R is None:
        raise ValueError(
            "context.H_p_X and context.max_H_p_X_R must be computed before calling "
            "compute_required_m(). Call compute_p_entropy() and "
            "compute_max_subset_p_entropy() first and store the results on context."
        )

    log_q_inv_eps = math.log(1.0 / context.epsilon, context.q)
    m_float = (context.n + context.p + log_q_inv_eps - context.H_p_X + context.max_H_p_X_R)
    # Take ceiling: m must be an integer and must satisfy the >= bound
    return max(context.r, math.ceil(m_float))

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

def get_information_set(q: int, m: int, d: int) -> list[tuple[int, ...]]:
    """
    Generates the core combinatorial tuples for Reed-Muller RM_q(d, m).
    Used as evaluation points by the Server and as polynomial exponents by the Client.

    This gives explicit description of I d,m information set given in Section IV Definition 3 of 2024 paper
    using a dfs approach to prune so that not every option is visited

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
    base_points = get_simple_information_set(q, m, d)
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