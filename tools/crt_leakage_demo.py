"""
File: crt_leakage_demo.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Exact demonstration of what the provider learns from SEVERAL encodings of the
    same data -- the situation CRT regime (a) creates -- and why the key length of each
    encoding must be sized jointly rather than per instance.

    One encoding x~ = x + kG reveals the coset x + rowspace(G): the provider learns that
    H(x~ - x) = 0 for the (n - m) x n parity-check matrix H of the code, i.e. n - m linear
    constraints on x. Theorem 1 sizes m so that, for the data's distribution, enough
    candidates survive those constraints. The provider holding L encodings of the same x
    under independent (k_i, G_i), over primes q_i, gets ALL of the constraint sets at once,
    and the candidates must satisfy every one of them.

    Heuristic count: with x uniform on a set of size 2^H, one encoding leaves about
    2^H / q^(n-m) candidates. L encodings leave about 2^H / prod_i q_i^(n - m_i). Sizing each
    m_i as if its instance were alone makes each factor q_i^(n-m_i) ~ 2^H / (slack), so the
    product overshoots 2^H and x is pinned. Sizing every m_i for the product Q = prod q_i
    (log_Q units) keeps the total like a single field of size Q.

    This script measures it exactly by brute force over a small box, so the claim in
    docs/Q_SWEEP.md section 6 does not rest on the heuristic.

Usage:
    .venv/bin/python tools/crt_leakage_demo.py
"""

import _path  # noqa: F401  -- keeps the tools/ convention

import itertools

import galois
import numpy as np

N, TRIALS, SEED = 8, 30, 0
BOX = np.array(list(itertools.product([-1, 0, 1], repeat=N)))    # 3^8 = 6561 candidates


def consistent(q: int, m: int, x, rng) -> np.ndarray:
    """
    Encodes x once over GF(q) with a fresh key and code, and returns which candidates in BOX
    the provider cannot rule out from (x~, G).

    :param q: Prime field size
    :param m: Key length
    :param x: The true data vector, an element of BOX
    :param rng: Generator for k and G
    :return: Boolean mask over BOX
    """
    GF = galois.GF(q)
    G = GF.Random((m, N), seed=rng)
    k = GF.Random(m, seed=rng)
    x_tilde = GF(x % q) + k @ G
    H = G.null_space()                      # rows h with G h = 0, so (kG) . h = 0
    diff = (np.asarray(x_tilde)[None, :].astype(np.int64) - BOX) % q
    return np.all((GF(diff) @ H.T) == 0, axis=1)


def surviving(instances, seed: int) -> int:
    """
    Number of candidates consistent with every encoding in the list.

    :param instances: List of (q, m), one per encoding of the same x
    :param seed: Trial seed
    :return: Candidate count; 1 means x is fully recovered
    """
    rng = np.random.default_rng([SEED, seed])
    x = BOX[rng.integers(len(BOX))]
    mask = np.ones(len(BOX), dtype=bool)
    for q, m in instances:
        mask &= consistent(q, m, x, rng)
    return int(mask.sum())


CASES = {
    "one encoding, q = 11, m = 6": [(11, 6)],
    "one encoding, q = 13, m = 6": [(13, 6)],
    "two encodings, each m sized ALONE (6 and 6)": [(11, 6), (13, 6)],
    "one encoding over a single prime ~ Q = 143 (q = 139), m = 7": [(139, 7)],
    "two encodings, m sized JOINTLY for Q = 143 (7 and 7)": [(11, 7), (13, 7)],
}


def main():
    print(f"x uniform over {{-1,0,1}}^{N}: {len(BOX)} candidates a priori "
          f"({np.log2(len(BOX)):.1f} bits), {TRIALS} seeded trials per case\n")
    print(f"  {'case':<64} {'median':>7} {'min':>5} {'x recovered':>12}")
    for name, cfg in CASES.items():
        counts = [surviving(cfg, s) for s in range(TRIALS)]
        print(f"  {name:<64} {int(np.median(counts)):>7} {min(counts):>5} "
              f"{sum(c == 1 for c in counts):>6}/{TRIALS}")


if __name__ == "__main__":
    main()
