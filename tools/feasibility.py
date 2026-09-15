"""
File: feasibility.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Parameter feasibility instrument for the ICC scheme. Given the scheme
    parameters, reports the key size m required by Theorem 1, the resulting worker count
    lambda = |I_{d,m}|, and -- critically -- the MEASURED cost of building and solving the
    interpolation system rather than a modelled one.

    Why measured: the earlier planning model priced decoding at O(lambda^3) from a dense
    Gaussian elimination. That model is arithmetically right but applied to the wrong matrix.
    The interpolation matrix on the standard information set satisfies

        M[t, a] != 0  <=>  supp(a) subset supp(t)

    so with total degree d each row has exactly C(2d, d) nonzeros, independent of m, and M is
    ~0.5% dense at m=48. Measured solve at lambda=1225 is 0.16 s against 17.6 s for a dense
    random matrix of the same size. Anything that has to appear in the paper should come from
    measure(), not estimate().

Usage:
    .venv/bin/python tools/feasibility.py            # default sweep, prints a table
    .venv/bin/python tools/feasibility.py --csv results/feasibility_sweep.csv
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, asdict, fields

import galois
import numpy as np

from config import SystemContext
from utils import build_monomial_matrix, compute_required_m, get_information_set

# Beyond this the interpolation matrix stops fitting comfortably in memory
# (lambda^2 int64: 1225 -> 12 MB, 6000 -> 288 MB, 12000 -> 1.2 GB).
DEFAULT_MAX_LAMBDA = 6000


@dataclass
class Feasibility:
    """One row of the feasibility table. Timing fields are None when not measured."""
    n: int
    q: int
    h: float
    r: int
    p: int
    epsilon: float
    d: int
    m: int
    lam: int
    matrix_mb: float
    t_build_s: float = None
    t_solve_s: float = None
    t_solve_warm_s: float = None
    exact: bool = None
    verdict: str = ""


def required_m(n: int, q: int, h: float, r: int, p: int, epsilon: float, d: int) -> int:
    """
    Computes the Theorem 1 key size from a per-symbol entropy, without needing real data.

    Delegates to utils.compute_required_m so the bound has a single implementation, including
    its guards. Under the i.i.d. source model H_p(X) = n*h and max_R H_p(X_R) = r*h.

    :param n: Data length
    :param q: Field size
    :param h: Per-symbol p-entropy in log_q units, in [0, 1]
    :param r: Privacy parameter
    :param p: Entropy order
    :param epsilon: Smoothing budget
    :param d: Max total degree
    :return: Minimum key size m satisfying Theorem 1
    """
    if not 0.0 <= h <= 1.0:
        raise ValueError(f"per-symbol entropy h must lie in [0, 1] (log_q units), got {h}")

    ctx = SystemContext(q=q, n=n, d=d, r=r, p=p, epsilon=epsilon)
    ctx.H_p_X = n * h
    ctx.max_H_p_X_R = r * h
    return compute_required_m(ctx)


def estimate(n: int, q: int, h: float, r: int, p: int = 2,
             epsilon: float = 1e-6, d: int = 2) -> Feasibility:
    """
    Pure arithmetic: key size, worker count and matrix footprint, with no field operations.

    Cheap enough to sweep densely. Use measure() for anything that needs a real cost.

    :param n: Data length
    :param q: Field size
    :param h: Per-symbol p-entropy in log_q units
    :param r: Privacy parameter
    :param p: Entropy order
    :param epsilon: Smoothing budget
    :param d: Max total degree
    :return: Feasibility row with timing fields unset
    """
    m = required_m(n, q, h, r, p, epsilon, d)
    lam = math.comb(m + d, d)
    matrix_mb = (lam * lam * 8) / 2 ** 20

    return Feasibility(n=n, q=q, h=h, r=r, p=p, epsilon=epsilon, d=d,
                       m=m, lam=lam, matrix_mb=matrix_mb)


def measure(n: int, q: int, h: float, r: int, p: int = 2,
            epsilon: float = 1e-6, d: int = 2,
            max_lambda: int = DEFAULT_MAX_LAMBDA, seed: int = 0) -> Feasibility:
    """
    Builds the real interpolation system and times it, including a correctness check.

    Times three things separately, because they scale differently and are confusable:
    building M, solving cold, and solving again against the cached matrix (which is what a
    gradient-descent step actually pays once the Client has warmed up).

    :param n: Data length
    :param q: Field size, must be prime for galois.GF
    :param h: Per-symbol p-entropy in log_q units
    :param r: Privacy parameter
    :param p: Entropy order
    :param epsilon: Smoothing budget
    :param d: Max total degree
    :param max_lambda: Skip measurement above this worker count
    :param seed: RNG seed for the right-hand side
    :return: Feasibility row with timings populated, or a skip verdict
    """
    row = estimate(n, q, h, r, p, epsilon, d)

    if row.lam > max_lambda:
        row.verdict = f"skipped (lambda {row.lam:,} > {max_lambda:,})"
        return row

    GF = galois.GF(q)
    exponents = get_information_set(q, row.m, d)
    points = GF(np.asarray(exponents) % q)

    t0 = time.perf_counter()
    M = build_monomial_matrix(GF, points, exponents)
    row.t_build_s = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    b = GF(rng.integers(0, q, size=row.lam))

    t0 = time.perf_counter()
    c = np.linalg.solve(M, b)
    row.t_solve_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.linalg.solve(M, b)
    row.t_solve_warm_s = time.perf_counter() - t0

    # Over a finite field the solve is exact, so this is a hard equality, not a tolerance
    row.exact = bool(np.array_equal(np.asarray(M @ c), np.asarray(b)))
    row.verdict = _verdict(row)
    return row


def _verdict(row: Feasibility) -> str:
    """
    Turns measured cost into a one-word judgement for the table.

    :param row: A measured Feasibility row
    :return: Short verdict string
    """
    if not row.exact:
        return "BROKEN (inexact solve)"
    per_step = (row.t_solve_warm_s or 0.0)
    if row.matrix_mb > 1024:
        return "no (memory)"
    if per_step < 1.0:
        return "yes"
    if per_step < 10.0:
        return "borderline"
    return "no (decode)"


def sweep(configs) -> list:
    """
    Measures a list of parameter dicts in order, printing a table as it goes.

    :param configs: Iterable of kwargs dicts accepted by measure()
    :return: List of Feasibility rows
    """
    header = (f"{'n':>6} {'h':>6} {'m':>6} {'lambda':>8} {'MB':>8} "
              f"{'build':>8} {'solve':>8} {'warm':>8}  verdict")
    print(header)
    print("-" * len(header))

    rows = []
    for cfg in configs:
        row = measure(**cfg)
        rows.append(row)
        fmt = lambda v: "     -  " if v is None else f"{v:8.3f}"
        print(f"{row.n:>6} {row.h:>6.2f} {row.m:>6} {row.lam:>8,} {row.matrix_mb:>8.1f} "
              f"{fmt(row.t_build_s)} {fmt(row.t_solve_s)} {fmt(row.t_solve_warm_s)}  "
              f"{row.verdict}")
    return rows


def write_csv(rows, path: str):
    """
    Writes measured rows to CSV so paper tables stay reproducible.

    :param rows: List of Feasibility rows
    :param path: Destination file path
    """
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Feasibility)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nwrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="ICC parameter feasibility sweep")
    parser.add_argument("--csv", help="write raw rows to this path")
    parser.add_argument("--max-lambda", type=int, default=DEFAULT_MAX_LAMBDA)
    args = parser.parse_args()

    q = int(galois.next_prime(2 ** 30))
    print(f"q = {q} ({q.bit_length()} bits), d = 2, r = 5, p = 2, eps = 1e-6\n")

    # h = 0.33 is the gradient-descent regime (three quantised factors); h = 0.46 is what
    # asymmetric operand scales buy; h = 0.48 is the one-shot sufficient-statistics regime.
    configs = [dict(n=n, q=q, h=h, r=5, max_lambda=args.max_lambda)
               for n in (60, 100, 150, 250)
               for h in (0.33, 0.46)]

    rows = sweep(configs)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
