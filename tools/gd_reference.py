"""
File: gd_reference.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: The plaintext side of the Phase 1 experiments (docs/GD_LINEAR_REGRESSION.md
    sections 4.4 and 10.5): references the scheme's training run is compared against, the
    least-squares optimum, and the convergence metrics. All written by hand -- no
    scikit-learn, no numpy.linalg solver.

    This is the HARNESS. It holds the plaintext data, which the trainer in icc/training.py
    never does. The references are:

      integer   the same loop as the scheme (training.run_gd) with the gradient computed
                exactly in integers from the stored codes. The scheme must reproduce it BIT
                FOR BIT -- same codes, same floats -- at every step, because encoding adds no
                arithmetic error; any difference is a bug.
      float_deq float GD on the dequantised codes: separates quantising the data from
                quantising the weights.
      float     float GD on the real-valued data before quantisation: the UNENCODED
                baseline. Encoded vs this, in steps, is the cost of quantisation; in
                wall-clock it is the cost of the scheme.

    Every float run uses the same step size, run length, initial point and clip box as the
    scheme, so the only differences are the ones named above.
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
import time
from typing import Optional, Tuple

import numpy as np

from training import GDSchedule, GradientFn


# ─────────────────────────────────────────────────────────────────────────────
# Linear algebra by hand
# ─────────────────────────────────────────────────────────────────────────────

def jacobi_eigh(A: np.ndarray, tol: float = 1e-14, max_sweeps: int = 100):
    """
    Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.

    Each rotation zeroes one off-diagonal pair; sweeping over all pairs until the
    off-diagonal mass is negligible converges quadratically. Exact enough for the 2x2 to
    11x11 matrices here, and short enough to read in full.

    :param A: Symmetric matrix
    :param tol: Stop when the off-diagonal Frobenius norm falls below tol * ||A||
    :param max_sweeps: Safety cap
    :return: (eigenvalues ascending, eigenvectors as columns)
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    V = np.eye(n)
    scale = max(float(np.sqrt(np.sum(A * A))), 1e-300)
    for _ in range(max_sweeps):
        off = math.sqrt(float(np.sum((A - np.diag(np.diag(A))) ** 2)))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for r in range(p + 1, n):
                if abs(A[p, r]) < 1e-300:
                    continue
                theta = (A[r, r] - A[p, p]) / (2.0 * A[p, r])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                J = np.eye(n)
                J[p, p] = J[r, r] = c
                J[p, r], J[r, p] = s, -s
                A = J.T @ A @ J
                V = V @ J
    vals = np.diag(A).copy()
    order = np.argsort(vals)
    return vals[order], V[:, order]


def condition_number(H: np.ndarray, rel_tol: float = 1e-10) -> Tuple[float, float, float]:
    """
    Condition number on the RANGE of a positive semidefinite matrix: largest eigenvalue over
    the smallest one that is not numerically zero. For P >= n_s the sample Hessian is
    singular, GD converges to the minimum-norm solution, and only the range governs speed.

    :return: (kappa, lambda_max, lambda_min on the range)
    """
    vals, _ = jacobi_eigh(H)
    lam_max = float(vals[-1])
    nonzero = vals[vals > rel_tol * lam_max]
    lam_min = float(nonzero[0])
    return lam_max / lam_min, lam_max, lam_min


def min_norm_solve(H: np.ndarray, b: np.ndarray, rel_tol: float = 1e-10) -> np.ndarray:
    """
    Minimum-norm solution of H w = b for symmetric PSD H, via its eigen-decomposition.
    Equals the ordinary solution when H is invertible, and is the point GD from w = 0
    converges to when it is not.
    """
    vals, V = jacobi_eigh(H)
    keep = vals > rel_tol * vals[-1]
    coef = (V[:, keep].T @ b) / vals[keep]
    return V[:, keep] @ coef


def augment(X: np.ndarray, intercept: bool) -> np.ndarray:
    """:return: [X, 1] with the constant column when there is an intercept, else X"""
    return np.hstack([X, np.ones((X.shape[0], 1))]) if intercept else X


def least_squares_optimum(X: np.ndarray, y: np.ndarray, intercept: bool) -> np.ndarray:
    """
    The least-squares fit, from the normal equations solved by hand (minimum norm if the
    design is rank-deficient).
    """
    Xa = augment(X, intercept)
    return min_norm_solve(Xa.T @ Xa, Xa.T @ y)


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, intercept: bool) -> float:
    """:return: Half the mean squared residual; w may be one model or a (T, width) stack"""
    Xa = augment(X, intercept)
    r = np.atleast_2d(w) @ Xa.T - y
    out = 0.5 * np.mean(r * r, axis=1)
    return out if np.ndim(w) > 1 else float(out[0])


# ─────────────────────────────────────────────────────────────────────────────
# References
# ─────────────────────────────────────────────────────────────────────────────

def integer_gradient(X_codes: np.ndarray, y_codes: np.ndarray, w_codes, b_code,
                     f_w: int, f_X: int) -> list:
    """
    The exact integer gradient the scheme decodes, computed directly from the codes: the
    integer oracle of query.least_squares_gradient, all components at once.

        residual_i = sum_k x^_ik w^_k + 2^{f_X} b^ - 2^{f_w} y^_i
        g_j = sum_i residual_i x^_ij,     g_b = sum_i residual_i 2^{f_X}

    int64 is exact here: |g| <= B = n 2^23 < 2^31 at the Phase 1 format, and an assertion
    guards larger formats.
    """
    Xc = np.asarray(X_codes, dtype=np.int64)
    yc = np.asarray(y_codes, dtype=np.int64)
    w = np.asarray(w_codes, dtype=np.int64)
    if (np.abs(Xc).max(initial=0) * max(int(np.abs(w).sum()), 1) * Xc.shape[0] * 4
            > 2 ** 62):
        raise OverflowError("format too large for exact int64 gradients")
    residual = Xc @ w - 2 ** f_w * yc
    if b_code is not None:
        residual = residual + 2 ** f_X * int(b_code)
    grad = [int(v) for v in residual @ Xc]
    if b_code is not None:
        grad.append(int(residual.sum()) * 2 ** f_X)
    return grad


def integer_gradient_fn(X_codes, y_codes, schedule: GDSchedule) -> GradientFn:
    """:return: GradientFn computing the exact integer gradient in plaintext"""
    def gradient(w_codes, b_code):
        t0 = time.perf_counter()
        g = integer_gradient(X_codes, y_codes, w_codes, b_code, schedule.w_spec.f,
                             schedule.f_X)
        return g, time.perf_counter() - t0, 0.0
    return gradient


def float_gd(X: np.ndarray, y: np.ndarray, schedule: GDSchedule,
             w0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unencoded gradient descent in float, by hand: one matrix-vector product per step, the
    same step size, initial point and clip box as the scheme, and no quantisation at all.

    :param X: Feature matrix (real-valued, or dequantised codes)
    :param y: Labels
    :param schedule: The scheme's schedule (eta, steps, clip bounds, intercept or not)
    :param w0: Initial model; zero by default
    :return: (iterates of shape (T+1, width), per-step seconds of shape (T,))
    """
    intercept = schedule.b_spec is not None
    Xa = augment(np.asarray(X, dtype=float), intercept)
    n_s, P = X.shape
    w = np.zeros(Xa.shape[1]) if w0 is None else np.asarray(w0, dtype=float).copy()
    lo = np.full(Xa.shape[1], -schedule.w_spec.clip)
    hi = -lo
    if intercept:
        lo[P], hi[P] = -schedule.b_spec.clip, schedule.b_spec.clip
    ws, times = [w.copy()], np.zeros(schedule.steps)
    for t in range(schedule.steps):
        t0 = time.perf_counter()
        g = Xa.T @ (Xa @ w - y)
        w = np.minimum(np.maximum(w - (schedule.eta / n_s) * g, lo), hi)
        times[t] = time.perf_counter() - t0
        ws.append(w.copy())
    return np.array(ws), times


# ─────────────────────────────────────────────────────────────────────────────
# Convergence metrics (docs/GD_LINEAR_REGRESSION.md section 10.5)
# ─────────────────────────────────────────────────────────────────────────────

def suboptimality(X, y, ws: np.ndarray, intercept: bool,
                  w_opt: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalised suboptimality (L(w_t) - L*) / (L(w_0) - L*) on the given data.

    Normalising by the initial gap rather than by L* keeps it defined when L* = 0, which
    happens for P >= n_s (the model interpolates). It starts at 1 and falls to 0 at the
    optimum, so a tolerance of 1e-2 means "99% of the initial excess loss removed".

    :param ws: Iterates, shape (T+1, width)
    :param w_opt: The optimum on this data; computed by hand if not given
    :return: Array of length T+1
    """
    w_opt = least_squares_optimum(X, y, intercept) if w_opt is None else w_opt
    L_star = loss(X, y, w_opt, intercept)
    L = loss(X, y, ws, intercept)
    gap0 = L[0] - L_star
    return (L - L_star) / gap0 if gap0 > 0 else np.zeros_like(L)


def steps_to_tolerance(excess: np.ndarray, tol: float) -> Optional[int]:
    """
    The first step after which the suboptimality stays at or below tol, or None if it never
    settles there (including when tol is below the run's quantisation floor).
    """
    above = np.nonzero(excess > tol)[0]
    if len(above) == 0:
        return 0
    first = int(above[-1]) + 1
    return first if first < len(excess) else None


def plateau(excess: np.ndarray, frac: float = 0.1) -> float:
    """:return: Median suboptimality over the last frac of the run (the run's floor)"""
    tail = excess[-max(1, int(len(excess) * frac)):]
    return float(np.median(tail))
