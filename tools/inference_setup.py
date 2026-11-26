"""
File: inference_setup.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: The pieces of one inference run, shared by the synthetic and real-data studies
    (docs/INFERENCE.md sections 3-5):

        fit the model in float on TRAINING rows  ->  quantise it (public)  ->  ScoreBudget
          -> B -> q -> entropy of the STORED rows (log_q units) -> m -> G, key -> encode, store
          -> every score through the scheme (icc/inference.py) -> link and metrics on the client

    plus the analytic error bound the measured score error is checked against, and the
    degree-2 (training-query) plan used only for the cost comparison.

    Seeds: one SeedSequence per run, spawned by the caller into independent streams; the
    'G' and 'key' streams here are for reproducible experiments only -- a real client must
    not seed its key (Client docstring).
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
import time
import warnings
from typing import Optional

import numpy as np

from client import Client
from config import SystemContext
from quantize import FixedPointSpec, ScoreBudget, choose_field_size, codes_to_field, to_integer
from server import Server
from utils import (compute_leakage_bound, compute_required_m, generate_random_G,
                   get_information_set, lambda_workers)

from gd_setup import field_for

# Privacy settings of the case study, unchanged from QUANTIZATION.md section 10
R_PRIV, P_ORDER, EPSILON, A_CONF = 5, 2, 1e-6, 5.0


# ─────────────────────────────────────────────────────────────────────────────
# The model: fitted in float on training rows, then quantised
# ─────────────────────────────────────────────────────────────────────────────

def fit_model(task: str, X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fits the model the client deploys, in floating point, on TRAINING rows only.

    :param task: 'linear' (ordinary least squares), 'logistic' or 'softmax' (sklearn
        LogisticRegression, default L2 penalty C = 1, lbfgs)
    :param X: Training features, shape (N_train, P)
    :param y: Training labels
    :return: (W, b) with W of shape (K, P) and b of shape (K,); K = 1 except for softmax
    """
    if task == "linear":
        A = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef[None, :-1], coef[-1:]
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    return clf.coef_.copy(), clf.intercept_.copy()


def pow2_ceiling(v: float) -> float:
    """:return: Smallest power of two >= v (at least 2^-4), the public clip for a model's coefficients"""
    return 2.0 ** max(-4, math.ceil(math.log2(max(v, 1e-12))))


def quantise_model(W: np.ndarray, b: np.ndarray, f_w: int) -> dict:
    """
    Quantises a fitted model. The clips are powers of two covering the model, so nothing is
    clipped; they are derived from the model, which is public anyway (it is the query).

    :param W: Float weights, shape (K, P)
    :param b: Float intercepts, shape (K,)
    :param f_w: Fractional bits of weights and intercepts
    :return: dict with w_spec, b_spec, W_codes (K, P), b_codes (K,), and the dequantised
        W_q, b_q the scheme effectively evaluates
    """
    w_spec = FixedPointSpec(f_w, pow2_ceiling(float(np.abs(W).max())))
    b_spec = FixedPointSpec(f_w, pow2_ceiling(float(np.abs(b).max())))
    W_codes, b_codes = to_integer(W, w_spec), to_integer(b, b_spec)
    return dict(w_spec=w_spec, b_spec=b_spec, W_codes=W_codes, b_codes=b_codes,
                W_q=W_codes / 2.0 ** f_w, b_q=b_codes / 2.0 ** f_w)


# ─────────────────────────────────────────────────────────────────────────────
# Storage-time chain
# ─────────────────────────────────────────────────────────────────────────────

def plan_inference(n_rows: int, n_features: int, x_spec: FixedPointSpec,
                   w_spec: FixedPointSpec, b_spec: Optional[FixedPointSpec],
                   entropy) -> dict:
    """
    q, m, lambda and eps_c for storing n_rows rows for degree-1 inference.

    :param entropy: EntropyResult of the stored vector, in bits
    :return: dict with q, log2_q, native, B, H_p_X, max_H (log_q units), h, m, lam, eps_c,
        eps_c_alt (the eq. (9)-(10) exponent of known issue 1), eps_c_bits, method, ctx
    """
    budget = ScoreBudget(n_features, x_spec, w_spec, b_spec)
    choice = choose_field_size(budget.magnitude_bound(), warn_native=False)
    return _finish_plan(choice, n_rows * n_features, 1, x_spec, entropy)


def _finish_plan(choice, n: int, d: int, x_spec: FixedPointSpec, entropy) -> dict:
    q = choice.q
    ctx = SystemContext(q=q, n=n, d=d, r=R_PRIV, p=P_ORDER, epsilon=EPSILON, a=A_CONF,
                        max_input=x_spec.max_int)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy.in_log_q(q, n)
    ctx.m = compute_required_m(ctx)
    eps_c = compute_leakage_bound(ctx)
    p, a, eps, mh = ctx.p, ctx.a, ctx.epsilon, ctx.max_H_p_X_R
    eps_c_alt = (p / (p - 1)) * math.log(
        1 + a * 2 ** ((2 * p - 1) / p) * (1 + q ** (-mh / p)) * eps ** (1 / p), q)
    return dict(q=q, log2_q=choice.log2_q, native=choice.native, B=choice.magnitude_bound,
                H_p_X=ctx.H_p_X, max_H=ctx.max_H_p_X_R, h=ctx.H_p_X / n, m=ctx.m,
                lam=lambda_workers(q, d, ctx.m), eps_c=eps_c, eps_c_alt=eps_c_alt,
                eps_c_bits=eps_c * choice.log2_q, method=entropy.method, ctx=ctx, d=d)


def upload(X_codes: np.ndarray, plan: dict, x_spec: FixedPointSpec, seed) -> tuple:
    """
    Uploads a stored feature matrix through the real scheme, timing each phase.

    :param X_codes: Integer codes, shape (N, P); flattened row-major
    :param plan: From plan_inference
    :param x_spec: Storage format
    :param seed: A SeedSequence, or entropy for one; spawns the 'G' and 'key' streams
    :return: (client, server, timings dict: construct_first, encode, store, build)
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    g_ss, key_ss = ss.spawn(2)
    GF, t_construct = field_for(plan["q"])
    ctx = plan["ctx"]
    t = dict(construct_first=t_construct)
    t0 = time.perf_counter()
    x = codes_to_field(GF, X_codes.ravel(), x_spec)
    G = generate_random_G(GF, ctx.m, ctx.n, rng=np.random.default_rng(g_ss))
    client = Client(ctx, GF, rng=np.random.default_rng(key_ss))
    x_tilde = client.encode_data(x, G)
    t["encode"] = time.perf_counter() - t0
    server = Server(ctx, GF)
    t0 = time.perf_counter()
    server.store_data(x_tilde, G)
    t["store"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    client._interpolation_matrix(server.points, get_information_set(ctx.q, ctx.m, ctx.d))
    t["build"] = time.perf_counter() - t0
    return client, server, t


# ─────────────────────────────────────────────────────────────────────────────
# Client-side post-processing and the error bound
# ─────────────────────────────────────────────────────────────────────────────

def integer_scores(X_codes: np.ndarray, W_codes: np.ndarray, b_codes: np.ndarray,
                   f_X: int) -> np.ndarray:
    """:return: The plaintext integer oracle X^ W^T + 2^{f_X} b^, shape (N, K), Python ints"""
    Xo = np.asarray(X_codes, dtype=object)
    return Xo @ np.asarray(W_codes, dtype=object).T + np.asarray(
        [2 ** f_X * int(v) for v in b_codes], dtype=object)


def score_error_bound(X: np.ndarray, x_spec: FixedPointSpec, W: np.ndarray,
                      f_w: int) -> np.ndarray:
    """
    Deterministic bound on |z_scheme - z_float| for every (row, output), docs/INFERENCE.md
    section 3. With x^/2^f_X = clip(x) + delta, |delta| <= 2^-(f_X+1), and
    w^/2^f_w = w + gamma, |gamma| <= 2^-(f_w+1) (the same for b):

        z_scheme - z_float = sum_j [ w_j (clip(x_j) - x_j) + w_j delta_j + gamma_j clip(x_j)
                                     + gamma_j delta_j ]  +  gamma_b

        |.| <= sum_j |w_j| (|x_j - clip(x_j)| + 2^-(f_X+1)) + 2^-(f_w+1) sum_j |clip(x_j)|
               + P 2^-(f_X+f_w+2) + 2^-(f_w+1)

    The first term is clipping; it is zero unless a feature fell outside +-A_X.

    :param X: REAL stored features, shape (N, P)
    :param x_spec: Storage format
    :param W: FLOAT weights, shape (K, P)
    :param f_w: Fractional bits of the weights and intercept
    :return: Bound, shape (N, K)
    """
    Xc = np.clip(X, -x_spec.clip, x_spec.clip)
    absW = np.abs(W)
    P = X.shape[1]
    data_term = (np.abs(X - Xc) + 2.0 ** -(x_spec.f + 1)) @ absW.T
    weight_term = 2.0 ** -(f_w + 1) * np.abs(Xc).sum(axis=1, keepdims=True)
    return data_term + weight_term + P * 2.0 ** -(x_spec.f + f_w + 2) + 2.0 ** -(f_w + 1)


def task_metrics(task: str, y: np.ndarray, scores: np.ndarray, y_scale: float = 1.0) -> dict:
    """
    Applies the link on the client and scores the predictions against the true labels.

    :param task: 'linear', 'logistic' or 'softmax'
    :param y: True labels of the stored rows
    :param scores: Real-valued scores, shape (N, K)
    :param y_scale: Linear task only: multiplies RMSE back to the label's original units
    :return: linear: rmse, r2; logistic: acc, auc, logloss; softmax: acc, logloss
    """
    from sklearn.metrics import log_loss, roc_auc_score
    if task == "linear":
        pred = scores[:, 0]
        resid = y - pred
        return dict(rmse=float(np.sqrt(np.mean(resid ** 2))) * y_scale,
                    r2=float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)))
    if task == "logistic":
        s = scores[:, 0]
        prob = 1.0 / (1.0 + np.exp(-s))
        return dict(acc=float(np.mean((s > 0).astype(int) == y)),
                    auc=float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan"),
                    logloss=float(log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])))
    S = scores - scores.max(axis=1, keepdims=True)
    prob = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
    return dict(acc=float(np.mean(scores.argmax(axis=1) == y)),
                logloss=float(log_loss(y, prob, labels=list(range(scores.shape[1])))))


def decisions(task: str, scores: np.ndarray) -> Optional[np.ndarray]:
    """:return: The client's decision per row (sign or argmax), or None for regression"""
    if task == "logistic":
        return (scores[:, 0] > 0).astype(int)
    if task == "softmax":
        return scores.argmax(axis=1)
    return None
