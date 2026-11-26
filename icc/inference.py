"""
File: inference.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Inference with a fixed, public linear model THROUGH the scheme, from the
    client's side (docs/INFERENCE.md).

    The client uploads its feature matrix once and holds a model trained elsewhere. For every
    stored row i and every output k it sends the degree-1 query

        z_ik(x) = sum_j W^[k, j] * x[i P + j]  +  2^{f_X} * b^[k]        (query.linear_score)

    and decodes all of them in one solve. At the scale 2^{f_X + f_w}, one example:

        w = [0.5, -1.25], b = 0.25, f_w = 4  ->  w^ = [8, -20], b^ = 4   (public coefficients)
        x_i = [2.0, 1.0],      f_X = 8      ->  x^_i = [512, 256]       (stored, encoded)
        z^_i = 8*512 - 20*256 + 2^8*4 = 0    ->  z_i = 0 / 2^12 = 0.0    (= 0.5*2 - 1.25 + 0.25)

    A linear-regression prediction is z itself. Logistic and softmax models use the same
    queries: the client applies the sigmoid or the softmax to the decoded scores, so the
    non-polynomial link never reaches the server and the query stays at degree 1.

    WHY THIS IS COVERED BY THEOREM 1 and live gradient descent is not: every query here is a
    function of the public model only, fixed before the first query is sent and never chosen
    from a decoded result, so the provider's view is X~ plus queries that are independent of
    the data. Each worker's output is computed from X~ and the query, so it adds nothing.
    That holds only if the model was NOT fitted on the stored rows (docs/INFERENCE.md
    section 2 lists the three conditions); this module cannot check that, the caller must.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from query import QueryResult, linear_score
from quantize import lift_balanced


def score_queries(n_rows: int, n_features: int, W_codes, b_codes=None,
                  f_X: Optional[int] = None) -> List:
    """
    Every score query for a model with K outputs over n_rows stored rows.

    :param n_rows: Number of stored rows
    :param n_features: Number of features P
    :param W_codes: Quantised weights, integer codes, shape (K, P) (or (P,) for K = 1)
    :param b_codes: Quantised intercepts at the weights' f_w, shape (K,), or None
    :param f_X: Fractional bits of the stored data; required with an intercept
    :return: Polynomials in row-major order: index i * K + k is output k of row i
    """
    W = np.atleast_2d(np.asarray(W_codes, dtype=np.int64))
    b = [None] * W.shape[0] if b_codes is None else [int(v) for v in np.atleast_1d(b_codes)]
    return [linear_score(i, n_features, W[k], b[k], f_X)
            for i in range(n_rows) for k in range(W.shape[0])]


def predict_scores(client, server, n_rows: int, n_features: int, W_codes, b_codes=None,
                   f_X: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Scores every stored row through the scheme: n_rows * K degree-1 queries, one decode.

    :param client: Client holding the key of the uploaded data
    :param server: Server holding the encoded data
    :param n_rows: Number of stored rows
    :param n_features: Number of features P
    :param W_codes: Quantised weights, shape (K, P) or (P,)
    :param b_codes: Quantised intercepts, shape (K,), or None
    :param f_X: Fractional bits of the stored data; required with an intercept
    :return: (integer scores at scale 2^{f_X + f_w}, shape (n_rows, K), as Python ints in an
        object array; dict with 'evaluate' and 'decode' seconds)
    :raises RuntimeError: if the server refuses any query, with its reason; the model's
        weights must lie inside the ScoreBudget the field was sized for
    """
    K = np.atleast_2d(np.asarray(W_codes)).shape[0]
    polys = score_queries(n_rows, n_features, W_codes, b_codes, f_X)
    t0 = time.perf_counter()
    responses: List[QueryResult] = [server.compute_request(f) for f in polys]
    t1 = time.perf_counter()
    refused = [r for r in responses if not r.ok]
    if refused:
        raise RuntimeError(f"server refused a score query ({refused[0].status.value}): "
                           f"{refused[0].reason}")
    decoded = client.decode_results(responses)
    t2 = time.perf_counter()
    values = np.array([int(d.value) for d in decoded], dtype=object)
    scores = lift_balanced(values, client.context.q).reshape(n_rows, K)
    return scores, dict(evaluate=t1 - t0, decode=t2 - t1)


def descale(scores, f_X: int, f_w: int) -> np.ndarray:
    """
    :param scores: Integer scores at scale 2^{f_X + f_w}
    :param f_X: Fractional bits of the data
    :param f_w: Fractional bits of the weights
    :return: Real-valued scores
    """
    return np.asarray(scores, dtype=float) / 2.0 ** (f_X + f_w)
