"""
File: real_data.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Real datasets for the inference study (docs/INFERENCE.md section 6), and the
    PROVISIONAL entropy bound used to size their key.

    PREPARATION. Each dataset is shuffled with a public seed (wine is stored sorted by class)
    and split in half: the model is fitted on the first half, the second half is uploaded
    and scored. Standardisation (feature means and standard deviations, and for regression
    the label's) is computed on the training half only, because statistics of the stored
    rows would make the stored codes, and the model, depend on the stored data.

    ENTROPY (provisional, pending Raviv; CLAUDE.md open issues 2 and 3). Real features are
    correlated, so the product formula is invalid, and quantize.entropy_bounds' density
    route is used instead, with two corrections that real data needs:

      1. discrete features (at most DISCRETE_MAX distinct training values, e.g. diabetes
         'sex') have no density; they are credited 0 bits, and the density of the remaining
         features is fitted separately within each discrete group, keeping the largest;
      2. recorded precision: values were recorded on a grid of step s_j (two decimals, say),
         so a feature carries at most b_j = log2(sigma_j / s_j) bits below sigma however fine
         the fixed-point grid is; each continuous feature is credited min(f_X, b_j) bits.

    Per stored row, with the continuous features' joint density bounded by the Gaussian fit
    M = (2 pi)^(-P_c/2) det(S)^(-1/2) on the training half (standardised units),

        H_p(row) >= sum_{continuous j} min(f_X, b_j) - log2 M          (min-entropy bound)

    and H_p(X) >= N_store * that, since stored rows are independent. The upper bound
    max_R H_p(X_R) <= r log2(levels) needs no model at all. The Gaussian M is a MODEL, not a
    guarantee -- that is the part Raviv has to rule on.
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
from dataclasses import dataclass

import numpy as np

from quantize import FixedPointSpec, to_integer
from synthetic_source import EntropyResult

DATASETS = {"diabetes": "linear", "wine": "softmax"}
SPLIT_SEED = 20261126
DISCRETE_MAX = 10


@dataclass
class Prepared:
    """One dataset, split and standardised on its training half."""
    name: str
    task: str
    feature_names: list
    X_train: np.ndarray        # standardised
    y_train: np.ndarray        # standardised for regression
    X_store: np.ndarray        # standardised with the TRAINING statistics
    y_store: np.ndarray
    X_train_raw: np.ndarray    # original units, for the recorded-precision estimate
    y_scale: float             # regression: label std of the training half, else 1


def load(name: str) -> Prepared:
    """
    :param name: 'diabetes' or 'wine' (also 'breast_cancer', deferred from the paper)
    :return: Prepared dataset
    """
    from sklearn import datasets
    loader = {"diabetes": datasets.load_diabetes, "wine": datasets.load_wine,
              "breast_cancer": datasets.load_breast_cancer}[name]
    d = loader()
    X, y = np.asarray(d.data, dtype=float), np.asarray(d.target)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(X))
    X, y = X[perm], y[perm]
    half = (len(X) + 1) // 2
    Xtr, Xst, ytr, yst = X[:half], X[half:], y[:half], y[half:]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    task = DATASETS.get(name, "logistic")
    y_scale = 1.0
    if task == "linear":
        ym, y_scale = float(ytr.mean()), float(ytr.std())
        ytr, yst = (ytr - ym) / y_scale, (yst - ym) / y_scale
    return Prepared(name, task, list(d.feature_names), (Xtr - mu) / sd, ytr, (Xst - mu) / sd,
                    yst, Xtr, y_scale)


def recorded_bits(X_raw: np.ndarray) -> np.ndarray:
    """
    Recorded precision of each feature in bits below one standard deviation,
    b_j = log2(sigma_j / s_j), with s_j the smallest gap between distinct training values.

    :param X_raw: Training features in original units
    :return: b_j per feature (may be negative for coarse discrete features)
    """
    out = []
    for col in X_raw.T:
        gaps = np.diff(np.unique(col))
        gaps = gaps[gaps > 1e-12 * max(1.0, np.abs(col).max())]
        out.append(math.log2(col.std() / gaps.min()) if len(gaps) else 0.0)
    return np.array(out)


def entropy_provisional(prep: Prepared, n_store_rows: int, x_spec: FixedPointSpec,
                        p: int = 2, r: int = 5) -> tuple:
    """
    The provisional bounds of the module docstring, from the TRAINING half only.

    :param prep: Prepared dataset
    :param n_store_rows: Number of stored rows
    :param x_spec: Storage format
    :param p: Renyi order (the bounds hold for every p > 1)
    :param r: Subset size
    :return: (EntropyResult with method 'provisional', dict of diagnostics)
    """
    X, raw = prep.X_train, prep.X_train_raw
    P = X.shape[1]
    distinct = np.array([len(np.unique(raw[:, j])) for j in range(P)])
    discrete = distinct <= DISCRETE_MAX
    cont = ~discrete
    b = recorded_bits(raw)
    f_eff = np.clip(np.minimum(x_spec.f, b), 0.0, None)
    bits_cells = float(f_eff[cont].sum())

    # Gaussian fit of the continuous block, within each discrete group; keep the largest density
    keys = [tuple(row) for row in raw[:, discrete]] if discrete.any() else [()] * len(raw)
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    Pc = int(cont.sum())
    log2_M = -math.inf
    for idx in groups.values():
        if len(idx) < Pc + 2:
            continue
        S = np.atleast_2d(np.cov(X[np.ix_(idx, np.where(cont)[0])].T))
        _, logdet = np.linalg.slogdet(S)
        log2_M = max(log2_M, -0.5 * Pc * math.log2(2 * math.pi) - 0.5 * logdet / math.log(2))
    if log2_M == -math.inf:
        S = np.atleast_2d(np.cov(X[:, cont].T))
        _, logdet = np.linalg.slogdet(S)
        log2_M = -0.5 * Pc * math.log2(2 * math.pi) - 0.5 * logdet / math.log(2)

    h_row = max(0.0, bits_cells - log2_M)
    n = n_store_rows * P
    result = EntropyResult(H_X_bits=n_store_rows * h_row,
                           max_R_bits=min(r, n) * x_spec.log2_levels,
                           method="provisional", max_R_exact=False)
    diag = dict(discrete=[prep.feature_names[j] for j in np.where(discrete)[0]],
                recorded_bits=b, f_eff=f_eff, log2_M=log2_M, h_row_bits=h_row,
                h_symbol_bits=h_row / P, n_groups=len(groups))
    return result, diag


def stored_codes(prep: Prepared, x_spec: FixedPointSpec) -> np.ndarray:
    """:return: Integer codes of the stored half"""
    return to_integer(prep.X_store, x_spec)
