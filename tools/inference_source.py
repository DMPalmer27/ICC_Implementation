"""
File: inference_source.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Synthetic sources for the inference study (docs/INFERENCE.md section 5): a
    feature distribution with EXACT Renyi entropies of the stored codes, and a public task
    design (linear, logistic or softmax) that generates labels for training and scoring.

    What is stored differs from the training study (tools/synthetic_source.py): inference
    uploads the FEATURE MATRIX ONLY. Labels never leave the client; it uses them after
    decoding, to measure how good the predictions are. So the stored vector is
    n = N * P i.i.d. codes, and the entropies are exact by the product rule:

        H_p(X)          = N * P * H_p(code)            Renyi entropy adds over independent coords
        max_R H_p(X_R)  = r * H_p(code)                every r-subset is r i.i.d. codes

    with H_p(code) from the exact discretised pmf (synthetic_source.discretised_pmf). For
    equicorrelated Gaussian features (rho > 0) the product rule fails, and the rigorous
    bounds of quantize.entropy_bounds are used instead: H_p(X) >= n f_X - log2 sup(joint
    density) and max_R H_p(X_R) <= r log2(levels).

    The task design is a fixed, public function of the family's population moments, never
    of a sample. Labels are drawn from the REAL feature values; the model the client deploys
    is then FITTED in floating point on an independent training sample, exactly as a real
    client would (the scheme only ever sees the stored rows, which the fit never touched).
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from quantize import FixedPointSpec, to_integer
from synthetic_source import FAMILIES, EntropyResult, discretised_pmf, renyi_bits

TASKS = ("linear", "logistic", "softmax")
N_CLASSES = 3                  # softmax task
LOGIT_SCALE = 2.5              # logistic / softmax separation: std of the true logit
SNR = 4.0                      # linear task: Var(signal) / Var(noise)
DESIGN_SEED = 20261126         # fixed public seed for the softmax class directions


@dataclass(frozen=True)
class FeatureSource:
    """
    The distribution of one stored row. Every field is public and fixed before any data.

    :param family: Marginal family, a key of synthetic_source.FAMILIES
    :param n_features: P
    :param rho: Pairwise correlation; > 0 only for family 'gauss'
    :param f_X: Fractional bits of the stored features
    :param clip_X: Clip bound A_X
    """
    family: str = "gauss"
    n_features: int = 5
    rho: float = 0.0
    f_X: int = 8
    clip_X: float = 4.0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; choose from {list(FAMILIES)}")
        if self.rho and self.family != "gauss":
            raise ValueError("correlation is defined for the Gaussian family only")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")

    @property
    def name(self) -> str:
        """:return: Short label, e.g. 'gauss' or 'gauss-rho0.9'"""
        return f"gauss-rho{self.rho:g}" if self.rho else self.family

    @property
    def x_spec(self) -> FixedPointSpec:
        return FixedPointSpec(f=self.f_X, clip=self.clip_X)

    @cached_property
    def code_pmf(self) -> np.ndarray:
        """:return: Exact pmf of one stored code (independent features), index code + max_int"""
        return discretised_pmf(FAMILIES[self.family], self.x_spec)

    @cached_property
    def moments(self) -> tuple:
        """
        :return: (mean, variance) of one REAL feature value, from the exact pmf at a fine
            grid (f = 12) so the design does not depend on f_X
        """
        spec = FixedPointSpec(12, self.clip_X)
        pmf = discretised_pmf(FAMILIES[self.family], spec)
        v = np.arange(-spec.max_int, spec.max_int + 1) / 2.0 ** 12
        mean = float(np.sum(pmf * v))
        return mean, float(np.sum(pmf * (v - mean) ** 2))

    @cached_property
    def cov(self) -> np.ndarray:
        """:return: Population covariance of one real row (design, not sample)"""
        _, var = self.moments
        P = self.n_features
        return var * ((1 - self.rho) * np.eye(P) + self.rho * np.ones((P, P)))

    def sample(self, rng: np.random.Generator, n_rows: int) -> np.ndarray:
        """
        :param rng: Generator for this draw
        :param n_rows: Number of rows N
        :return: Real feature matrix, shape (N, P)
        """
        P = self.n_features
        if self.rho:
            z0 = rng.standard_normal((n_rows, 1))
            return math.sqrt(self.rho) * z0 + math.sqrt(1 - self.rho) * rng.standard_normal((n_rows, P))
        return FAMILIES[self.family].sampler(rng, (n_rows, P))

    def quantise(self, X: np.ndarray) -> np.ndarray:
        """:return: Stored integer codes of a real feature matrix"""
        return to_integer(X, self.x_spec)

    def entropies(self, n_rows: int, p: int = 2, r: int = 5) -> EntropyResult:
        """
        Theorem 1's entropy inputs for the stored vector of N rows, in bits.

        :param n_rows: Number of stored rows N (n = N * P)
        :param p: Renyi order
        :param r: Subset size
        :return: EntropyResult: method 'exact' for independent features, 'bounds' for rho > 0
        """
        n = n_rows * self.n_features
        if not self.rho:
            h = renyi_bits(self.code_pmf, p)
            return EntropyResult(H_X_bits=n * h, max_R_bits=min(r, n) * h, method="exact",
                                 max_R_exact=True, h_feature_bits=h)
        P, rho = self.n_features, self.rho
        log2_det = ((P - 1) * math.log2(1 - rho) if P > 1 else 0.0) + math.log2(1 + (P - 1) * rho)
        log2_sup_row = -0.5 * P * math.log2(2 * math.pi) - 0.5 * log2_det
        return EntropyResult(H_X_bits=n * self.f_X - n_rows * log2_sup_row,
                             max_R_bits=min(r, n) * self.x_spec.log2_levels, method="bounds",
                             max_R_exact=False)


@dataclass(frozen=True)
class TaskDesign:
    """
    The true model that generates labels, fixed by the source's population moments.

        linear    y = x . w* + b* + e,       Var(x . w*) = SNR * Var(e),  Var y = 1
        logistic  y ~ Bernoulli(sigmoid(s)), s = (x - mu) . u * LOGIT_SCALE
        softmax   y ~ Categorical(softmax(S)), S_k = (x - mu) . u_k * LOGIT_SCALE

    with u, u_k unit-variance directions (u' Sigma u = 1). The linear direction alternates in
    sign with decreasing magnitude, as in synthetic_source; the softmax directions are a fixed
    public Gaussian draw (DESIGN_SEED).

    :param task: 'linear', 'logistic' or 'softmax'
    :param source: The feature source
    """
    task: str
    source: FeatureSource

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; choose from {TASKS}")

    @cached_property
    def directions(self) -> np.ndarray:
        """:return: Unit-variance directions, shape (K, P); K = 1 except for softmax"""
        P, S = self.source.n_features, self.source.cov
        if self.task == "softmax":
            D = np.random.default_rng(DESIGN_SEED).standard_normal((N_CLASSES, P))
        else:
            D = np.array([[(-1) ** k / (1 + 0.5 * k) for k in range(P)]])
        return D / np.sqrt(np.einsum("kp,pq,kq->k", D, S, D))[:, None]

    def labels(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        :param X: Real feature matrix, shape (N, P)
        :param rng: Generator for the label noise
        :return: Labels: real (linear), {0,1} (logistic) or {0..K-1} (softmax)
        """
        mu, _ = self.source.moments
        U = self.directions
        if self.task == "linear":
            a = math.sqrt(SNR / (1.0 + SNR))
            signal = (X - mu) @ U[0] * a
            return signal + 0.25 + rng.standard_normal(len(X)) * math.sqrt(1.0 / (1.0 + SNR))
        S = (X - mu) @ U.T * LOGIT_SCALE
        if self.task == "logistic":
            return (rng.random(len(X)) < 1.0 / (1.0 + np.exp(-S[:, 0]))).astype(int)
        S = S - S.max(axis=1, keepdims=True)
        prob = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
        u = rng.random((len(X), 1))
        return np.minimum((u > np.cumsum(prob, axis=1)).sum(axis=1), N_CLASSES - 1)
