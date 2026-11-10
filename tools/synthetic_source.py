"""
File: synthetic_source.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Synthetic regression data drawn from KNOWN distributions, with the exact Renyi
    entropies of the stored integer codes, for the Phase 1 convergence sweep
    (docs/GD_LINEAR_REGRESSION.md sections 3 and 10).

    Everything here is fixed by the design and none of it is estimated from a sample: the
    feature family, its exact discretised pmf, the true weights w*, the intercept b*, the
    noise level, and from those the population moments used to set the learning rate.
    That is what keeps the data-dependent-transform questions (PLAN.md open questions 4, 7)
    out of the experiment.

    THE GENERATIVE MODEL, and the one detail that makes the entropy exact.

        features   x_ij  i.i.d. from the family (or equicorrelated Gaussian, rho > 0)
        stored     x^_ij = round(clip(x_ij, +-A_X) * 2^f_X)                (quantize.to_integer)
        labels     y_i = (x^_i / 2^f_X) . w* + b* + e_i,   e_i i.i.d. noise
        stored     y^_i = round(clip(y_i, +-A_X) * 2^f_X)

    The label is generated from the STORED feature values x^/2^f_X, not from the reals. Then,
    given the stored features, y^ depends on them only through the scalar
    u = x^ . w* + 2^f_X b* (code units), and its conditional pmf is the noise pmf shifted by
    u, plus the two clip atoms. With w* and b* on a dyadic grid 2^-a, u lives on that grid,
    and the joint pmf of one sample's block (x^_i, y^_i) becomes a sum over u under the
    product of the feature pmfs -- computable exactly by 1-D convolutions. The cost of the
    detail is invisible: the label differs from x . w* + b* + e by at most
    2^-(f_X+1) * sum |w*| (about 0.006 at f_X = 7), far below the noise.

    ENTROPIES (bits, order p):
      H_p(X) = n_s * H_p(block)               exact: samples are independent blocks, and Renyi
                                             entropy is additive over INDEPENDENT blocks
      max_R H_p(X_R)                          exact maximum over every r-subset, by a knapsack
                                             over blocks, except that patterns holding the
                                             label and SOME but not all of a block's features
                                             use a valid UPPER bound (the direction Theorem 1
                                             needs); EntropyResult.max_R_exact says whether
                                             the optimum touched one
    Correlated features (rho > 0) break the product structure. Those use the joint-density
    lower bound of quantize.entropy_bounds instead -- the proposed route of
    docs/QUANTIZATION.md section 8, which assumes the clip is wide enough that no saturation
    atom dominates (true for a Gaussian at A_X = 4, f_X = 7).

    Every CDF is written out by hand from math.erf / math.exp / math.atan; nothing is drawn
    from scipy.stats.
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Optional

import numpy as np

from quantize import FixedPointSpec, to_integer

SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# One-dimensional distributions, by hand
# ─────────────────────────────────────────────────────────────────────────────

def _phi_cdf(z: float) -> float:
    """:return: Standard normal CDF at z"""
    return 0.5 * math.erfc(-z / SQRT2)


def _phi_sf(z: float) -> float:
    """:return: Standard normal survival function 1 - CDF at z, accurate in the right tail"""
    return 0.5 * math.erfc(z / SQRT2)


@dataclass(frozen=True)
class Marginal:
    """
    A one-dimensional distribution given by its CDF and survival function (both by hand, so
    cell masses in either tail are taken as a difference of SMALL numbers rather than of
    numbers near 1) and a sampler built from uniform and normal draws.
    """
    name: str
    cdf: Callable[[float], float]
    sf: Callable[[float], float]
    sampler: Callable[[np.random.Generator, tuple], np.ndarray]
    sup_density: float      # sup of the density; inf for families with atoms
    note: str


def _laplace(b: float) -> Marginal:
    cdf = lambda x: 0.5 * math.exp(x / b) if x < 0 else 1.0 - 0.5 * math.exp(-x / b)
    sf = lambda x: 1.0 - 0.5 * math.exp(x / b) if x < 0 else 0.5 * math.exp(-x / b)

    def sample(rng, size):
        u = rng.random(size) - 0.5
        return -b * np.sign(u) * np.log1p(-2.0 * np.abs(u))
    return Marginal("laplace", cdf, sf, sample, 1.0 / (2.0 * b), f"Laplace, scale {b:.4f}")


def _student3() -> Marginal:
    # Y = T / sqrt(3) with T ~ Student-t(3 dof), so Var Y = 1. The t(3) CDF has the closed
    # form 1/2 + (1/pi)[t/(sqrt3 (1 + t^2/3)) + atan(t/sqrt3)]; at t = sqrt3 y it reads
    # 1/2 + (y/(1+y^2) + atan y)/pi.
    cdf = lambda y: 0.5 + (y / (1.0 + y * y) + math.atan(y)) / math.pi
    sf = lambda y: cdf(-y)

    def sample(rng, size):
        z = rng.standard_normal(size)
        chi2 = np.sum(rng.standard_normal(size + (3,)) ** 2, axis=-1)
        return z / np.sqrt(chi2 / 3.0) / SQRT3
    # density of Y at 0: sqrt3 * f_T(0) = sqrt3 * 2 / (pi sqrt3) = 2/pi
    return Marginal("student3", cdf, sf, sample, 2.0 / math.pi,
                    "Student-t, 3 dof, scaled to variance 1")


def _lognormal(s: float) -> Marginal:
    sd = math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
    cdf = lambda y: 0.0 if y <= 0 else _phi_cdf(math.log(y * sd) / s)
    sf = lambda y: 1.0 if y <= 0 else _phi_sf(math.log(y * sd) / s)
    sample = lambda rng, size: np.exp(s * rng.standard_normal(size)) / sd
    # density of exp(sZ)/sd peaks at its mode exp(-s^2)/sd
    mode = math.exp(-s * s) / sd
    peak = 1.0 / (mode * s * math.sqrt(2 * math.pi)) * math.exp(
        -(math.log(mode * sd) ** 2) / (2 * s * s))
    return Marginal("lognormal", cdf, sf, sample, peak,
                    f"exp(N(0, {s}^2)) rescaled to variance 1")


def _mixture(name, comps, note, atoms=()) -> Marginal:
    """Mixture of Gaussians (weight, mean, sd) plus point masses (weight, location)."""
    def cdf(x):
        return (sum(w * _phi_cdf((x - mu) / sd) for w, mu, sd in comps)
                + sum(w for w, loc in atoms if loc <= x))

    def sf(x):
        return (sum(w * _phi_sf((x - mu) / sd) for w, mu, sd in comps)
                + sum(w for w, loc in atoms if loc > x))

    weights = [w for w, _, _ in comps] + [w for w, _ in atoms]

    def sample(rng, size):
        which = rng.choice(len(weights), size=size, p=weights)
        out = np.zeros(size)
        for i, (_, mu, sd) in enumerate(comps):
            mask = which == i
            out[mask] = mu + sd * rng.standard_normal(int(mask.sum()))
        for i, (_, loc) in enumerate(atoms):
            out[which == len(comps) + i] = loc
        return out
    sup = math.inf if atoms else max(
        sum(w / (sd * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sd) ** 2)
            for w, mu, sd in comps)
        for x in np.linspace(-4, 4, 4001))
    return Marginal(name, cdf, sf, sample, sup, note)


_BIMODAL_SD = math.sqrt(1.0 - 0.81)
_DISCRETE_LEVEL = math.sqrt(1.5)

FAMILIES: Dict[str, Marginal] = {
    "gauss": Marginal("gauss", _phi_cdf, _phi_sf,
                      lambda rng, size: rng.standard_normal(size),
                      1.0 / math.sqrt(2 * math.pi), "N(0, 1)"),
    "uniform": Marginal(
        "uniform",
        lambda x: min(max((x + SQRT3) / (2 * SQRT3), 0.0), 1.0),
        lambda x: min(max((SQRT3 - x) / (2 * SQRT3), 0.0), 1.0),
        lambda rng, size: rng.uniform(-SQRT3, SQRT3, size),
        1.0 / (2 * SQRT3), "U[-sqrt3, sqrt3]"),
    "laplace": _laplace(1.0 / SQRT2),
    "student3": _student3(),
    "exponential": Marginal(
        "exponential",
        lambda x: 0.0 if x <= 0 else -math.expm1(-x),
        lambda x: 1.0 if x <= 0 else math.exp(-x),
        lambda rng, size: -np.log1p(-rng.random(size)),
        1.0, "Exp(1): mean 1, variance 1"),
    "lognormal": _lognormal(0.75),
    "spiked": _mixture("spiked", [(0.2, 0.0, math.sqrt(5.0))],
                       "80% exactly 0, 20% N(0, 5)", atoms=[(0.8, 0.0)]),
    "bimodal": _mixture("bimodal", [(0.5, -0.9, _BIMODAL_SD), (0.5, 0.9, _BIMODAL_SD)],
                        "1/2 N(-0.9, 0.19) + 1/2 N(0.9, 0.19)"),
    "discrete3": _mixture("discrete3", [], "uniform on {-1, 0, 1} * sqrt(1.5)",
                          atoms=[(1 / 3, -_DISCRETE_LEVEL), (1 / 3, 0.0),
                                 (1 / 3, _DISCRETE_LEVEL)]),
}

NOISE = {
    "gauss": lambda sigma: Marginal("gauss", lambda x: _phi_cdf(x / sigma),
                                    lambda x: _phi_sf(x / sigma),
                                    lambda rng, size: sigma * rng.standard_normal(size),
                                    1.0 / (sigma * math.sqrt(2 * math.pi)), "Gaussian noise"),
    "laplace": lambda sigma: _laplace(sigma / SQRT2),
}


def discretised_pmf(marginal: Marginal, spec: FixedPointSpec) -> np.ndarray:
    """
    Exact pmf of quantize.to_integer applied to one draw from the marginal.

    Code c collects the cell [(c - 1/2) 2^-f, (c + 1/2) 2^-f); the two boundary codes +-max_int
    collect the whole clipped tails. Cell edges sit at half-integers on the code lattice, so
    they never coincide with an atom of the families above (atoms at 0 and +-sqrt(1.5) * 2^f).

    :param marginal: Distribution of the real value
    :param spec: Fixed-point format
    :return: Array of length 2 * max_int + 1, indexed by code + max_int, summing to 1
    """
    M, scale = spec.max_int, 2.0 ** spec.f
    upper = [(c + 0.5) / scale for c in range(-M, M)]       # right edges of codes -M..M-1
    cdf = np.array([marginal.cdf(e) for e in upper])
    sf = np.array([marginal.sf(e) for e in upper])
    pmf = np.empty(2 * M + 1)
    pmf[0] = cdf[0]
    pmf[-1] = sf[-1]
    # Interior cell mass, from whichever of CDF / survival is the smaller pair: a difference
    # of two numbers near 1 would lose the tail cells to cancellation
    left_cdf, right_cdf = cdf[:-1], cdf[1:]
    left_sf, right_sf = sf[:-1], sf[1:]
    pmf[1:-1] = np.where(right_cdf <= 0.5, right_cdf - left_cdf, left_sf - right_sf)
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


def renyi_bits(pmf: np.ndarray, p: int) -> float:
    """
    :param pmf: Probability masses (any shape; zeros ignored)
    :param p: Order, p >= 2
    :return: H_p in bits, (1/(1-p)) log2 sum P^p
    """
    v = np.asarray(pmf, dtype=float).ravel()
    v = v[v > 0]
    return math.log2(float(np.sum(v ** p))) / (1 - p)


# ─────────────────────────────────────────────────────────────────────────────
# The source
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntropyResult:
    """Theorem 1's entropy inputs for one source, in bits, with their provenance."""
    H_X_bits: float          # H_p of the whole stored vector (exact, or a LOWER bound)
    max_R_bits: float        # max over r-subsets (exact, or an UPPER bound)
    method: str              # 'exact' or 'bounds' (rho > 0)
    max_R_exact: bool        # the maximum touched only exactly computed patterns
    h_feature_bits: float = float("nan")    # H_p of one stored feature code
    h_label_bits: float = float("nan")      # H_p of one stored label code
    H_block_bits: float = float("nan")      # H_p of one sample's (features, label) block

    def in_log_q(self, q: int, n: int) -> tuple:
        """
        :param q: Field size
        :param n: Vector length, used to clamp H_p(X) <= n as entropy_bounds does
        :return: (H_p(X), max_R H_p(X_R)) in log_q units, ready for SystemContext
        """
        b = math.log2(q)
        return min(self.H_X_bits / b, float(n)), self.max_R_bits / b


@dataclass(frozen=True)
class SourceSpec:
    """
    One synthetic regression source. Every field is public and fixed before any data exists.

    :param family: Feature family, a key of FAMILIES
    :param n_samples: n_s
    :param n_features: P
    :param rho: Pairwise feature correlation; > 0 only for family 'gauss'
    :param snr: Signal-to-noise ratio Var(x.w*) / Var(e)
    :param noise: 'gauss' or 'laplace'
    :param label_mean: Target E[y]; b* is set from it so the intercept is always exercised
    :param f_X: Fractional bits of the stored data (features and labels)
    :param clip_X: Clip bound A_X of the stored data
    :param grid_bits: a, w* and b* are multiples of 2^-a
    """
    family: str = "gauss"
    n_samples: int = 25
    n_features: int = 3
    rho: float = 0.0
    snr: float = 1.0
    noise: str = "gauss"
    label_mean: float = 0.25
    f_X: int = 7
    clip_X: float = 4.0
    grid_bits: int = 4

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; choose from {list(FAMILIES)}")
        if self.rho and self.family != "gauss":
            raise ValueError("correlation is defined for the Gaussian family only")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        if self.noise not in NOISE:
            raise ValueError(f"unknown noise {self.noise!r}")

    # ── derived design quantities ───────────────────────────────────────────

    @property
    def n(self) -> int:
        """:return: Stored vector length n_s (P + 1)"""
        return self.n_samples * (self.n_features + 1)

    @property
    def x_spec(self) -> FixedPointSpec:
        """:return: Storage format of features and labels (f_y = f_X is required)"""
        return FixedPointSpec(f=self.f_X, clip=self.clip_X)

    @property
    def marginal(self) -> Marginal:
        return FAMILIES[self.family]

    @cached_property
    def feature_pmf(self) -> np.ndarray:
        """:return: Exact pmf of one stored feature code, indexed by code + max_int"""
        return discretised_pmf(self.marginal, self.x_spec)

    @cached_property
    def feature_moments(self) -> tuple:
        """:return: (mean, variance) of the DEQUANTISED feature value, exact from the pmf"""
        M, scale = self.x_spec.max_int, 2.0 ** self.f_X
        v = np.arange(-M, M + 1) / scale
        mean = float(np.sum(self.feature_pmf * v))
        return mean, float(np.sum(self.feature_pmf * (v - mean) ** 2))

    @cached_property
    def feature_cov(self) -> np.ndarray:
        """:return: Population covariance of the dequantised features (design, not sample)"""
        _, var = self.feature_moments
        P = self.n_features
        return var * ((1 - self.rho) * np.eye(P) + self.rho * np.ones((P, P)))

    @cached_property
    def weights(self) -> tuple:
        """
        The true model, on the dyadic grid 2^-a.

        Direction d_k = (-1)^k / (1 + k/2): alternating signs, decreasing magnitudes, so that
        every size P has a fixed public pattern. Scaled so the signal variance is
        snr / (1 + snr), i.e. Var y = 1 before rounding to the grid, then rounded; the noise
        is set from the ROUNDED signal variance so the SNR is exact. b* is the grid point
        nearest label_mean - E[x] . w*.

        :return: (w* as a float array, b*, noise standard deviation sigma_e)
        """
        P, grid = self.n_features, 2.0 ** self.grid_bits
        d = np.array([(-1) ** k / (1 + 0.5 * k) for k in range(P)])
        target = self.snr / (1.0 + self.snr)
        w = d * math.sqrt(target / float(d @ self.feature_cov @ d))
        w = np.round(w * grid) / grid
        if not np.any(w):
            raise ValueError("w* rounds to zero on the 2^-a grid; raise grid_bits")
        signal = float(w @ self.feature_cov @ w)
        mean_x, _ = self.feature_moments
        b = round((self.label_mean - mean_x * float(w.sum())) * grid) / grid
        return w, b, math.sqrt(signal / self.snr)

    @property
    def noise_marginal(self) -> Marginal:
        return NOISE[self.noise](self.weights[2])

    def augmented_hessian(self) -> np.ndarray:
        """
        Population second-moment matrix E[[x;1][x;1]^T] of the dequantised features with the
        intercept's constant column. The per-sample least-squares Hessian, from the design
        alone; its extreme eigenvalues set eta and the predicted condition number.

        :return: (P+1) x (P+1) matrix
        """
        P = self.n_features
        mean_x, _ = self.feature_moments
        mu = np.full(P, mean_x)
        H = np.empty((P + 1, P + 1))
        H[:P, :P] = self.feature_cov + np.outer(mu, mu)
        H[:P, P] = H[P, :P] = mu
        H[P, P] = 1.0
        return H

    # ── sampling ────────────────────────────────────────────────────────────

    def sample(self, rng: np.random.Generator) -> dict:
        """
        Draws one dataset.

        :param rng: Generator for this dataset (the 'data' stream)
        :return: dict with X_real (n_s, P), y_real (n_s,), X_codes, y_codes (int64), and the
            stored vector 'codes' = [X_codes flattened row-major, y_codes]
        """
        n_s, P, spec = self.n_samples, self.n_features, self.x_spec
        if self.rho:
            z0 = rng.standard_normal((n_s, 1))
            X = math.sqrt(self.rho) * z0 + math.sqrt(1 - self.rho) * rng.standard_normal((n_s, P))
        else:
            X = self.marginal.sampler(rng, (n_s, P))
        Xc = to_integer(X, spec)
        w, b, _ = self.weights
        y = (Xc / 2.0 ** self.f_X) @ w + b + self.noise_marginal.sampler(rng, (n_s,))
        yc = to_integer(y, spec)
        return dict(X_real=X, y_real=y, X_codes=Xc, y_codes=yc,
                    codes=np.concatenate([Xc.ravel(), yc]))

    # ── entropy ─────────────────────────────────────────────────────────────

    def entropies(self, p: int = 2, r: int = 5) -> EntropyResult:
        """
        Theorem 1's entropy inputs in bits: exact for independent features, rigorous bounds
        in the directions the theorem needs for correlated ones.

        :param p: Renyi order
        :param r: Subset size
        :return: EntropyResult
        """
        if self.rho:
            return self._entropy_bounds(p, r)
        return exact_entropies(self, p, r)

    def _entropy_bounds(self, p: int, r: int) -> EntropyResult:
        """
        The joint-density route of quantize.entropy_bounds for correlated Gaussian features,
        evaluated in bits (entropy_bounds itself returns log_q units, clamped to n, and the
        field is not known yet when this is called).

        One block's density is p_X(x) * p_e(y - x.w - b), bounded by
        sup p_X * sup p_e = (2 pi)^(-P/2) |Sigma|^(-1/2) * sup p_e, with |Sigma| of the
        equicorrelation matrix (1-rho)^(P-1) (1+(P-1)rho). Then, exactly as entropy_bounds:
        H_p(X) >= n f_X - log2 sup(joint density), and max_R H_p(X_R) <= r log2(levels).
        tests/test_gd.py checks this equals entropy_bounds once a q is fixed.
        """
        lower = self.n * self.f_X - self.joint_log2_sup_density()
        upper = r * self.x_spec.log2_levels
        return EntropyResult(H_X_bits=lower, max_R_bits=upper, method="bounds",
                             max_R_exact=False)

    def joint_log2_sup_density(self) -> float:
        """:return: log2 of the bound on the joint density of all n stored coordinates"""
        P, rho = self.n_features, self.rho
        log2_det = ((P - 1) * math.log2(1 - rho) if P > 1 else 0.0) + math.log2(1 + (P - 1) * rho)
        return self.n_samples * (-0.5 * P * math.log2(2 * math.pi) - 0.5 * log2_det
                                 + math.log2(self.noise_marginal.sup_density))


# ─────────────────────────────────────────────────────────────────────────────
# Exact entropies for independent features
# ─────────────────────────────────────────────────────────────────────────────

def _strided_convolve(acc: np.ndarray, offset: int, t: np.ndarray, W: int, M: int):
    """
    Convolves a distribution on the integer grid with that of W * c, c in [-M, M] with
    weights t: the distribution of U + W c when U ~ acc.

    :return: (new array, new offset); index i holds the value U = i + offset
    """
    if W == 0:
        return acc * float(t.sum()), offset
    tt = t if W > 0 else t[::-1]
    Wa = abs(W)
    out = np.zeros(len(acc) + 2 * M * Wa)
    for ci, val in enumerate(tt):
        if val:
            out[ci * Wa: ci * Wa + len(acc)] += val * acc
    return out, offset - M * Wa


def _u_distribution(spec: SourceSpec, tilt: int):
    """
    Distribution of U = 2^a (x^ . w* + 2^f_X b*) when the feature codes are drawn from the
    feature pmf raised to the power tilt (tilt = 1 is the true distribution; tilt = p is the
    'tilted' measure whose total mass is the collision sum).

    :return: (array, offset) over the integer grid of U
    """
    a, M = spec.grid_bits, spec.x_spec.max_int
    w, b, _ = spec.weights
    W = [int(round(v * 2 ** a)) for v in w]
    t = spec.feature_pmf ** tilt
    acc, off = np.ones(1), int(round(b * 2 ** (spec.f_X + a)))
    for Wk in W:
        acc, off = _strided_convolve(acc, off, t, Wk, M)
    return acc, off


def _label_kernel(spec: SourceSpec, U_lo: int, U_hi: int):
    """
    Conditional pmf of the stored label given U, for every U in [U_lo, U_hi], in the shift
    form the convolution needs.

    The label code is clip(round(u + 2^f_X e)) with u = U / 2^a. Interior code c has mass
    F(c + 1/2 - u) - F(c - 1/2 - u) under the noise CDF in code units, which depends only on
    j = c 2^a - U, so one vector g[j] serves every U; the two clip atoms are separate vectors
    over U.

    :return: (g, g_offset, top, bot): g[j - g_offset] for the interior; top[U - U_lo] and
        bot[U - U_lo] the masses on codes +max_int and -max_int
    """
    a, M, f = spec.grid_bits, spec.x_spec.max_int, spec.f_X
    noise, scale, half = spec.noise_marginal, 2.0 ** f, 2 ** (a - 1)
    j_lo, j_hi = -(M - 1) * 2 ** a - U_hi, (M - 1) * 2 ** a - U_lo
    # F evaluated on the 2^-a grid of code units, from j_lo - half to j_hi + half
    ks = range(j_lo - half, j_hi + half + 1)
    F = np.array([noise.cdf(k / 2 ** a / scale) for k in ks])
    S = np.array([noise.sf(k / 2 ** a / scale) for k in ks])
    # mass of [k-half, k+half] for each j, taking the difference on the small side
    lo_c, hi_c = F[:-2 * half], F[2 * half:]
    lo_s, hi_s = S[:-2 * half], S[2 * half:]
    g = np.clip(np.where(hi_c <= 0.5, hi_c - lo_c, lo_s - hi_s), 0.0, None)
    U = np.arange(U_lo, U_hi + 1)
    edge = (2 * M - 1) * half                 # (M - 1/2) on the 2^-a grid
    top = np.array([noise.sf((edge - u) / 2 ** a / scale) for u in U])
    bot = np.array([noise.cdf((-edge - u) / 2 ** a / scale) for u in U])
    return g, j_lo, top, bot


def _collision_given_u(spec: SourceSpec, g, g_off, top, bot, U_lo, U_hi, p):
    """
    c_p(U) = sum over label codes of P(y^ | U)^p, for every U, via per-residue prefix sums:
    the interior codes of one U read g at j = c 2^a - U, an arithmetic progression of stride
    2^a, so each sum is a difference of two prefix-sum entries.
    """
    a, M = spec.grid_bits, spec.x_spec.max_int
    stride = 2 ** a
    gp = g ** p
    # prefix sums along each residue class: cs[i] = gp[i] + cs[i - stride], with a block of
    # zeros in front so cs[i - stride] is always defined
    padded = np.concatenate([np.zeros(stride), gp])
    rows = -(-len(padded) // stride)
    padded = np.concatenate([padded, np.zeros(rows * stride - len(padded))])
    cs = np.cumsum(padded.reshape(rows, stride), axis=0).ravel()
    U = np.arange(U_lo, U_hi + 1)
    first = -(M - 1) * stride - U - g_off + stride      # index of c = -(M-1) in padded
    last = (M - 1) * stride - U - g_off + stride        # index of c = M-1
    interior = cs[last] - cs[first - stride]
    return interior + top ** p + bot ** p


def exact_entropies(spec: SourceSpec, p: int = 2, r: int = 5) -> EntropyResult:
    """
    Exact Renyi entropies of the stored vector for independent features.

      H_p(block):  sum_x prod_k pi(x_k)^p * c_p(U(x)) = sum_U T_p(U) c_p(U), where T_p is the
                   distribution of U under the tilted product measure -- exact, with no
                   independence assumed between features and label.
      H_p(label):  from the label's exact marginal pmf, sum_U P_U(U) P(y^ | U).
      max_R:       knapsack over blocks with per-pattern values. A pattern of c features and
                   no label is exactly c * h_feature (independent); {label} and the full
                   block are exact; label plus some-but-not-all features uses the valid upper
                   bound min(H_p(x_J) + log2 |label alphabet|, H_p(y) + log2 |x_J support|),
                   from sum_b P(a,b)^p >= |B|^(1-p) P(a)^p.

    :param spec: Source with rho = 0
    :param p: Renyi order
    :param r: Subset size
    :return: EntropyResult in bits
    """
    if spec.rho:
        raise ValueError("exact entropies need independent features; use spec.entropies")
    P, n_s, M = spec.n_features, spec.n_samples, spec.x_spec.max_int

    PU, off = _u_distribution(spec, 1)
    TU, off_t = _u_distribution(spec, p)
    assert off == off_t and len(PU) == len(TU)
    U_lo, U_hi = off, off + len(PU) - 1
    g, g_off, top, bot = _label_kernel(spec, U_lo, U_hi)

    c_p = _collision_given_u(spec, g, g_off, top, bot, U_lo, U_hi, p)
    H_block = math.log2(float(np.sum(TU * c_p))) / (1 - p)

    # Label marginal: interior code c collects sum_U P_U(U) g[c 2^a - U]
    stride = 2 ** spec.grid_bits
    U = np.arange(U_lo, U_hi + 1)
    label = np.empty(2 * M + 1)
    for c in range(-(M - 1), M):
        label[c + M] = float(np.dot(PU, g[c * stride - U - g_off]))
    label[0], label[-1] = float(np.dot(PU, bot)), float(np.dot(PU, top))
    h_y = renyi_bits(label / label.sum(), p)
    h_x = renyi_bits(spec.feature_pmf, p)

    log2_label_alphabet = math.log2(spec.x_spec.levels)
    log2_x_support = math.log2(int(np.count_nonzero(spec.feature_pmf)))

    def best_per_count(allow_bounds: bool):
        v = [0.0] + [-math.inf] * (P + 1)
        for c in range(1, P + 2):
            cands = [c * h_x] if c <= P else []
            j = c - 1                          # features alongside the label
            if j == 0:
                cands.append(h_y)
            elif j == P:
                cands.append(H_block)
            elif allow_bounds:
                cands.append(min(j * h_x + log2_label_alphabet, h_y + j * log2_x_support))
            v[c] = max(cands) if cands else -math.inf
        return v

    def knapsack(v):
        best = [0.0] + [-math.inf] * r
        for _ in range(n_s):
            best = [max((best[s - c] + v[c] for c in range(0, min(s, P + 1) + 1)
                         if best[s - c] > -math.inf), default=-math.inf)
                    for s in range(r + 1)]
        return best[r]

    max_R = knapsack(best_per_count(True))
    exact_only = knapsack(best_per_count(False))
    return EntropyResult(H_X_bits=n_s * H_block, max_R_bits=max_R, method="exact",
                         max_R_exact=abs(max_R - exact_only) < 1e-9,
                         h_feature_bits=h_x, h_label_bits=h_y, H_block_bits=H_block)


def exact_entropies_bruteforce(spec: SourceSpec, p: int = 2, r: int = 5) -> dict:
    """
    Independent check of exact_entropies for TINY formats: enumerate every feature-code
    tuple, build the joint block pmf directly from the noise CDF (no convolution, no prefix
    sums), marginalise every pattern exactly, and maximise over all r-subsets of the vector
    by enumerating how many coordinates each block contributes.

    :return: dict with H_block, h_y, and max_R (all exact, bits)
    """
    import itertools
    P, M, f, a = spec.n_features, spec.x_spec.max_int, spec.f_X, spec.grid_bits
    w, b, _ = spec.weights
    noise, scale = spec.noise_marginal, 2.0 ** f
    codes = range(-M, M + 1)
    joint = {}
    for xs in itertools.product(codes, repeat=P):
        px = math.prod(spec.feature_pmf[c + M] for c in xs)
        u = sum(c * wk for c, wk in zip(xs, w)) + b * scale
        for c in codes:
            if c == M:
                py = noise.sf((M - 0.5 - u) / scale)
            elif c == -M:
                py = noise.cdf((-M + 0.5 - u) / scale)
            else:
                py = noise.cdf((c + 0.5 - u) / scale) - noise.cdf((c - 0.5 - u) / scale)
            joint[xs + (c,)] = px * py
    keys = list(joint)
    probs = np.array([joint[k] for k in keys])

    def pattern_entropy(pattern):
        if not pattern:
            return 0.0
        agg = {}
        for k, pr in zip(keys, probs):
            sub = tuple(k[i] for i in pattern)
            agg[sub] = agg.get(sub, 0.0) + pr
        return renyi_bits(np.array(list(agg.values())), p)

    per_count = {}
    for c in range(P + 2):
        per_count[c] = max(pattern_entropy(s) for s in itertools.combinations(range(P + 1), c))
    best = -math.inf
    for counts in itertools.product(range(P + 2), repeat=spec.n_samples):
        if sum(counts) == r:
            best = max(best, sum(per_count[c] for c in counts))
    return dict(H_block=renyi_bits(probs, p), h_y=pattern_entropy((P,)), max_R=best)
