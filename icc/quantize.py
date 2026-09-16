"""
File: quantize.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Fixed-point quantisation between the reals and F_q for the ICC scheme.

    The scheme computes a polynomial f of total degree <= d over data stored in F_q, so
    real-valued ML features have to be embedded in F_q in a way that (i) commutes with
    polynomial evaluation and (ii) can be lifted back to the reals after decoding. Only
    one family of maps does both, and it has two stages:

        reals  --clip, scale by 2^f, round-->  a window of Z  --mod q-->  F_q

    Stage one (to_integer) is lossy and is where the precision/range tradeoff lives. Stage
    two (codes_to_field) is the canonical ring homomorphism Z -> Z/qZ, which is what makes
    polynomial evaluation over the field reproduce the integer computation; it is inverted
    by reading field elements as BALANCED representatives in [-(q-1)/2, (q-1)/2].

    *** q MUST BE PRIME. *** Stage two needs F_q ~= Z/qZ as a RING, which fails for every
    prime power q = p^k with k > 1: in F_{p^k} the image of Z is only the prime subfield,
    so integers do not embed at all. See require_prime_field, which enforces it, for the
    concrete failure. The ICC scheme is stated over a general F_q; this is a constraint the
    quantisation layer adds.

    See docs/QUANTIZATION.md for why the alternative embeddings (logarithmic,
    modular-rational, companding) do not work, and for the derivations behind the two bound
    functions here.

    Two things have to be sized, and they pull in opposite directions:

      * Decodability (correctness). Reduction mod q is a ring homomorphism from Z, so the
        decoded field element equals the true integer result mod q. Lifting it back to the
        integers is exact iff that integer never leaves the balanced window, i.e.
        q >= 2*B + 1 where B bounds the magnitude of the integer computation. This pushes
        q up. B must be a HARD bound, not a typical case: the client keeps only the key k,
        never the data, so it cannot detect wraparound after the fact. Clipping is what
        makes a hard bound available -- see gradient_bound_worst_case.

      * Privacy (Theorem 1). The per-symbol entropy in log_q units is
        h = H_p(symbol in bits) / log2(q), and m >= n(1 - h) + ... , so a larger q with the
        same data lowers h and inflates the key size and the worker count. This pushes q
        down.

    So q should be the smallest prime satisfying the decodability bound, and nothing larger.

    Implementation ceiling worth knowing: galois evaluates F_q natively (uint32,
    ufunc_mode='jit-calculate') only while q^2 fits in int64, i.e. q <= 3037000499
    (log2 q <= 31.5). One prime above that and it silently switches to python-calculate
    with dtype=object: measured 28x slower to build the interpolation matrix and 7x slower
    to solve at lambda=1225. This is a tooling limit, not a limit of the scheme.
"""

import math
from dataclasses import dataclass
from typing import Optional

import galois
import numpy as np

# Largest q for which galois keeps native uint32 arithmetic (q^2 < 2^63). Measured, not
# documented upstream: q = 3037000493 gives ufunc_mode 'jit-calculate', the next prime
# 3037000507 gives 'python-calculate' with dtype=object.
NATIVE_Q_MAX = int(math.isqrt(2 ** 63))  # 3037000499

# sup of the standard normal density, used as the default density bound for standardised
# features. See entropy_bounds().
SUP_DENSITY_STANDARD_NORMAL = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class FixedPointSpec:
    """
    One operand family's fixed-point format: f fractional bits and a symmetric clip bound.

    Clipping is not a convenience here, it is what makes the no-wraparound bound sound.
    Without it there is no a priori bound on |x| and therefore no way to choose q such that
    the integer gradient is guaranteed representable.
    """
    f: int          # fractional bits; step size is 2^-f
    clip: float     # values are clipped to [-clip, +clip] before scaling

    @property
    def step(self) -> float:
        """:return: Quantisation step in data units"""
        return 2.0 ** -self.f

    @property
    def max_int(self) -> int:
        """:return: Largest magnitude any quantised integer can take, by construction"""
        return int(math.floor(self.clip * 2 ** self.f))

    @property
    def levels(self) -> int:
        """:return: Number of distinct integer codes the map can produce"""
        return 2 * self.max_int + 1

    @property
    def log2_levels(self) -> float:
        """:return: log2 of the code alphabet size; a rigorous upper bound on H_p per symbol"""
        return math.log2(self.levels)


def to_integer(values, spec: FixedPointSpec, rounding: str = "nearest",
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Maps reals to the fixed-point integer lattice: clip, scale by 2^f, round.

    Rounding mode 'nearest' is the default and is what the stored data should use. Stored
    data is quantised exactly once per training run, so rounding error never accumulates
    and the unbiasedness that motivates stochastic rounding buys nothing; measured, it also
    buys no entropy (the source is already smooth on the lattice scale) while inflating the
    RMS error by sqrt(2). 'stochastic' is provided for the per-step weight quantisation,
    where low precision f_w can stall gradient descent -- see docs/PLAN.md Week 7.

    :param values: Array-like of reals
    :param spec: Fixed-point format to use
    :param rounding: 'nearest' or 'stochastic'
    :param rng: Generator required when rounding='stochastic'
    :return: int64 array of integer codes, each with magnitude <= spec.max_int
    """
    v = np.clip(np.asarray(values, dtype=np.float64), -spec.clip, spec.clip) * 2 ** spec.f

    if rounding == "nearest":
        out = np.rint(v)
    elif rounding == "stochastic":
        if rng is None:
            raise ValueError("rounding='stochastic' requires an rng")
        floor = np.floor(v)
        out = floor + (rng.random(v.shape) < (v - floor))
    else:
        raise ValueError(f"unknown rounding mode {rounding!r}")

    # Clipping happens before scaling, so this only guards against a rint/floor edge case
    return np.clip(out, -spec.max_int, spec.max_int).astype(np.int64)


def from_integer(codes, spec: FixedPointSpec) -> np.ndarray:
    """
    Inverse of to_integer up to the quantisation error: descale by 2^f.

    :param codes: Integer codes
    :param spec: Fixed-point format they were produced with
    :return: float64 array of reals
    """
    return np.asarray(codes, dtype=np.float64) / 2 ** spec.f


def to_field(GF: type[galois.FieldArray], values, spec: FixedPointSpec,
             rounding: str = "nearest",
             rng: Optional[np.random.Generator] = None) -> galois.FieldArray:
    """
    Quantises reals and embeds them in F_q as x -> round(x * 2^f) mod q.

    The embedding is the restriction of the ring homomorphism Z -> Z/qZ to the balanced
    window, which is exactly why polynomial evaluation over F_q reproduces the integer
    computation: any polynomial with integer coefficients commutes with reduction mod q.

    :param GF: Galois field object
    :param values: Array-like of reals
    :param spec: Fixed-point format
    :param rounding: 'nearest' or 'stochastic'
    :param rng: Generator required when rounding='stochastic'
    :return: GF array of the same shape as values
    """
    codes = to_integer(values, spec, rounding=rounding, rng=rng)
    return codes_to_field(GF, codes, spec)


def require_prime_field(GF: type[galois.FieldArray]):
    """
    Rejects extension fields, on which the fixed-point embedding is not merely inaccurate
    but meaningless.

    The embedding relies on F_q being isomorphic to Z/qZ AS A RING, which holds only when q
    is prime. In an extension field F_{p^k} with k > 1 the image of Z is just the prime
    subfield: 1+1+...+1 takes only p distinct values, so in GF(2^5) the integers collapse to
    {0, 1} and 3 + 5 = 6 rather than 8. Multiplying by 2^f cannot separate anything either,
    because the element labelled 2 is not 1+1 there.

    This is a real footgun rather than a theoretical one: galois labels extension-field
    elements with integers 0..q-1 too, and some operations coincidentally look correct
    (GF(2)*GF(7) prints 14), so an extension field runs to completion and returns confident
    nonsense. The ICC scheme itself is stated over a general F_q and its Reed-Muller
    machinery works there; it is the QUANTISATION layer that requires a prime field. Worth
    stating in the paper as a constraint the embedding imposes on the scheme.

    :param GF: Galois field object
    :raises ValueError: if GF is an extension field
    """
    if GF.degree != 1:
        raise ValueError(
            f"fixed-point embedding requires a PRIME field, got GF({GF.characteristic}^"
            f"{GF.degree}) of order {GF.order}. In characteristic {GF.characteristic} the "
            f"image of Z has only {GF.characteristic} elements, so integers do not embed "
            f"and polynomial evaluation does not reproduce integer arithmetic. Use "
            f"galois.GF(p) for a prime p (choose_prime returns one), or the residue-number-"
            f"system variant over several prime fields."
        )


def codes_to_field(GF: type[galois.FieldArray], codes,
                   spec: Optional[FixedPointSpec] = None) -> galois.FieldArray:
    """
    Embeds already-quantised integer codes in F_q, keeping the sign convention in one place.

    This is the second half of the two-stage map from the reals. Stage one (to_integer)
    lands in a window of Z; stage two is the canonical ring homomorphism Z -> Z/qZ ~= F_q,
    i.e. `codes % q`. Only the second stage needs the field: it is a ring homomorphism, which
    is what makes polynomial evaluation over F_q reproduce the integer computation. See
    require_prime_field for why q must be prime for that isomorphism to exist.

    Separate from to_field because the client quantises once and may then need to assemble,
    reshape or concatenate the codes before uploading; doing `% q` at those call sites is
    how a sign convention gets silently duplicated and then diverges from lift_balanced.

    :param GF: Galois field object; must be a prime field
    :param codes: Integer codes, any shape
    :param spec: Optional format, checked to fit in the balanced window of F_q
    :return: GF array of the same shape
    """
    require_prime_field(GF)
    if spec is not None and spec.levels > GF.order:
        raise OverflowError(
            f"the code alphabet ({spec.levels} levels) does not fit in the balanced window "
            f"of F_q with q = {GF.order}; the stored data itself would wrap"
        )
    return GF(np.asarray(codes, dtype=np.int64) % GF.order)


def lift_balanced(z, q: int) -> np.ndarray:
    """
    Reads field elements as signed integers in [-(q-1)/2, (q-1)/2].

    This is the only place the sign convention lives. It is correct iff the true integer
    result lies in that window; assert_representable is how a caller proves it does.

    :param z: Field elements, or any integer array in [0, q)
    :param q: Field size
    :return: int64 array of balanced representatives (object dtype for q > 2^62)
    """
    v = np.asarray(z, dtype=object if q > 2 ** 62 else np.int64)
    half = (q - 1) // 2
    return np.where(v > half, v - q, v)


def from_field(z, q: int, total_f: int) -> np.ndarray:
    """
    Lifts field elements to signed integers and descales by 2^total_f.

    total_f is the sum of the fractional bits of every quantised factor in the computation,
    since degree-k products multiply the scales. For the least-squares gradient with the
    query polynomial of docs/QUANTIZATION.md it is 2*f_X + f_w.

    :param z: Field elements returned by Client.decode_result
    :param q: Field size
    :param total_f: Combined fractional bits of the result
    :return: float64 array of reals
    """
    return np.asarray(lift_balanced(z, q), dtype=np.float64) / 2.0 ** total_f


def assert_representable(integers, q: int, what: str = "result"):
    """
    Fails loudly if an integer would wrap when reduced mod q.

    Only usable where the true integers are actually in hand -- a test, or the client at
    quantisation time. It is deliberately NOT available at decode time: the client stores
    only k, so after the fact a wrapped result is indistinguishable from a correct one.
    That asymmetry is why the field size must come from a hard bound.

    :param integers: True integer values of the computation
    :param q: Field size
    :param what: Label used in the error message
    :raises OverflowError: if any value falls outside the balanced window
    """
    half = (q - 1) // 2
    v = np.asarray(integers, dtype=object)
    worst = max(abs(int(x)) for x in np.atleast_1d(v).ravel())
    if worst > half:
        raise OverflowError(
            f"{what} reaches |{worst}| > (q-1)/2 = {half} with q = {q}: the balanced lift "
            f"would be wrong. Increase q (log2 q >= {math.log2(2 * worst + 1):.1f}) or "
            f"reduce the fractional bits / clip bounds."
        )


def choose_prime(magnitude_bound: float, warn_native: bool = True) -> int:
    """
    Smallest prime q with q >= 2*B + 1, i.e. the cheapest field that cannot wrap.

    Smallest is deliberate: every spare bit of q divides into the entropy ratio
    h = H_p(symbol) / log2 q and comes back as extra key length and extra workers.

    :param magnitude_bound: Hard bound B on the magnitude of the integer result
    :param warn_native: Print a notice when the prime crosses the galois native ceiling
    :return: The prime field size to use
    """
    q = int(galois.next_prime(int(math.ceil(2 * magnitude_bound)) + 1))
    if warn_native and q > NATIVE_Q_MAX:
        print(f"  note: q = {q} (log2 q = {math.log2(q):.1f}) exceeds NATIVE_Q_MAX = "
              f"{NATIVE_Q_MAX}; galois will fall back to dtype=object arithmetic "
              f"(~28x slower interpolation-matrix build, ~7x slower solve).")
    return q


@dataclass(frozen=True)
class FieldChoice:
    """
    A field size together with the accounting for how it was chosen.

    leakage_bits is the part that matters and the reason this is a dataclass rather than a
    bare int: q is PUBLIC (the provider needs it to do arithmetic), so if q was derived from
    the data then publishing it is a side channel that Theorem 1 does not cover. Carrying the
    bound alongside q forces the caller to compare it against the scheme's own leakage
    budget instead of forgetting it exists.
    """
    q: int
    magnitude_bound: int
    policy: str             # how the bound was obtained and coarsened
    leakage_bits: float     # upper bound on bits about the data revealed by publishing q
    log2_q: float
    native: bool            # stays under the galois native-arithmetic ceiling

    def __str__(self) -> str:
        return (f"q = {self.q} (log2 q = {self.log2_q:.2f}, "
                f"{'native' if self.native else 'OBJECT dtype'}), policy '{self.policy}', "
                f"leaks <= {self.leakage_bits:.2f} bits")


# Public grid for the coarsened policy: powers of two from 2^8 to 2^48. Must be fixed in
# advance and independent of the data, otherwise the grid itself leaks.
POWER_OF_TWO_GRID = tuple(2 ** k for k in range(8, 49))


def leakage_budget_bits(eps_c: float, q: int) -> float:
    """
    Converts the Theorem 1 leakage bound eps_c into bits, so it can be compared against
    other channels on the same scale.

    eps_c is stated in log_q units (Theorem 1 defines it with log_q), so the bit value is
    eps_c * log2(q). This exists to make the comparison in choose_field_size mechanical: the
    whole scheme's per-subset budget is a few hundredths of a bit, which is the number any
    side channel has to be measured against.

    :param eps_c: Leakage bound from utils.compute_leakage_bound, in log_q units
    :param q: Field size
    :return: The same bound expressed in bits
    """
    return eps_c * math.log2(q)


def choose_field_size(magnitude_bound: int, data_dependent: bool = False,
                      grid=None, warn_native: bool = True) -> FieldChoice:
    """
    The field-size rule: smallest prime that cannot wrap, with the leakage of the choice
    itself made explicit.

    Three policies, and the recommendation is the first:

    'public-format' (data_dependent=False). The bound came from public parameters only
        (clip, f_X, f_w, n_s, P), as gradient_bound_worst_case does, so q is a deterministic
        function of public data and leaks NOTHING. leakage_bits = 0. This is the default and
        should stay the default; see the arithmetic below for why.

    'coarsened' (data_dependent=True with a grid). The bound came from the data, but is
        rounded up to a PUBLIC, FIXED grid before choosing q, so the only thing published is
        which bucket the data fell into. Then H(q) <= log2|grid| and, by the chain rule,
        I(X_R; X~, q) <= I(X_R; X~) + H(q), i.e. the side channel adds at most log2|grid|
        bits on top of eps_c. The grid must be fixed in advance: a grid derived from the data
        leaks through the grid instead.

    'data-exact' (data_dependent=True, no grid). q = next_prime(2B+1) for a data-derived B,
        which pins B to within a prime gap and so reveals essentially log2(B) bits about an
        aggregate of the whole data vector. Unquantified; refused rather than warned about.

    WHY THE DEFAULT IS THE PUBLIC-FORMAT POLICY, in one comparison. At the case-study
    parameters the scheme's own budget is eps_c = 0.0016 in log_q units, i.e.
    leakage_budget_bits = 0.0016 * 29.3 = 0.047 BITS per r-subset. The coarsened policy with
    the power-of-two grid leaks up to log2(41) = 5.4 bits. So paying for the tighter
    data-dependent bound would add roughly 100x the entire leakage the scheme is designed to
    guarantee, in exchange for saving 4.2 bits of q (about 5 key symbols and 400 workers).
    That is not a close call, and it is why gradient_bound_data_dependent exists mainly to
    quantify what is being given up rather than to be used.

    Two caveats worth keeping in view. H(q) is a worst-case bound on I(X_R; q) and a
    sensitivity-based analysis might well be far tighter, since B is an aggregate over all n
    coordinates and any single r-subset moves it weakly -- that is open question 7. And a
    grid coarse enough to be competitive (2 or 3 buckets) is close enough to the public-format
    policy that the difference stops mattering.

    :param magnitude_bound: Hard bound B on the magnitude of the integer result
    :param data_dependent: True if B was computed from the data rather than the public format
    :param grid: Public, fixed, ascending bucket edges; required when data_dependent is True
    :param warn_native: Print a notice when q crosses the galois native ceiling
    :return: FieldChoice with q and the leakage accounting
    :raises ValueError: for the unquantified 'data-exact' policy
    """
    if not data_dependent:
        policy, bound, leak = "public-format", int(math.ceil(magnitude_bound)), 0.0
    elif grid is None:
        raise ValueError(
            "a data-derived magnitude bound cannot be published as-is: q would pin the "
            "bound to within a prime gap and reveal ~log2(B) bits about an aggregate of the "
            "whole data vector, which Theorem 1 does not account for. Pass a public, fixed "
            "grid (e.g. POWER_OF_TWO_GRID) to bound the leakage by log2|grid| bits, or use "
            "the public-format bound and leak nothing."
        )
    else:
        edges = sorted(int(g) for g in grid)
        bucket = next((g for g in edges if g >= magnitude_bound), None)
        if bucket is None:
            raise ValueError(
                f"magnitude bound {magnitude_bound} exceeds the largest grid edge "
                f"{edges[-1]}; extend the grid (publicly, in advance)"
            )
        policy, bound, leak = "coarsened", bucket, math.log2(len(edges))

    q = choose_prime(bound, warn_native=warn_native)
    return FieldChoice(q=q, magnitude_bound=bound, policy=policy, leakage_bits=leak,
                       log2_q=math.log2(q), native=q <= NATIVE_Q_MAX)


def gradient_bound_worst_case(n_samples: int, n_features: int,
                              x_spec: FixedPointSpec, w_spec: FixedPointSpec,
                              y_spec: Optional[FixedPointSpec] = None) -> int:
    """
    Hard bound on the integer least-squares gradient, from the clip bounds alone.

    For the query polynomial of docs/QUANTIZATION.md,

        g_j = sum_i ( sum_k xq[i,k] * wq[k]  -  2^{f_w} * yq[i] ) * xq[i,j]

    every factor is bounded by its spec, so

        |g_j| <= n_s * ( P * Xmax^2 * Wmax  +  2^{f_w} * Xmax * Ymax )

    with Xmax = x_spec.max_int and so on. Requires f_y = f_X, which is what makes both
    terms land at the common scale 2^{2 f_X + f_w}; the 2^{f_w} on the label term is a
    public integer folded into the polynomial's coefficients, so the STORED quantisation of
    y stays independent of the per-step weight precision. That matters because the data is
    uploaded once and must serve every gradient-descent step.

    This bound needs nothing but the public format, so it leaks nothing. It is also loose:
    it assumes every sample simultaneously attains the clip bound. Measured 691x loose
    (9.4 bits of q) on a seeded synthetic case and 4.2 bits on the diabetes subsample --
    see gradient_bound_data_dependent for the tighter, leakier alternative.

    :param n_samples: Number of samples n_s
    :param n_features: Number of features P
    :param x_spec: Format of the feature matrix
    :param w_spec: Format of the weight vector
    :param y_spec: Format of the labels; defaults to x_spec, which the derivation assumes
    :return: Hard bound B on |g_j|
    """
    y_spec = y_spec or x_spec
    if y_spec.f != x_spec.f:
        raise ValueError(
            f"the query polynomial requires f_y == f_X so both gradient terms share the "
            f"scale 2^(2 f_X + f_w); got f_y = {y_spec.f}, f_X = {x_spec.f}"
        )
    xmax, wmax, ymax = x_spec.max_int, w_spec.max_int, y_spec.max_int
    return n_samples * (n_features * xmax * xmax * wmax + 2 ** w_spec.f * xmax * ymax)


def gradient_bound_data_dependent(x_codes, y_codes, w_spec: FixedPointSpec) -> int:
    """
    Hard bound on the integer gradient using the actual quantised data, valid for every
    admissible weight vector.

    Replaces the worst-case sums over samples with the true ones, keeping the worst case
    only over w (which is not known until the gradient step). The client can evaluate this
    at upload time, while it still holds the data, and remember the single resulting
    scalar -- O(1) extra client state alongside k.

    PRIVACY CAVEAT, and it is a real one: q then depends on the data, and q is public. That
    is a leakage channel Theorem 1 does not cover, since the theorem conditions on the
    scheme parameters. Rounding q up to a coarse public grid (say the next power of two)
    caps the leaked quantity at a few bits, but it does not make it zero. Open question 7 of
    docs/QUANTIZATION.md. Use gradient_bound_worst_case unless that is resolved.

    :param x_codes: Quantised feature matrix, shape (n_s, P), integer codes
    :param y_codes: Quantised labels, shape (n_s,), integer codes
    :param w_spec: Format of the weight vector, giving the bound on |wq|
    :return: Hard bound B on |g_j|, maximised over j
    """
    X = np.abs(np.asarray(x_codes, dtype=object))
    y = np.abs(np.asarray(y_codes, dtype=object))
    wmax = w_spec.max_int

    per_sample = X.sum(axis=1) * wmax + 2 ** w_spec.f * y   # bounds |residual_i|
    return int(max((per_sample * X[:, j]).sum() for j in range(X.shape[1])))


def symbol_entropy_lower_bits(spec: FixedPointSpec,
                              sup_density: float = SUP_DENSITY_STANDARD_NORMAL,
                              saturation_mass: Optional[float] = None) -> float:
    """
    Lower bound, in bits, on H_p of one quantised symbol, valid for every order p > 1.

    Renyi entropy of any order p > 1 is bounded below by the min-entropy, so bounding the
    largest atom of the quantised distribution bounds every H_p at once:

        sum_i P_i^p <= (max_i P_i)^(p-1)   =>   H_p >= -log2(max_i P_i).

    Two kinds of atom compete for that maximum:

      * an interior cell, of width 2^-f, has mass at most 2^-f * M where M bounds the
        source density -- contributing f - log2(M) bits;
      * each SATURATION atom collects a whole clipped tail, so its mass does not shrink
        with f at all -- contributing -log2(saturation_mass) bits.

    The bound is the smaller of the two, and the second one is easy to forget. It was found
    by a falsification rather than by derivation: for a 4-sigma-clipped Gaussian at f = 16,
    study A of tools/quantization_study.py measures H_8 = 16.93 bits, below the
    interior-cell value f + 1.325 = 17.33, because the tail atom (mass 3.2e-5) has by then
    overtaken a typical cell (mass 6.1e-6). The fix is not to weaken the bound but to widen
    the clip: choose the clip bound so the tail mass is at most 2^-f * M, and the interior
    term binds again. Measured at n_s=25, P=3, f_X=8, f_w=0: widening the clip from 4 to 5
    sigma costs 0.94 bits of q via the magnitude bound and buys 6.79 bits of entropy
    headroom, so it is very cheap -- the clip enters the magnitude bound polynomially but
    the tail mass it controls decays like a Gaussian.

    :param spec: Fixed-point format
    :param sup_density: Bound M on the source's marginal density
    :param saturation_mass: Bound on the probability mass landing on one clip boundary;
        None asserts that clipping is not in play (an unbounded-support source with a
        wide enough clip, or genuinely bounded data)
    :return: Lower bound on H_p(symbol) in bits, for every p > 1
    """
    interior = spec.f - math.log2(sup_density)
    if saturation_mass is None:
        return interior
    if not 0.0 < saturation_mass < 1.0:
        raise ValueError(f"saturation_mass must lie in (0, 1), got {saturation_mass}")
    return min(interior, -math.log2(saturation_mass))


def entropy_bounds(n: int, r: int, q: int, spec: FixedPointSpec,
                   sup_density: float = SUP_DENSITY_STANDARD_NORMAL,
                   saturation_mass: Optional[float] = None,
                   joint_log2_sup_density: Optional[float] = None) -> tuple[float, float]:
    """
    Theorem 1's two entropy inputs as RIGOROUS BOUNDS in the directions the theorem needs,
    for data that came from quantising a continuous source.

    Theorem 1 subtracts H_p(X) and adds max_R H_p(X_R), so it needs a LOWER bound on the
    first and an UPPER bound on the second; getting either backwards under-estimates m and
    breaks the privacy guarantee. Both bounds below hold for every order p > 1.

    Lower bound on H_p(X): n times symbol_entropy_lower_bits under a product source, or
    n*f - log2(M_n) when joint_log2_sup_density gives log2(M_n) for the JOINT density of
    all n coordinates. The joint form is the interesting one -- it needs no independence
    assumption, and for a Gaussian source -log2(M_n) = h_diff(X) - 0.721*n, so it pays for
    correlation automatically through log2|Sigma| instead of ignoring it.

    Upper bound on max_R H_p(X_R): Renyi entropy is maximised by the uniform distribution,
    so H_p(X_R) <= log2|alphabet| = r * spec.log2_levels bits. This one holds
    unconditionally -- no source model at all, correlated or otherwise.

    WHY THIS EXISTS, and what it is FOR. utils.compute_p_entropy and
    utils.compute_max_subset_p_entropy are closed forms under an i.i.d. source model, which
    is exact for the synthetic control and an open question for correlated real features
    (CLAUDE.md open issue 2: Renyi entropy of order p >= 2 is not subadditive, so summing
    per-coordinate entropies bounds H_p(X) in neither direction). This function is a
    PROPOSED replacement for real data, and it is not a third estimator to be mixed with
    those two -- it computes bounds, not estimates. It trades an unverifiable independence
    assumption for a single scalar density bound. It still needs Raviv's sign-off before
    any privacy claim rests on it; see open question 1 of docs/PLAN.md and section 6 of
    docs/QUANTIZATION.md.

    :param n: Data length
    :param r: Privacy parameter
    :param q: Field size
    :param spec: Fixed-point format the data was quantised with
    :param sup_density: Bound M on one coordinate's marginal density
    :param saturation_mass: Bound on the mass on one clip boundary; see
        symbol_entropy_lower_bits
    :param joint_log2_sup_density: log2 of a bound on the JOINT density of all n
        coordinates. When given it replaces the product-source calculation, and
        sup_density / saturation_mass are ignored.
    :return: (lower bound on H_p(X), upper bound on max_R H_p(X_R)), both in log_q units
    """
    b = math.log2(q)
    if joint_log2_sup_density is not None:
        h_lower_bits = n * spec.f - joint_log2_sup_density
    else:
        h_lower_bits = n * symbol_entropy_lower_bits(spec, sup_density, saturation_mass)
    max_r_upper_bits = r * spec.log2_levels

    # Entropy in log_q units cannot exceed the vector length; clamp so a generous density
    # bound cannot hand compute_required_m an impossible H_p(X) > n.
    return min(h_lower_bits / b, float(n)), max_r_upper_bits / b
