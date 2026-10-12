"""
File: query.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Explicit query polynomials, and the public overflow check that gates them.

    A query used to be an opaque Python callable. That was enough to evaluate it, but not to
    reason about it: nobody could tell, before running it, whether the result would fit in
    the field. A Polynomial instead carries its terms explicitly, as signed integer
    coefficients on monomials, so its worst case over the declared input range is a
    closed-form function of public quantities:

        B(f) = sum over terms a of |c_a| * max_input^deg(a)

    If B(f) <= (q-1)/2 then for EVERY data vector whose integer codes satisfy
    |x_i| <= max_input, the integer value f(x) lies in the balanced window
    [-(q-1)/2, (q-1)/2], so the field result lifts back to f(x) exactly
    (docs/QUANTIZATION.md section 7). If B(f) > (q-1)/2 the result MAY have wrapped, and
    since the client keeps only the key it could never tell. So the server refuses such a
    query up front and returns a status saying why, instead of a number that looks valid.

    The check reads only the coefficients, the degree, q and max_input -- all public -- so
    refusing a query reveals nothing about the data.

    Coefficients are signed Python integers, never field elements. Read as a field element,
    a coefficient of -1 is q-1, and the bound would be wildly pessimistic.

    The degree check rides along because it fails the same way: the shares only support
    polynomials of total degree <= d (2024 paper section IV; the interpolation is over
    RM_q(d, m)), and a higher-degree query decodes to a wrong value with no error.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

import galois
import numpy as np

# A monomial is the sorted tuple of its variable indices, with repetition:
# x0^2 * x3 is (0, 0, 3), and the constant term is ().
Monomial = Tuple[int, ...]


class Polynomial:
    """
    A polynomial over the data vector with signed integer coefficients.

    Build one from variables and integer arithmetic, which reads like the formula:

        x = variables(n)
        f = x[0] ** 2 + x[1] * x[2] - 3 * x[4]

    Calling it on a galois array evaluates it in that field, so the same object is both the
    query the server checks and the plaintext reference a test compares against.
    """

    def __init__(self, terms: Optional[Dict[Monomial, int]] = None):
        """
        :param terms: Map from monomial to integer coefficient; zero coefficients are dropped
        """
        merged: Dict[Monomial, int] = {}
        for mono, coeff in (terms or {}).items():
            if not isinstance(coeff, (int, np.integer)):
                raise TypeError(
                    f"coefficient {coeff!r} of {mono} must be an integer; field elements "
                    f"and floats are not accepted because the overflow bound needs the "
                    f"signed integer value"
                )
            key = tuple(sorted(int(v) for v in mono))
            if any(v < 0 for v in key):
                raise ValueError(f"monomial {mono} has a negative variable index")
            merged[key] = merged.get(key, 0) + int(coeff)
        self.terms: Dict[Monomial, int] = {m: c for m, c in merged.items() if c != 0}
        self._compiled = {}  # field order -> per-degree (index matrix, coefficient array)

    # ── construction ────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(other) -> "Polynomial":
        """
        Lifts an integer constant to a Polynomial so arithmetic can mix the two.

        :param other: Polynomial or integer
        :return: Polynomial
        """
        if isinstance(other, Polynomial):
            return other
        if isinstance(other, (int, np.integer)):
            return Polynomial({(): int(other)})
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        out = dict(self.terms)
        for mono, coeff in other.terms.items():
            out[mono] = out.get(mono, 0) + coeff
        return Polynomial(out)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return other
        out: Dict[Monomial, int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                mono = tuple(sorted(m1 + m2))
                out[mono] = out.get(mono, 0) + c1 * c2
        return Polynomial(out)

    __rmul__ = __mul__

    def __pow__(self, exponent: int):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("only non-negative integer powers are supported")
        result = Polynomial({(): 1})
        for _ in range(exponent):
            result = result * self
        return result

    def __eq__(self, other):
        return isinstance(other, Polynomial) and self.terms == other.terms

    def __repr__(self):
        if not self.terms:
            return "Polynomial(0)"
        parts = []
        for mono, coeff in sorted(self.terms.items(), key=lambda t: (len(t[0]), t[0])):
            body = "*".join(f"x{v}" for v in mono) or "1"
            parts.append(f"{coeff}*{body}")
        return "Polynomial(" + " + ".join(parts) + ")"

    # ── public properties used by the overflow check ────────────────────────

    @property
    def degree(self) -> int:
        """:return: Total degree; 0 for a constant or the zero polynomial"""
        return max((len(m) for m in self.terms), default=0)

    @property
    def num_variables(self) -> int:
        """:return: One past the largest variable index used, i.e. the data length it needs"""
        return max((max(m) + 1 for m in self.terms if m), default=0)

    def worst_case_bound(self, max_input: int) -> int:
        """
        Largest |f(x)| over every integer vector with |x_i| <= max_input.

        By the triangle inequality |f(x)| <= sum |c_a| * prod |x_i| <= sum |c_a| *
        max_input^deg(a). It is attained when every input sits at +-max_input with signs
        that make all terms agree, so it is the exact worst case over the box when the
        monomials can be sign-aligned, and an upper bound otherwise. Either way it is a
        HARD bound, which is what the no-wraparound argument needs.

        :param max_input: Declared bound on the magnitude of every stored integer code
        :return: Bound B(f)
        """
        return sum(abs(c) * max_input ** len(m) for m, c in self.terms.items())

    # ── evaluation ───────────────────────────────────────────────────────────

    def _compile(self, GF: Type[galois.FieldArray]):
        """
        Groups terms by degree into index matrices and field coefficient arrays, once per field.

        Evaluation is then one gather and d multiplies per degree, vectorised over terms,
        rather than a Python loop over terms on every worker.

        :param GF: Field to evaluate in
        :return: List of (index matrix of shape (T, deg), coefficients in GF of shape (T,))
        """
        if GF.order not in self._compiled:
            by_degree: Dict[int, list] = {}
            for mono, coeff in self.terms.items():
                by_degree.setdefault(len(mono), []).append((mono, coeff))
            compiled = []
            for deg, items in sorted(by_degree.items()):
                idx = np.array([m for m, _ in items], dtype=np.int64).reshape(len(items), deg)
                coeffs = GF([c % GF.order for _, c in items])
                compiled.append((idx, coeffs))
            self._compiled[GF.order] = compiled
        return self._compiled[GF.order]

    def __call__(self, z: galois.FieldArray):
        """
        Evaluates the polynomial in the field of z. Integer coefficients are reduced mod q.

        :param z: Data vector or worker share, a 1-D galois array
        :return: Scalar field element f(z)
        """
        GF = type(z)
        if self.num_variables > z.shape[0]:
            raise ValueError(f"polynomial uses x{self.num_variables - 1} but the data has "
                             f"length {z.shape[0]}")
        acc = GF(0)
        for idx, coeffs in self._compile(GF):
            if idx.shape[1] == 0:
                acc = acc + np.sum(coeffs)
                continue
            prod = z[idx[:, 0]]
            for col in range(1, idx.shape[1]):
                prod = prod * z[idx[:, col]]
            acc = acc + np.sum(coeffs * prod)
        return acc

    def evaluate_integers(self, values) -> int:
        """
        Evaluates over the integers with Python's arbitrary precision -- the exact oracle the
        decoded and lifted field result is supposed to equal.

        :param values: Sequence of integer codes
        :return: f(values) as an exact integer
        """
        v = [int(a) for a in values]
        total = 0
        for mono, coeff in self.terms.items():
            term = coeff
            for i in mono:
                term *= v[i]
            total += term
        return total


def variables(n: int):
    """
    The data variables x_0 ... x_{n-1}, for building polynomials with ordinary arithmetic.

    :param n: Data length
    :return: List of n degree-1 Polynomials
    """
    return [Polynomial({(i,): 1}) for i in range(n)]


def least_squares_gradient(n_samples: int, n_features: int, w_codes, f_w: int,
                           j: int) -> Polynomial:
    """
    Builds the degree-2 query for component j of the least-squares gradient.

        g_j(z) = sum_i ( sum_k z[X(i,k)] * w_k  -  2^{f_w} * z[y(i)] ) * z[X(i,j)]

    with the data laid out as the flattened feature matrix followed by the labels. The
    weights are public COEFFICIENTS, not data, so the stored shares never change across
    gradient-descent steps; the 2^{f_w} on the label term brings both terms to the common
    scale 2^{2 f_X + f_w} without making the stored labels depend on f_w
    (docs/QUANTIZATION.md section 5). With every |w_k| at the clip bound, its
    worst_case_bound equals quantize.gradient_bound_worst_case exactly.

    :param n_samples: Number of samples n_s
    :param n_features: Number of features P
    :param w_codes: Quantised weight vector, integer codes
    :param f_w: Fractional bits of the weights
    :param j: Gradient component
    :return: The query polynomial
    """
    x = variables(n_samples * (n_features + 1))
    iX = lambda i, k: i * n_features + k
    iy = lambda i: n_samples * n_features + i
    w = [int(v) for v in w_codes]

    g = Polynomial()
    for i in range(n_samples):
        residual = Polynomial()
        for k in range(n_features):
            residual = residual + w[k] * x[iX(i, k)]
        residual = residual - 2 ** f_w * x[iy(i)]
        g = g + residual * x[iX(i, j)]
    return g


class QueryStatus(Enum):
    """
    Outcome of the server's pre-dispatch check. Anything but OK means the workers were never
    contacted and there is no value to decode.
    """
    OK = "ok"
    MAY_OVERFLOW = "may_overflow"        # B(f) > (q-1)/2: the lifted result could be wrong
    DEGREE_TOO_HIGH = "degree_too_high"  # deg f > d: the shares cannot interpolate it


@dataclass(frozen=True)
class QueryResult:
    """
    What the server returns for one query: either the workers' evaluations, or a refusal
    with the reason. The bound and window are carried either way so the caller can see how
    close a query came.

    bound is None when the context declares no max_input, i.e. the data is native F_q data
    with no integer interpretation, for which overflow is not defined.
    """
    status: QueryStatus
    points: Optional[list] = None
    results: Optional[list] = None
    bound: Optional[int] = None
    window: Optional[int] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """:return: True iff the query was evaluated and is safe to decode"""
        return self.status is QueryStatus.OK


@dataclass(frozen=True)
class DecodedResult:
    """
    What the client gets back from decoding: the field value g(k) = f(x) if the query was
    accepted, or the server's status and reason if it was not.
    """
    status: QueryStatus
    value: Optional[Union[galois.FieldArray, int]] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """:return: True iff value holds a trustworthy result"""
        return self.status is QueryStatus.OK


def check_query(poly: Polynomial, q: int, d: int, max_input: Optional[int]) -> QueryResult:
    """
    The pre-dispatch gate: degree first, then the overflow bound.

    Uses only public quantities -- the query's own coefficients and degree, the field size
    and the declared input bound -- so the decision, including a refusal, is a deterministic
    function of things the provider already holds and leaks nothing about the data.

    :param poly: Query polynomial
    :param q: Field size
    :param d: Maximum degree the shares support
    :param max_input: Declared bound on |stored integer code|, or None for native F_q data
    :return: QueryResult carrying only the status, bound and reason (no evaluations)
    """
    window = (q - 1) // 2
    if poly.degree > d:
        return QueryResult(
            QueryStatus.DEGREE_TOO_HIGH, window=window,
            reason=f"query has degree {poly.degree} but the shares support degree <= {d}; "
                   f"it would decode to a wrong value")
    if max_input is None:
        return QueryResult(QueryStatus.OK, window=window)
    bound = poly.worst_case_bound(max_input)
    if bound > window:
        return QueryResult(
            QueryStatus.MAY_OVERFLOW, bound=bound, window=window,
            reason=f"worst case |f(x)| <= {bound:,} exceeds the window (q-1)/2 = "
                   f"{window:,} (log2 {np.log2(float(bound)):.2f} > "
                   f"{np.log2(float(window)):.2f}); the result could wrap mod q")
    return QueryResult(QueryStatus.OK, bound=bound, window=window)
