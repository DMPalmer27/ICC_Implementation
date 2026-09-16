"""
File: quantization_study.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Reproducible measurements behind the quantisation design in
    docs/QUANTIZATION.md. Everything the document asserts numerically is produced here,
    seeded, so the paper's tables can be regenerated from scratch.

    Four studies:

      A. Entropy bound validation. The exact Renyi entropy of a discretised, clipped
         Gaussian against the sup-density lower bound of quantize.entropy_bounds. Computed
         from the true cell probabilities, not sampled, so there is no estimator bias --
         which matters because the case study has only 25 samples and no empirical
         estimator can resolve a 2^18-ary distribution from those.

      B. Field-size budget. For a grid of (f_X, f_w): the prime forced by the
         no-wraparound bound, the resulting per-symbol entropy h in log_q units, the
         Theorem 1 key size m, the worker count lambda, and whether q stays under the
         galois native-arithmetic ceiling. Reported for both the public worst-case bound
         and the tighter data-dependent one.

      C. Residue entropy. h_i of a quantised Gaussian reduced mod a small prime, which is
         the quantity the CRT / residue-number-system variant of docs/PLAN.md Week 8 turns
         on. Exact, from cell probabilities.

      D. End-to-end decodability. A real ICC run: quantise, store, query the least-squares
         gradient as a degree-2 polynomial, decode, lift, compare against an exact integer
         oracle and against the plaintext float gradient. This is the check that the whole
         chain -- fixed point, ring homomorphism, balanced lift -- is exact and not merely
         approximately right.

Usage:
    .venv/bin/python tools/quantization_study.py
    .venv/bin/python tools/quantization_study.py --csv results/quantization_budget.csv
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import math

import galois
import numpy as np
from scipy.stats import norm

from client import Client
from config import SystemContext
from quantize import (
    NATIVE_Q_MAX,
    POWER_OF_TWO_GRID,
    SUP_DENSITY_STANDARD_NORMAL,
    FixedPointSpec,
    assert_representable,
    choose_field_size,
    choose_prime,
    codes_to_field,
    entropy_bounds,
    gradient_bound_data_dependent,
    gradient_bound_worst_case,
    leakage_budget_bits,
    lift_balanced,
    symbol_entropy_lower_bits,
    to_integer,
)
from server import Server
from utils import compute_leakage_bound, compute_required_m, generate_random_G

SEED = 0
CLIP = 4.0          # clip bound in standard deviations, for standardised features
R = 5               # privacy parameter used throughout the study
P_ORDER = 2         # entropy order
EPSILON = 1e-6      # smoothing budget


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def discretised_gaussian_pmf(spec: FixedPointSpec) -> np.ndarray:
    """
    Exact pmf of a standard normal pushed through to_integer with nearest rounding.

    Each interior code owns a cell of width 2^-f; the two extreme codes additionally carry
    the clipped tail mass, which is the part of the distribution the sup-density argument
    has to be checked against.

    :param spec: Fixed-point format
    :return: Probability vector over the spec.levels integer codes
    """
    edges = (np.arange(-spec.max_int, spec.max_int + 2) - 0.5) * spec.step
    cdf = norm.cdf(edges)
    pmf = np.diff(cdf)
    pmf[0] += norm.cdf(edges[0])          # lower tail saturates onto the lowest code
    pmf[-1] += 1.0 - norm.cdf(edges[-1])  # upper tail onto the highest
    return pmf


def renyi_bits(pmf: np.ndarray, order: int) -> float:
    """
    Renyi entropy of a pmf in bits.

    :param pmf: Probability vector
    :param order: Entropy order, > 1
    :return: H_order(pmf) in bits
    """
    p = pmf[pmf > 0]
    return (1.0 / (1 - order)) * math.log2(float(np.sum(p ** order)))


def saturation_mass(clip: float) -> float:
    """
    Mass a standard normal puts on one clip boundary, i.e. the size of the saturation atom.

    This is the term that competes with the interior-cell bound in
    quantize.symbol_entropy_lower_bits, and the one that binds once f is large.

    :param clip: Clip bound in standard deviations
    :return: P(X < -clip) for X ~ N(0, 1)
    """
    return float(norm.cdf(-clip))


def theorem1(n: int, q: int, spec: FixedPointSpec, r: int = R,
             epsilon: float = EPSILON) -> tuple[float, int, int, float]:
    """
    Runs the Theorem 1 bound with the quantisation entropy bounds.

    :param n: Data length
    :param q: Field size
    :param spec: Fixed-point format of the stored data
    :param r: Privacy parameter
    :param epsilon: Smoothing budget
    :return: (h in log_q units, key size m, worker count lambda, leakage bound eps_c)
    """
    ctx = SystemContext(q=q, n=n, d=2, r=r, p=P_ORDER, epsilon=epsilon)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy_bounds(
        n, r, q, spec, saturation_mass=saturation_mass(spec.clip))
    ctx.m = compute_required_m(ctx)
    return ctx.H_p_X / n, ctx.m, math.comb(ctx.m + ctx.d, ctx.d), compute_leakage_bound(ctx)


def case_study_data(n_samples: int, n_features: int, seed: int = SEED):
    """
    The diabetes subsample of docs/PLAN.md, standardised per column.

    Standardising is what lets one FixedPointSpec serve every feature; with per-feature
    scales the query polynomial would need a public rescaling multiplier per (j, k) pair,
    which inflates the magnitude bound to the worst feature's scale anyway.

    :param n_samples: Samples to draw
    :param n_features: Feature columns to keep
    :param seed: RNG seed
    :return: (X standardised, y standardised)
    """
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data.data), n_samples, replace=False)
    X = data.data[idx][:, :n_features]
    y = data.target[idx]
    return (X - X.mean(0)) / X.std(0), (y - y.mean()) / y.std()


# ─────────────────────────────────────────────────────────────────────────────
# Study A -- is the sup-density entropy bound valid, and how loose?
# ─────────────────────────────────────────────────────────────────────────────

def study_entropy_bound():
    """
    Checks the lower bound H_p >= f - log2(sup density) against the exact entropy of a
    discretised clipped Gaussian, for several f and several orders p.
    """
    section("A. Entropy lower bound vs exact discretised-Gaussian entropy")
    bound_const = -math.log2(SUP_DENSITY_STANDARD_NORMAL)
    print(f"  interior-cell term:  f + {bound_const:.3f} bits")
    print(f"  saturation term:     -log2 P(|X| clipped on one side), independent of f")
    print(f"  bound is the MINIMUM of the two, and holds for every order p > 1\n")

    for clip in (4.0, 6.0):
        sat = saturation_mass(clip)
        print(f"  clip = {clip:.0f} sigma  (saturation term = {-math.log2(sat):.2f} bits, "
              f"atom mass {sat:.1e})")
        print(f"  {'f':>3} {'levels':>9} {'H_2':>8} {'H_3':>8} {'H_8':>8} {'bound':>8} "
              f"{'binds':>10} {'slack(H_8)':>11} {'log2 levels':>12}")
        for f in (4, 8, 12, 16, 20):
            spec = FixedPointSpec(f=f, clip=clip)
            pmf = discretised_gaussian_pmf(spec)
            h2, h3, h8 = (renyi_bits(pmf, o) for o in (2, 3, 8))
            interior = f + bound_const
            bound = symbol_entropy_lower_bits(spec, saturation_mass=sat)
            binds = "interior" if bound == interior else "saturation"
            print(f"  {f:>3} {spec.levels:>9,} {h2:>8.3f} {h3:>8.3f} {h8:>8.3f} "
                  f"{bound:>8.3f} {binds:>10} {h8 - bound:>11.3f} {spec.log2_levels:>12.3f}")
        print()

    print("  Where the interior term binds, slack is 0.500 bits at H_2 regardless of f:")
    print("  the exact Gaussian value is f + log2(2*sqrt(pi)) = f + 1.825 against a bound")
    print("  of f + log2(sqrt(2*pi)). That is tight enough to use in place of an entropy")
    print("  estimate -- which the case study cannot produce anyway, since 25 samples")
    print("  cannot resolve a 2^18-ary distribution.")
    print("\n  The saturation term is the trap. At clip = 4 sigma, f = 16 the exact")
    print("  H_8 = 16.93 falls BELOW the interior term f + 1.325 = 17.33, because the")
    print("  clipped tail atom has overtaken a typical interior cell. Ignoring the")
    print("  saturation term there would have over-estimated H_p(X) and under-estimated m,")
    print("  i.e. failed in the unsafe direction. Rule: keep P_sat <= 2^-f * M.")
    print(f"\n  {'f':>3} {'min clip (sigma) keeping the interior term binding':>52}")
    for f in (8, 12, 16, 20, 24):
        need = 2.0 ** -f * SUP_DENSITY_STANDARD_NORMAL
        print(f"  {f:>3} {float(norm.isf(need)):>52.2f}")
    print("  Widening the clip is cheap: it enters the magnitude bound polynomially while")
    print("  the tail mass decays like a Gaussian. At n_s=25, P=3, f_X=8, f_w=0, going from")
    print("  4 to 5 sigma costs 0.94 bits of q and buys 6.79 bits of entropy headroom.")
    print("  So widen the clip, do not weaken the bound.")


# ─────────────────────────────────────────────────────────────────────────────
# Study B -- the field-size budget and what it costs in workers
# ─────────────────────────────────────────────────────────────────────────────

def study_field_budget(n_samples: int = 25, n_features: int = 3):
    """
    Sweeps (f_X, f_w) and reports the forced prime, h, m, lambda under both magnitude
    bounds.

    :param n_samples: Samples in the case study
    :param n_features: Features in the case study
    :return: List of row dicts for CSV
    """
    section(f"B. Field-size budget, n = {n_samples * (n_features + 1)} "
            f"({n_samples} samples x {n_features} features + {n_samples} labels)")

    X, y = case_study_data(n_samples, n_features)
    n = n_samples * (n_features + 1)
    print(f"  diabetes subsample, standardised: max|X| = {np.abs(X).max():.2f} sigma, "
          f"max|y| = {np.abs(y).max():.2f} sigma, clip = {CLIP:.0f} sigma")
    print(f"  r = {R}, p = {P_ORDER}, eps = {EPSILON:g}, d = 2, "
          f"native ceiling log2 q <= {math.log2(NATIVE_Q_MAX):.1f}")

    print(f"\n{'':>9} | {'worst-case bound (public)':^34} | "
          f"{'data-dependent bound (leaky)':^34} |")
    print(f"{'f_X':>4} {'f_w':>4} | {'log2q':>6} {'h':>6} {'m':>4} {'lambda':>8} {'nat':>3} | "
          f"{'log2q':>6} {'h':>6} {'m':>4} {'lambda':>8} {'nat':>3} | bits saved")
    print("-" * 100)

    rows = []
    for f_X in (6, 8, 10, 12):
        for f_w in (0, 4, 8):
            x_spec = FixedPointSpec(f=f_X, clip=CLIP)
            w_spec = FixedPointSpec(f=f_w, clip=CLIP)
            Xq = to_integer(X, x_spec)
            yq = to_integer(y, x_spec)

            bounds = {
                "worst_case": gradient_bound_worst_case(n_samples, n_features, x_spec, w_spec),
                "data_dependent": gradient_bound_data_dependent(Xq, yq, w_spec),
            }
            out = {}
            for kind, B in bounds.items():
                q = choose_prime(B, warn_native=False)
                h, m, lam, eps_c = theorem1(n, q, x_spec)
                out[kind] = (q, math.log2(q), h, m, lam, eps_c)
                rows.append(dict(bound=kind, f_X=f_X, f_w=f_w, n=n, q=q,
                                 log2_q=round(math.log2(q), 3), magnitude_bound=B,
                                 h=round(h, 4), m=m, lam=lam, eps_c=round(eps_c, 6),
                                 native=q <= NATIVE_Q_MAX))
            (_, b1, h1, m1, l1, _), (_, b2, h2, m2, l2, _) = \
                out["worst_case"], out["data_dependent"]
            nat = lambda b: "y" if 2 ** b <= NATIVE_Q_MAX else "n"
            print(f"{f_X:>4} {f_w:>4} | {b1:>6.1f} {h1:>6.3f} {m1:>4} {l1:>8,} {nat(b1):>3} | "
                  f"{b2:>6.1f} {h2:>6.3f} {m2:>4} {l2:>8,} {nat(b2):>3} | {b1 - b2:>5.1f}")
        print()

    print("  Reading it: f_w is the dominant knob because it enters log2 q once while f_X")
    print("  enters twice but also raises the numerator of h. The native ceiling binds:")
    print("  2 f_X + f_w <= ~18 under the worst-case bound at this n_s and clip.")

    # The comparison that settles which bound to ship. eps_c is in log_q units; convert to
    # bits so the two channels are on the same scale.
    x_spec, w_spec = FixedPointSpec(8, CLIP), FixedPointSpec(0, CLIP)
    B_wc = gradient_bound_worst_case(n_samples, n_features, x_spec, w_spec)
    B_dd = gradient_bound_data_dependent(to_integer(X, x_spec), to_integer(y, x_spec), w_spec)
    public = choose_field_size(B_wc, warn_native=False)
    coarse = choose_field_size(B_dd, data_dependent=True, grid=POWER_OF_TWO_GRID,
                               warn_native=False)
    _, _, _, eps_c = theorem1(n, public.q, x_spec)
    budget = leakage_budget_bits(eps_c, public.q)
    print(f"\n  Choosing q: the leakage accounting (f_X = 8, f_w = 0)")
    print(f"    {public}")
    print(f"    {coarse}")
    print(f"    bits of q saved by coarsening       : {public.log2_q - coarse.log2_q:.2f}")
    print(f"    scheme's own budget, eps_c={eps_c:.4f}   : {budget:.3f} bits per r-subset")
    print(f"    side channel opened to save them    : {coarse.leakage_bits:.2f} bits "
          f"({coarse.leakage_bits / budget:.0f}x the budget)")
    print(f"  So use the public-format bound. The data-dependent one is an instrument for")
    print(f"  quantifying what is given up, not a configuration to ship.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Study C -- residue entropy, the lever behind the CRT variant
# ─────────────────────────────────────────────────────────────────────────────

def study_residue_entropy():
    """
    Exact h_i = H_2(quantised symbol mod q_i) / log2 q_i for a grid of (f_X, q_i).

    The point: folding a source with H_2 = f + 1.825 bits into a field with
    log2 q_i <= that value makes the residue essentially uniform, so h_i -> 1, and then
    Theorem 1's n(1 - h) term vanishes and m stops scaling with n.
    """
    section("C. Residue entropy h_i for the CRT / RNS variant (exact, clip 4 sigma)")
    primes = (1031, 8209, 65537, 1073741827)
    print(f"{'f_X':>4} {'H_2 bits':>9} |" +
          "".join(f" h_i(q={p:>10,})" for p in primes))
    print("-" * (16 + 18 * len(primes)))

    for f in (6, 8, 10, 12, 16):
        spec = FixedPointSpec(f=f, clip=CLIP)
        pmf = discretised_gaussian_pmf(spec)
        codes = np.arange(-spec.max_int, spec.max_int + 1)
        cells = []
        for qi in primes:
            folded = np.bincount(codes % qi, weights=pmf, minlength=qi)
            folded /= folded.sum()
            cells.append(renyi_bits(folded, 2) / math.log2(qi))
        print(f"{f:>4} {renyi_bits(pmf, 2):>9.3f} |" +
              "".join(f" {c:>17.4f}" for c in cells))

    print("\n  h_i = min(H_2(symbol), log2 q_i) / log2 q_i to within numerical display:")
    print("  entropy in bits is capped by the source, so shrinking the field raises the")
    print("  ratio. Design rule: pick each CRT prime with log2 q_i <= H_p(symbol) in bits")
    print("  and h_i = 1, giving m_i >= r + p + log_{q_i}(1/eps), independent of n.")
    print("  Both Theorem 1 entropy terms then become tight and assumption-free")
    print("  (H_p(X) = n, max_R = r), which also sidesteps the correlated-features problem.")
    print("  Correctness is immediate (reduction mod q_i is a ring homomorphism); the")
    print("  privacy composition across the L instances is NOT -- docs/PLAN.md question 5.")


# ─────────────────────────────────────────────────────────────────────────────
# Study D -- end to end through the real scheme
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_polynomial(GF, n_samples: int, n_features: int, w_codes, f_w: int, j: int):
    """
    Builds the degree-2 query polynomial for gradient component j.

    g_j(z) = sum_i ( sum_k z[X(i,k)] * wq[k]  -  2^{f_w} * z[y(i)] ) * z[X(i,j)]

    The weights appear as public COEFFICIENTS, not as data: the workers' shares are written
    once at storage time and never change across the training run, so a gradient step costs
    a fresh polynomial and nothing else. The 2^{f_w} multiplier on the label term is what
    aligns the two terms at the common scale 2^{2 f_X + f_w} without making the stored
    quantisation of y depend on the weight precision.

    :param GF: Galois field object
    :param n_samples: Number of samples
    :param n_features: Number of features
    :param w_codes: Quantised weight vector, integer codes
    :param f_w: Fractional bits of the weights
    :param j: Gradient component to build
    :return: Callable taking a worker's share and returning a field element
    """
    iX = lambda i, k: i * n_features + k
    iy = lambda i: n_samples * n_features + i
    w_gf = [GF(int(v) % GF.order) for v in w_codes]
    two_fw = GF(2 ** f_w % GF.order)

    def f(z):
        acc = GF(0)
        for i in range(n_samples):
            residual = GF(0)
            for k in range(n_features):
                residual = residual + z[iX(i, k)] * w_gf[k]
            residual = residual - two_fw * z[iy(i)]
            acc = acc + residual * z[iX(i, j)]
        return acc

    return f


def study_end_to_end(n_samples: int = 4, n_features: int = 2, f_X: int = 8, f_w: int = 4):
    """
    Runs the full scheme on quantised data and checks the decoded gradient three ways.

    Kept small so it runs in seconds; the point is exactness, not scale. The exactness
    claim has two independent parts and both are checked: the field computation reproduces
    the integer computation (ring homomorphism), and the balanced lift recovers the signed
    integer (no wraparound).

    :param n_samples: Samples
    :param n_features: Features
    :param f_X: Fractional bits of the stored data
    :param f_w: Fractional bits of the weights
    """
    section(f"D. End-to-end: quantise -> store -> query gradient -> decode -> lift")

    rng = np.random.default_rng(SEED)
    n = n_samples * (n_features + 1)
    x_spec = FixedPointSpec(f=f_X, clip=CLIP)
    w_spec = FixedPointSpec(f=f_w, clip=CLIP)

    X = rng.normal(size=(n_samples, n_features))
    w_true = rng.normal(size=n_features)
    y = X @ w_true + 0.1 * rng.normal(size=n_samples)
    w = rng.normal(size=n_features)

    Xq, yq, wq = to_integer(X, x_spec), to_integer(y, x_spec), to_integer(w, w_spec)

    B = gradient_bound_worst_case(n_samples, n_features, x_spec, w_spec)
    q = choose_prime(B)
    GF = galois.GF(q)
    ctx = SystemContext(q=q, n=n, d=2, r=3, p=P_ORDER, epsilon=1e-4)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy_bounds(n, ctx.r, q, x_spec)
    ctx.m = compute_required_m(ctx)
    lam = math.comb(ctx.m + ctx.d, ctx.d)

    print(f"  n = {n}, f_X = {f_X}, f_w = {f_w}, clip = {CLIP:.0f}")
    print(f"  worst-case |g| bound B = {B:,}  ->  q = {q:,} (log2 q = {math.log2(q):.1f})")
    print(f"  h = {ctx.H_p_X / n:.3f}, m = {ctx.m}, lambda = {lam}, "
          f"eps_c = {compute_leakage_bound(ctx):.6f}")

    # Storage: the data vector is the flattened feature matrix followed by the labels
    x_field = codes_to_field(GF, np.concatenate([Xq.ravel(), yq]), x_spec)
    client, server = Client(ctx, GF), Server(ctx, GF)
    G = generate_random_G(GF, ctx.m, n)
    server.store_data(client.encode_data(x_field, G), G)

    scale = 2.0 ** (2 * f_X + f_w)
    g_float = (X @ w - y) @ X
    # Second reference: the float gradient evaluated at the DEQUANTISED operands. The gap
    # to g_float is the quantisation error of the inputs; the gap to this is the error the
    # field computation itself introduces, which should be exactly zero.
    Xd, yd, wd = Xq / 2 ** f_X, yq / 2 ** f_X, wq / 2 ** f_w
    g_dequant = (Xd @ wd - yd) @ Xd

    print(f"\n{'j':>3} {'decoded (lifted)':>20} {'integer oracle':>20} {'exact':>6} "
          f"{'decoded/2^s':>13} {'float grad':>13} {'err vs float':>13} "
          f"{'err vs dequant':>15}")

    all_exact = True
    for j in range(n_features):
        points, results = server.compute_request(
            _gradient_polynomial(GF, n_samples, n_features, wq, f_w, j))
        decoded = int(lift_balanced(client.decode_result(points, results), q))

        oracle = int(sum(
            (sum(int(Xq[i, k]) * int(wq[k]) for k in range(n_features))
             - 2 ** f_w * int(yq[i])) * int(Xq[i, j])
            for i in range(n_samples)))
        assert_representable(oracle, q, what=f"gradient component {j}")

        exact = decoded == oracle
        all_exact &= exact
        approx = decoded / scale
        print(f"{j:>3} {decoded:>20,} {oracle:>20,} {str(exact):>6} "
              f"{approx:>13.6f} {g_float[j]:>13.6f} {abs(approx - g_float[j]):>13.2e} "
              f"{abs(approx - g_dequant[j]):>15.2e}")

    worst = max(abs(int(sum((sum(int(Xq[i, k]) * int(wq[k]) for k in range(n_features))
                             - 2 ** f_w * int(yq[i])) * int(Xq[i, j])
                            for i in range(n_samples)))) for j in range(n_features))
    print(f"\n  field computation reproduces the integer computation: {all_exact}")
    print(f"  worst-case bound B = {B:,} vs actual max |g| = {worst:,} "
          f"-> {B / worst:.0f}x loose ({math.log2(B / worst):.1f} bits of q)")
    print(f"  'err vs dequant' is the error the SCHEME contributes: zero to float64")
    print(f"  rounding. 'err vs float' is input quantisation only, and at f_w = {f_w} it is")
    print(f"  dominated by the weight precision, not by f_X = {f_X}.")
    return all_exact


def main():
    parser = argparse.ArgumentParser(description="ICC quantisation design study")
    parser.add_argument("--csv", help="write the field-size budget rows to this path")
    args = parser.parse_args()

    print(f"seed = {SEED}; every number below is reproducible from this script")
    study_entropy_bound()
    rows = study_field_budget()
    study_residue_entropy()
    ok = study_end_to_end()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} budget rows to {args.csv}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
