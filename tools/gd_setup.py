"""
File: gd_setup.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: The storage-time chain for one Phase 1 training run, in one place
    (docs/GD_LINEAR_REGRESSION.md sections 4.2, 10.2 and 10.4):

        format -> QueryBudget -> B -> q -> entropy (exact, or bounds for rho > 0) -> m
               -> G, key -> encode -> store -> interpolation matrix built once

    plus the public schedule (eta, T) derived from the source's population design. Every
    arrow points forward and nothing reads the sample except the encode itself.

    Seeds: one SeedSequence per run, spawned into independent 'data', 'G' and 'key'
    streams, so changing one never moves the others. A seeded key is for experiments only
    -- a real client must not seed its key (Client docstring).
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import galois
import numpy as np

from client import Client
from config import SystemContext
from quantize import (NATIVE_Q_MAX, FixedPointSpec, QueryBudget, choose_field_size,
                      codes_to_field)
from server import Server
from training import GDSchedule
from utils import (compute_leakage_bound, compute_required_m, generate_random_G,
                   get_information_set, lambda_workers)

from gd_reference import condition_number
from synthetic_source import EntropyResult, SourceSpec

# Phase 1 format, fixed across the whole grid so every comparison is like for like
# (section 10.2: B = n 2^23, native up to n ~ 180).
F_W, CLIP_W, CLIP_B = 4, 2.0, 4.0
# Privacy settings of the case study
R_PRIV, P_ORDER, EPSILON, A_CONF, DEGREE = 5, 2, 1e-6, 5.0, 2
# Run length rule (section 10.4)
TOL_MIN, T_MIN, T_CAP = 1e-3, 200, 2000

_FIELDS = {}


def field_for(q: int):
    """
    galois.GF(q), built once per q and reused; construction is not part of a run's setup
    cost (the field depends only on the public format, not on the upload).

    :param q: Prime field size
    :return: (GF class, seconds the first construction took)
    """
    if q not in _FIELDS:
        t0 = time.perf_counter()
        GF = galois.GF(q) if q <= NATIVE_Q_MAX else galois.GF(q, primitive_element=2,
                                                               verify=False)
        _FIELDS[q] = (GF, time.perf_counter() - t0)
    return _FIELDS[q]


def specs_for(spec: SourceSpec):
    """:return: (x_spec, w_spec, b_spec) of the Phase 1 format"""
    return spec.x_spec, FixedPointSpec(F_W, CLIP_W), FixedPointSpec(F_W, CLIP_B)


def design_schedule(spec: SourceSpec, steps: Optional[int] = None) -> tuple:
    """
    The public schedule, from the population design only (section 10.4):

        eta = 1 / (lambda_max(H_pop) * spread)      H_pop = E[[x;1][x;1]^T], exact from the pmf
        spread = (1 + sqrt((P+1) / n_s))^2           Marchenko-Pastur edge
        T   = clamp(ceil(3 kappa_pop spread ln(1/tol_min)), T_MIN, T_CAP)

    Why the spread factor: with few samples the SAMPLE Hessian's top eigenvalue lands well
    above the population one -- up to about the Marchenko-Pastur edge (1 + sqrt(dim/n_s))^2
    times it -- and eta = 1/lambda_max(H_pop) alone diverged on Gaussian data at n_s = 10,
    P = 3 (smoke run, 2026-11-10). The factor depends only on (n_s, P), so eta stays a
    function of the design and never of the sample. It is conservative for data whose top
    direction is a non-zero mean, which concentrates better.

    :param spec: Source
    :param steps: Override T (smoke runs)
    :return: (GDSchedule, dict with kappa_pop, lam_max_pop, lam_min_pop, spread, T_design)
    """
    kappa, lam_max, lam_min = condition_number(spec.augmented_hessian())
    spread = (1 + math.sqrt((spec.n_features + 1) / spec.n_samples)) ** 2
    T = min(max(T_MIN, math.ceil(3 * kappa * spread * math.log(1 / TOL_MIN))), T_CAP)
    x_spec, w_spec, b_spec = specs_for(spec)
    schedule = GDSchedule(eta=1.0 / (lam_max * spread), steps=steps or T, w_spec=w_spec,
                          f_X=spec.f_X, b_spec=b_spec)
    return schedule, dict(kappa_pop=kappa, lam_max_pop=lam_max, lam_min_pop=lam_min,
                          spread=spread, T_design=T)


def plan_context(spec: SourceSpec, entropy: Optional[EntropyResult] = None) -> dict:
    """
    q, m, lambda and eps_c for a source without uploading anything -- used by --dry-run.

    :return: dict with q, log2_q, budget B, entropy, H_p_X, max_H, m, lam, eps_c (both
        exponents of known issue 1), h, m_source, ctx
    """
    x_spec, w_spec, b_spec = specs_for(spec)
    budget = QueryBudget(spec.n_samples, spec.n_features, DEGREE, x_spec, w_spec,
                         b_spec=b_spec)
    choice = choose_field_size(budget.magnitude_bound(), warn_native=False)
    q, n = choice.q, spec.n
    entropy = entropy or spec.entropies(P_ORDER, R_PRIV)
    ctx = SystemContext(q=q, n=n, d=DEGREE, r=R_PRIV, p=P_ORDER, epsilon=EPSILON, a=A_CONF,
                        max_input=x_spec.max_int)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy.in_log_q(q, n)
    ctx.m = compute_required_m(ctx)
    eps_c = compute_leakage_bound(ctx)
    # Known issue 1: the derivation at eq. (9)-(10) carries q^(-max H / p); report both
    p, a, eps, mh = ctx.p, ctx.a, ctx.epsilon, ctx.max_H_p_X_R
    eps_c_alt = (p / (p - 1)) * math.log(
        1 + a * 2 ** ((2 * p - 1) / p) * (1 + q ** (-mh / p)) * eps ** (1 / p), q)
    return dict(q=q, log2_q=choice.log2_q, native=choice.native,
                B=budget.magnitude_bound(), entropy=entropy, H_p_X=ctx.H_p_X,
                max_H=ctx.max_H_p_X_R, h=ctx.H_p_X / n, m=ctx.m,
                lam=lambda_workers(q, DEGREE, ctx.m), eps_c=eps_c, eps_c_alt=eps_c_alt,
                eps_c_bits=eps_c * choice.log2_q, m_source=entropy.method, ctx=ctx)


@dataclass
class Upload:
    """One uploaded dataset: the harness's plaintext, and the client/server pair."""
    spec: SourceSpec
    data: dict                 # plaintext, harness only
    plan: dict
    client: Client
    server: Server
    timings: dict = field(default_factory=dict)


def upload(spec: SourceSpec, seed, entropy: Optional[EntropyResult] = None) -> Upload:
    """
    Samples a dataset and uploads it through the real scheme, timing each phase.

    :param spec: Source
    :param seed: Entropy for this run's SeedSequence (an int or a list of ints)
    :param entropy: Precomputed entropies of the source (they depend only on the spec)
    :return: Upload
    """
    data_ss, g_ss, key_ss = np.random.SeedSequence(seed).spawn(3)
    data = spec.sample(np.random.default_rng(data_ss))
    plan = plan_context(spec, entropy)
    GF, t_construct = field_for(plan["q"])
    ctx = plan["ctx"]

    t = dict(construct_first=t_construct)
    t0 = time.perf_counter()
    x = codes_to_field(GF, data["codes"], spec.x_spec)
    G = generate_random_G(GF, ctx.m, spec.n, rng=np.random.default_rng(g_ss))
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
    t["setup"] = t["encode"] + t["store"] + t["build"]
    return Upload(spec=spec, data=data, plan=plan, client=client, server=server, timings=t)
