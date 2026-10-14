"""
File: q_sweep.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Parameter sweep measuring how the cost of the full ICC scheme -- field
    construction, storage, worker evaluation and decoding -- depends on the field size q,
    and which parameter combinations are feasible. Raw rows go to results/q_sweep_*.csv;
    tools/plot_sweep.py turns them into the figures, and docs/Q_SWEEP.md interprets them.

    The question it answers is whether a massive q, as quantising reals into F_q demands,
    makes computation explode. Two effects have to be kept apart, because they have
    different causes and different fixes:

      * PER-OPERATION COST. galois keeps elements as machine integers and compiled ufuncs only
        while q^2 < 2^63 (quantize.NATIVE_Q_MAX, log2 q <= 31.5). Above that every element is
        a Python int and every operation goes through the interpreter. Experiments A and B
        hold the problem size (m, hence lambda) FIXED and vary q, so any change in time is
        purely arithmetic cost.

      * PROBLEM SIZE. With quantised data the key length m is not free: Theorem 1 is in log_q
        units and a quantised symbol carries a fixed number of bits, so h = H/log2 q falls and
        m ~ n(1 - h) grows with q -- bounded above by about n + p. Experiment C derives q, m
        and lambda from a real-valued storage format and measures the end-to-end cost.

    CRT regime (a) is measured alongside C as a comparison: L instances over distinct primes
    just below the native ceiling, product Q >= 2B + 1, recombined by the Chinese Remainder
    Theorem. Every instance uses the SAME m, sized for Q (log_Q units), which makes it the same
    scheme as one big prime of size ~Q computed in native arithmetic. Sizing each instance's m
    for its own small prime instead is NOT safe -- the provider intersects the L cosets and
    recovers x; see docs/Q_SWEEP.md section 6. The privacy of the jointly sized version still
    needs the smoothing lemma over the ring Z_Q (an open question for Raviv), but its COST is
    what this sweep measures, and cost does not depend on that answer.

    Every measured cell checks correctness before its timings are kept: the decoded value must
    equal the field evaluation (experiments A, B) or the exact integer oracle (C, CRT).

    Experiments:
      T  theory     m and lambda versus log2 q, and the q a format requires (no timing)
      A  q-scaling  fixed m in {20, 48}, q from 2^13 to 2^521
      B  lambda     m from 10 to 100 at q = 2^30 (native), 2^61, 2^127
      C  formats    (n_s, f_X, f_w) grid -> q, m, lambda -> measured, single prime and CRT
      D  construct  galois.GF(q) WITH its primitive-element search, capped, per q

Usage:
    .venv/bin/python tools/q_sweep.py --exp all --jobs 4
    .venv/bin/python tools/q_sweep.py --exp A --fast          # smoke run, ~1 min
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import galois
import numpy as np

from client import Client
from config import SystemContext
from query import check_query, least_squares_gradient
from quantize import (NATIVE_Q_MAX, FixedPointSpec, QueryBudget, choose_field_size,
                      codes_to_field, entropy_bounds, lift_balanced, to_integer)
from server import Server
from utils import (compute_leakage_bound, compute_required_m, generate_random_G,
                   get_information_set, lambda_workers)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 2026

# Scheme parameters held fixed across the sweep: the case study's privacy settings
R_PRIV, P_ORDER, EPSILON, DEGREE = 5, 2, 1e-6, 2
CLIP_X, CLIP_W = 4.0, 2.0      # A_X = 4 sigma, A_w = 2 (docs/QUANTIZATION.md section 6.1)
N_FEATURES = 3                 # P; n = n_s (P + 1)

# Field sizes for experiment A. 31.5 is the largest prime galois keeps native.
BITS_A = (13, 20, 25, 30, 31.5, 32, 40, 48, 61, 64, 89, 127, 256, 521)
BITS_B = (30, 61, 127)
M_A = (20, 48)
M_B = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
# Experiment C grid: n_s in {10, 25, 50} gives n = 40, 100, 200
NS_C = (10, 25, 50)
FX_C = (4, 7, 10, 13, 16)
FW_C = (4, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Fields and primes
# ─────────────────────────────────────────────────────────────────────────────

def prime_for_bits(bits: float) -> int:
    """
    A fixed, reproducible prime of the requested size.

    31.5 means the largest native prime. Above 2^200 a prime below 2^bits is found once and
    is cheap to recheck; the two Mersenne primes are used where they exist so the field is a
    recognisable one.

    :param bits: Requested log2 q
    :return: A prime q with log2 q close to bits
    """
    if bits == 31.5:
        return int(galois.prev_prime(NATIVE_Q_MAX))
    known = {61: 2 ** 61 - 1, 127: 2 ** 127 - 1, 256: 2 ** 256 - 189, 521: 2 ** 521 - 1}
    if bits in known:
        return known[bits]
    return int(galois.prev_prime(2 ** int(bits)))


def make_field(q: int):
    """
    Builds GF(q) without the primitive-element search above the native ceiling.

    galois.GF(q) normally finds a primitive element, which needs the factorisation of q - 1;
    that is instant for structured primes but hangs for a generic large one (2^521 - 1 did
    not finish in 10 minutes). The scheme never uses the primitive element -- it only adds,
    multiplies and solves -- so it is skipped by passing one with verify=False. The element
    passed is a placeholder and is NOT verified to be primitive; nothing here may call
    log/exp on these fields. Experiment D measures the verified construction separately.

    :param q: Prime field size
    :return: galois FieldArray subclass for GF(q)
    """
    if q <= NATIVE_Q_MAX:
        return galois.GF(q)
    return galois.GF(q, primitive_element=2, verify=False)


def crt_primes(target: int):
    """
    The fewest distinct native primes whose product just reaches the target, chosen of
    EQUAL size rather than as large as possible.

    The size matters for privacy cost, not just arithmetic: m is sized jointly in log_Q units
    (docs/Q_SWEEP.md section 6), so an oversized Q lowers h and inflates m and lambda for no
    gain. Taking L = ceil(log2(target) / 31.5) primes, each just above target^(1/L), keeps
    Q ~ target -- the same modulus a single prime would need -- so CRT and single-prime runs
    of the same format get the same m and differ only in arithmetic. (A first version took
    the largest native primes; a 34.5-bit format then got Q ~ 2^63 and m = 90 instead of 80.)

    :param target: Required product, 2B + 1 for a magnitude bound B
    :return: List of L distinct primes, each <= NATIVE_Q_MAX, with product >= target
    """
    L = max(1, math.ceil(math.log2(target) / math.log2(NATIVE_Q_MAX)))
    while True:
        base = math.ceil(target ** (1.0 / L)) if L > 1 else target
        primes, q = [], base - 1
        for _ in range(L):
            q = int(galois.next_prime(q))
            primes.append(q)
        if math.prod(primes) >= target and primes[-1] <= NATIVE_Q_MAX:
            return primes
        L += 1      # rounding pushed a prime past the native limit; use one more


def crt_combine(residues, primes) -> int:
    """
    Chinese Remainder reconstruction, then the balanced lift mod Q = prod(primes).

    :param residues: r_i = value mod q_i
    :param primes: The q_i, pairwise coprime
    :return: The unique integer in [-(Q-1)/2, (Q-1)/2] with the given residues
    """
    Q = math.prod(primes)
    total = 0
    for r, q in zip(residues, primes):
        Mi = Q // q
        total += int(r) * Mi * pow(Mi, -1, q)
    total %= Q
    return total - Q if total > (Q - 1) // 2 else total


# ─────────────────────────────────────────────────────────────────────────────
# Theory (no timing)
# ─────────────────────────────────────────────────────────────────────────────

def theorem1_m(n: int, Q: int, x_spec: FixedPointSpec) -> tuple:
    """
    Key length for quantised data stored in a field (or CRT ring) of size Q.

    Uses the rigorous entropy BOUNDS of quantize.entropy_bounds (docs/QUANTIZATION.md
    section 8; open questions 1 and 9 in docs/PLAN.md), so every m reported here is
    conditional on that proposal being accepted.

    :param n: Data length
    :param Q: Field size, or product of CRT primes
    :param x_spec: Storage format
    :return: (m, h, eps_c)
    """
    ctx = SystemContext(q=Q, n=n, d=DEGREE, r=R_PRIV, p=P_ORDER, epsilon=EPSILON)
    ctx.H_p_X, ctx.max_H_p_X_R = entropy_bounds(n, R_PRIV, Q, x_spec)
    ctx.m = compute_required_m(ctx)
    return ctx.m, ctx.H_p_X / n, compute_leakage_bound(ctx)


def theory_rows() -> list:
    """
    Two analytic tables: m, lambda against log2 q for fixed formats, and the field size each
    format requires. Cheap, exact, and the context every timing has to be read in.

    :return: Rows tagged kind = 'm_vs_q' or 'q_vs_format'
    """
    rows = []
    for n_s in (10, 25, 50, 100):
        n = n_s * (N_FEATURES + 1)
        x_spec = FixedPointSpec(f=7, clip=CLIP_X)
        for bits in np.arange(20, 524, 4.0):
            m, h, eps_c = theorem1_m(n, 2 ** int(bits), x_spec)
            rows.append(dict(kind="m_vs_q", n_s=n_s, n=n, f_X=7, log2_q=float(bits), m=m,
                             h=h, lam=lambda_workers(2 ** int(bits), DEGREE, m), eps_c=eps_c,
                             m_cap=n + P_ORDER))
    for n_s in (10, 25, 50, 100, 1000):
        for f_w in (0, 4, 8):
            for f_X in range(2, 25):
                x_spec, w_spec = FixedPointSpec(f_X, CLIP_X), FixedPointSpec(f_w, CLIP_W)
                B = QueryBudget(n_s, N_FEATURES, DEGREE, x_spec, w_spec).magnitude_bound()
                rows.append(dict(kind="q_vs_format", n_s=n_s, f_X=f_X, f_w=f_w,
                                 log2_q=math.log2(2 * B + 1),
                                 crt_L=len(crt_primes(2 * B + 1))))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# One measured cell (runs in its own subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _rss_bytes() -> int:
    """:return: Peak resident set size of this process so far, in bytes"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def measure_cell(spec: dict) -> dict:
    """
    Runs storage, P gradient queries and their decoding, with every phase timed.

    Phases, all through the real Client/Server code paths:
      construct  galois.GF(q_i) for each instance
      build      information set + interpolation matrix M (client cache; paid once)
      encode     key draw and x~ = x + kG
      store      server sharding x~ - tG to lambda workers
      eval       the overflow gate + every worker evaluating every query   (per step)
      decode     one elimination for all P queries, evaluate at k, lift    (per step)

    eval and decode are repeated; the first repeat is reported as cold (it compiles the
    polynomials for the field) and the median of the rest as warm. A training step costs
    eval + decode; construct + build + encode + store is paid once per upload.

    :param spec: Cell description built by the experiment builders
    :return: The spec plus timings, sizes, memory and the correctness verdict
    """
    rng = np.random.default_rng(spec["seed"])
    n_s, P, d = spec["n_s"], spec["P"], spec["d"]
    n = n_s * (P + 1)
    primes = [int(q) for q in spec["primes"]]
    Q = math.prod(primes)
    m = spec["m"]
    integer_mode = spec["mode"] == "quant"

    # Data and queries
    if integer_mode:
        x_spec, w_spec = FixedPointSpec(spec["f_X"], CLIP_X), FixedPointSpec(spec["f_w"], CLIP_W)
        X, y = rng.standard_normal((n_s, P)), rng.standard_normal(n_s)
        codes = np.concatenate([to_integer(X, x_spec).ravel(), to_integer(y, x_spec)])
        w_codes = to_integer(rng.uniform(-CLIP_W, CLIP_W, P), w_spec)
        f_w, max_input = spec["f_w"], x_spec.max_int
    else:
        codes, w_codes, f_w, max_input = None, rng.integers(-32, 33, P), 4, None
    queries = [least_squares_gradient(n_s, P, w_codes, f_w, j) for j in range(P)]

    row = dict(spec, n=n, lam=lambda_workers(Q, d, m), log2_q=math.log2(Q), L=len(primes),
               native=all(q <= NATIVE_Q_MAX for q in primes), status="ok")
    t = dict(construct=0.0, build=0.0, encode=0.0, store=0.0)
    instances = []
    for i, q in enumerate(primes):
        t0 = time.perf_counter(); GF = make_field(q); t["construct"] += time.perf_counter() - t0
        # Per-instance gate is degree-only (max_input None) under CRT: the overflow window is
        # (Q-1)/2, checked once below, not (q_i - 1)/2.
        ctx = SystemContext(q=q, n=n, d=d, r=R_PRIV, p=P_ORDER, epsilon=EPSILON,
                            max_input=max_input if len(primes) == 1 else None)
        ctx.m = m
        inst_rng = np.random.default_rng([spec["seed"], i])
        client, server = Client(ctx, GF, rng=inst_rng), Server(ctx, GF)
        x = codes_to_field(GF, codes, x_spec) if integer_mode else GF.Random(n, seed=inst_rng)
        G = generate_random_G(GF, m, n, rng=inst_rng)

        t0 = time.perf_counter(); x_tilde = client.encode_data(x, G)
        t["encode"] += time.perf_counter() - t0
        t0 = time.perf_counter(); server.store_data(x_tilde, G)
        t["store"] += time.perf_counter() - t0
        # Build M now, through the client's cache, so per-step decode times are warm
        t0 = time.perf_counter()
        client._interpolation_matrix(server.points, get_information_set(q, m, d))
        t["build"] += time.perf_counter() - t0
        instances.append((GF, client, server, x))

    gate = check_query(queries[0], Q, d, max_input)   # identical bound for every component
    row.update(bound_log2=math.log2(gate.bound) if gate.bound else None)
    if not gate.ok:
        row.update(status=f"refused:{gate.status.value}")
        return row

    evals, decodes, exact = [], [], True
    for rep in range(spec["repeats"]):
        te = td = 0.0
        residues = [[] for _ in queries]
        for GF, client, server, x in instances:
            t0 = time.perf_counter()
            responses = [server.compute_request(f) for f in queries]
            te += time.perf_counter() - t0
            t0 = time.perf_counter()
            decoded = client.decode_results(responses)
            td += time.perf_counter() - t0
            for j, dres in enumerate(decoded):
                exact &= dres.ok
                residues[j].append(int(dres.value) if dres.ok else None)
                if not integer_mode:
                    exact &= dres.ok and dres.value == queries[j](x)
        if integer_mode:
            for j, f in enumerate(queries):
                value = (crt_combine(residues[j], primes) if len(primes) > 1
                         else int(lift_balanced(np.array(residues[j][0], dtype=object), Q)))
                exact &= value == f.evaluate_integers(codes)
        evals.append(te)
        decodes.append(td)

    warm = lambda v: float(np.median(v[1:])) if len(v) > 1 else float(v[0])
    row.update(t_construct=t["construct"], t_build=t["build"], t_encode=t["encode"],
               t_store=t["store"], t_eval_cold=evals[0], t_decode_cold=decodes[0],
               t_eval=warm(evals), t_decode=warm(decodes),
               t_setup=sum(t.values()), exact=bool(exact),
               peak_rss_mb=_rss_bytes() / 2 ** 20)
    row["t_step"] = row["t_eval"] + row["t_decode"]
    if not exact:
        row["status"] = "WRONG"
    return row


def construct_cell(spec: dict) -> dict:
    """
    Experiment D: the default, verified galois.GF(q), which searches for a primitive element.

    :param spec: Cell with 'primes' holding one prime
    :return: The spec plus construction time
    """
    q = int(spec["primes"][0])
    t0 = time.perf_counter()
    galois.GF(q)
    return dict(spec, log2_q=math.log2(q), t_construct_verified=time.perf_counter() - t0,
                status="ok")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment builders
# ─────────────────────────────────────────────────────────────────────────────

def _cell(exp, primes, m, n_s=25, mode="field", repeats=3, **extra):
    """:return: A cell spec; primes are stored as strings so JSON keeps big ints exact"""
    return dict(exp=exp, primes=[str(q) for q in primes], m=m, n_s=n_s, P=N_FEATURES,
                d=DEGREE, mode=mode, repeats=repeats, seed=SEED, **extra)


def build_A(fast: bool):
    bits = (13, 30, 32, 127) if fast else BITS_A
    return [_cell("A", [prime_for_bits(b)], m, bits_label=b,
                  repeats=2 if fast else 3)
            for m in ((20,) if fast else M_A) for b in bits]


def build_B(fast: bool):
    ms = (10, 30) if fast else M_B
    return [_cell("B", [prime_for_bits(b)], m, bits_label=b, repeats=2 if fast else 3)
            for b in BITS_B for m in ms]


def build_C(fast: bool):
    cells = []
    grid = [(10, 7, 4), (10, 16, 8)] if fast else \
        [(ns, fx, fw) for ns in NS_C for fx in FX_C for fw in FW_C]
    for n_s, f_X, f_w in grid:
        n = n_s * (N_FEATURES + 1)
        x_spec, w_spec = FixedPointSpec(f_X, CLIP_X), FixedPointSpec(f_w, CLIP_W)
        B = QueryBudget(n_s, N_FEATURES, DEGREE, x_spec, w_spec).magnitude_bound()
        single = [choose_field_size(B, warn_native=False).q]
        multi = crt_primes(2 * B + 1)
        for backend, primes in (("single", single), ("crt", multi)):
            if backend == "crt" and len(multi) == 1:
                continue        # fits one native prime: CRT would be the same computation
            m, h, eps_c = theorem1_m(n, math.prod(primes), x_spec)
            # 2 repeats (1 cold + 1 warm): the large-lambda cells dominate the sweep's runtime
            cells.append(_cell("C", primes, m, n_s=n_s, mode="quant", f_X=f_X, f_w=f_w,
                               backend=backend, h=h, eps_c=eps_c, repeats=2))
    return cells


def build_D(fast: bool):
    bits = (30, 127) if fast else BITS_A
    return [dict(exp="D", primes=[str(prime_for_bits(b))], bits_label=b, kind="construct")
            for b in bits]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def predicted_mb(spec: dict) -> float:
    """
    Predicted peak memory of a cell, calibrated against measured peaks.

    Each of the L instances keeps its dense lambda x lambda interpolation matrix (8 bytes per
    entry in either mode: native int64, or an object pointer -- ~99.5% of entries are 0, which
    CPython caches). Building M needs temporaries worth ~2.5 matrices, but only one instance
    builds at a time. Plus ~0.7 GB of interpreter, galois and numba. Measured peaks at
    lambda = 11,781-14,365 fall within 0.75-1.0x of this. (A first version charged 3 matrices
    per instance, which skipped a CRT cell that fits.)

    :param spec: Cell spec
    :return: Predicted peak MB
    """
    lam = math.comb(spec["m"] + spec["d"], spec["d"])
    return 700 + 8 * lam * lam * (len(spec["primes"]) + 2.5) / 2 ** 20


def run_cell(spec: dict, timeout: float, mem_cap_mb: float) -> dict:
    """
    run_cell_subprocess plus the conditions it ran under: power source, and any sleep/wake
    transition during the cell. A cell that slept is marked 'slept' so --resume re-runs it.

    :param spec: Cell spec
    :param timeout: Seconds before the cell is recorded as a timeout
    :param mem_cap_mb: Memory skip threshold
    :return: Result row
    """
    power, start = power_source(), time.time()
    row = run_cell_subprocess(spec, timeout, mem_cap_mb)
    events = sleep_events(start, time.time())
    row.update(power=power, sleep_events=events)
    if events and row["status"] == "ok":
        row["status"] = "slept"
    return row


def run_cell_subprocess(spec: dict, timeout: float, mem_cap_mb: float) -> dict:
    """
    Runs one cell in a fresh interpreter with a wall-clock cap, so a slow or memory-hungry
    cell becomes a recorded result instead of a hung sweep.

    :param spec: Cell spec
    :param timeout: Seconds before the cell is recorded as a timeout
    :param mem_cap_mb: Cells predicted to need more are skipped, not run
    :return: Result row
    """
    if spec.get("kind") != "construct" and predicted_mb(spec) > mem_cap_mb:
        return dict(spec, status="skipped:memory", predicted_mb=predicted_mb(spec))
    # The cell gets its own process group and the whole group is killed on timeout. On macOS
    # a venv's python is a launcher that starts the real interpreter as a CHILD, so killing
    # only the direct child (what subprocess.run(timeout=) does) leaves the interpreter
    # running and holding the pipes open -- the cap then never fires. Found the hard way: a
    # 2^521 construction cell ran 31 minutes past a 2-minute cap.
    proc = subprocess.Popen([sys.executable, __file__, "--cell", json.dumps(spec)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return dict(spec, status=f"timeout:{int(timeout)}s")
    lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(spec, status="error", error=stderr.strip().splitlines()[-1:])
    return json.loads(lines[-1])


def power_source() -> str:
    """
    :return: 'AC' or 'battery' on macOS, None elsewhere. Recorded per cell: a run that mixes
        the two is not a controlled comparison.
    """
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC" if "AC Power" in out else "battery"


_PMSET_EVENT = __import__("re").compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) [+-]\d{4} (Sleep|DarkWake|Wake) ")


def sleep_events(start: float, end: float) -> int:
    """
    Counts system sleep/wake transitions between two wall-clock times, from the macOS power
    log. A cell that straddles one is not a clean timing even if its clock excludes the sleep:
    the maintenance wake-ups between sleeps run throttled. Found when a lid close mid-sweep
    produced a 38.7 s cell whose neighbours predicted ~25 s.

    :param start: Cell start, time.time()
    :param end: Cell end, time.time()
    :return: Number of transitions; 0 off macOS, where this is not checked
    """
    if sys.platform != "darwin":
        return 0
    from datetime import datetime
    log = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True).stdout
    count = 0
    for line in log.splitlines()[-4000:]:
        match = _PMSET_EVENT.match(line)
        if match:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            count += start - 1 <= ts <= end + 1
    return count


def machine_info() -> dict:
    """:return: Hardware and library versions, saved next to every result file"""
    info = dict(platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), numpy=np.__version__,
                galois=galois.__version__, cpu_count=os.cpu_count())
    if sys.platform == "darwin":
        for key, name in (("cpu", "machdep.cpu.brand_string"), ("mem_bytes", "hw.memsize")):
            info[key] = subprocess.run(["sysctl", "-n", name], capture_output=True,
                                       text=True).stdout.strip()
    return info


def write_rows(rows: list, path: Path):
    """
    Writes rows to CSV with the union of their keys as columns.

    :param rows: Result dicts
    :param path: Output CSV path
    """
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                             for k, v in r.items()})


_KEY_FIELDS = ("exp", "kind", "primes", "m", "n_s", "mode", "f_X", "f_w", "backend",
               "bits_label")


def cell_key(row: dict) -> str:
    """
    Identity of a cell, comparable between a fresh spec and a row read back from CSV (where
    every value is a string and lists are JSON).

    :param row: Spec or result row
    :return: Key string
    """
    parts = []
    for k in _KEY_FIELDS:
        v = row.get(k)
        parts.append(json.dumps(v) if isinstance(v, list) else "" if v is None else str(v))
    return "|".join(parts)


def main():
    parser = argparse.ArgumentParser(description="ICC q / parameter feasibility sweep")
    parser.add_argument("--exp", default="all", help="T, A, B, C, D or all (comma separated)")
    parser.add_argument("--fast", action="store_true", help="tiny grids, for smoke testing")
    parser.add_argument("--jobs", type=int, default=1,
                        help="cells run concurrently. Keep 1 for any timing that will be "
                             "reported: 4 concurrent cells on an M2 Pro inflated large-q "
                             "timings by up to 3.5x (docs/Q_SWEEP.md section 2). Use >1 "
                             "only for smoke runs or correctness checks.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per cell")
    parser.add_argument("--mem-cap-mb", type=float, default=10240.0,
                        help="skip cells predicted to need more; times --jobs must fit in "
                             "RAM (default: 10 GB, for one cell at a time on 16 GB)")
    parser.add_argument("--out", default=str(RESULTS), help="output directory")
    parser.add_argument("--resume", action="store_true",
                        help="keep finished cells from the existing CSV; run only the rest "
                             "(and any that slept, errored or came out wrong)")
    parser.add_argument("--rerun", action="append", default=[], metavar="KEY=VALUE",
                        help="force cells whose spec field KEY equals VALUE to be measured "
                             "again, e.g. --rerun bits_label=127; implies --resume")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:   # child process: one cell, one JSON line
        spec = json.loads(args.cell)
        row = construct_cell(spec) if spec.get("kind") == "construct" else measure_cell(spec)
        print(json.dumps(row))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_fast" if args.fast else ""
    exps = "TABCD" if args.exp == "all" else args.exp.replace(",", "").upper()
    (out / f"q_sweep_machine{tag}.json").write_text(json.dumps(machine_info(), indent=2))

    if "T" in exps:
        rows = theory_rows()
        write_rows(rows, out / f"q_sweep_T{tag}.csv")
        print(f"T: {len(rows)} analytic rows")

    builders = dict(A=build_A, B=build_B, C=build_C, D=build_D)
    for exp in [e for e in exps if e in builders]:
        cells = builders[exp](args.fast)
        timeout = 120.0 if exp == "D" else (120.0 if args.fast else args.timeout)
        path = out / f"q_sweep_{exp}{tag}.csv"
        done = {}
        if (args.resume or args.rerun) and path.exists():
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    # memory skips depend on --mem-cap-mb, so they are re-attempted too
                    if r["status"] not in ("slept", "error", "WRONG", "skipped:memory"):
                        done[cell_key(r)] = r
        forced = [kv.split("=", 1) for kv in args.rerun]
        todo = [c for c in cells if cell_key(c) not in done
                or any(str(c.get(k)) == v for k, v in forced)]
        for c in todo:
            done.pop(cell_key(c), None)
        print(f"{exp}: {len(todo)} of {len(cells)} cells to run, {args.jobs} at a time, "
              f"cap {timeout:.0f}s", flush=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_cell, c, timeout, args.mem_cap_mb) for c in todo]
            for fut in futures:
                r = fut.result()
                done[cell_key(r)] = r
                # Written after EVERY cell, in grid order, so an interrupted run loses at
                # most the cell in flight (a battery death would otherwise lose them all)
                write_rows([done[cell_key(c)] for c in cells if cell_key(c) in done], path)
                summary = (f"step {r['t_step']:.2f}s setup {r['t_setup']:.2f}s"
                           if "t_step" in r else
                           f"construct {r.get('t_construct_verified', float('nan')):.2f}s"
                           if "t_construct_verified" in r else "")
                print(f"  log2q={r.get('log2_q', float('nan')):6.1f} m={r.get('m', '-'):>4} "
                      f"lam={r.get('lam', '-'):>6} L={r.get('L', 1)} "
                      f"{r['status']:<16} {summary}", flush=True)
        rows = [done[cell_key(c)] for c in cells if cell_key(c) in done]
        bad = [r for r in rows if r["status"] in ("WRONG", "error", "slept")]
        print(f"{exp}: done in {time.perf_counter() - t0:.0f}s; "
              f"{len(bad)} wrong/error/slept cells", flush=True)
        if bad:
            for r in bad:
                print("   ", r.get("status"), r.get("error"), flush=True)


if __name__ == "__main__":
    main()
