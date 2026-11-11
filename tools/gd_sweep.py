"""
File: gd_sweep.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Phase 1 of docs/GD_LINEAR_REGRESSION.md (section 10): least-squares linear
    regression trained by live gradient descent THROUGH THE SCHEME on a grid of synthetic
    distributions, against the same descent unencoded, measuring convergence in steps and
    in wall-clock time.

    Every run:
      1. samples a dataset from a known source and uploads it through the real scheme
         (tools/gd_setup.upload: q from the public format, m from the exact entropy);
      2. trains through the scheme for the design's T steps (icc/training.py);
      3. replays the same loop with the exact integer gradient in plaintext and requires the
         two trajectories to be IDENTICAL -- codes and floats, every step -- else the run is
         marked WRONG;
      4. runs float GD on the dequantised codes and on the real-valued data (the unencoded
         baseline), same eta, T, start and clip;
      5. records suboptimality curves, steps to tolerance, plateaus, timings, and context
         (sample and population kappa, saturation, divergence, m, lambda, eps_c).

    Sweeps (section 10.6; cells duplicated across sweeps are run once, under S1):
      S1 shape          9 families at (n_s, P) = (25, 3)
      S2 conditioning   correlated Gaussian, rho in {0.5, 0.9, 0.95}   (rho = 0 is S1 gauss)
      S3 size           (50,1), (10,9), (6,10), (10,3) x {gauss, exponential, spiked}
      S4 noise          SNR in {10, 1, 0.1} x {gauss, laplace} noise    (SNR 1 gauss is S1)

    Outputs, written after EVERY run so an interruption loses at most the run in flight:
      results/gd_sweep.csv              one row per run
      results/gd_traj/<run>.csv         per-step curves of that run
      results/gd_sweep_machine.json     hardware and library versions

Usage (reported timings: one run at a time, on AC power, lid open, under caffeinate):
    caffeinate -dimsu .venv/bin/python tools/gd_sweep.py --dry-run     # grid, T, m, lambda, hours
    caffeinate -dimsu .venv/bin/python tools/gd_sweep.py                # everything, ~a day
    caffeinate -dimsu .venv/bin/python tools/gd_sweep.py --resume       # continue
    .venv/bin/python tools/gd_sweep.py --fast                           # smoke run, minutes
    .venv/bin/python tools/gd_sweep.py --sweep S1 --seeds 1             # one sweep, one seed
    caffeinate -dimsu .venv/bin/python tools/gd_sweep.py --order coverage --resume
        # first seed of every cell (cheapest first), then the remaining seeds cheapest-first;
        # an interruption at any point leaves every cell covered before any gets a 2nd seed
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from training import run_gd, train_linear_regression

from gd_reference import (augment, condition_number, float_gd, integer_gradient_fn,
                          least_squares_optimum, plateau, steps_to_tolerance, suboptimality)
from gd_setup import design_schedule, plan_context, upload
from q_sweep import machine_info, power_source, sleep_events, write_rows
from synthetic_source import FAMILIES, SourceSpec

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOLS = (1e-1, 1e-2, 1e-3)
SEEDS = 5
BASE = (25, 3)
SIZES = ((50, 1), (10, 9), (6, 10), (10, 3))
SIZE_FAMILIES = ("gauss", "exponential", "spiked")
RHOS = (0.5, 0.9, 0.95)
NOISE_CELLS = [(snr, noise) for noise in ("gauss", "laplace") for snr in (10.0, 1.0, 0.1)
               if not (snr == 1.0 and noise == "gauss")]

# Per-step cost model for --dry-run only: measured 1.71 s/step at lambda = 3081 with 3
# queries of 100 data symbols (results/q_sweep_C.csv), scaling as lambda^1.52 (docs/Q_SWEEP.md
# section 4) and linearly in queries x terms per query. An estimate, not a measurement.
_STEP_REF = (1.71, 3081, 3 * 100)


def build_grid(fast: bool, seeds: int) -> list:
    """
    :param fast: Smoke grid: the base size shrinks to (10, 3), S3 keeps two sizes, one seed
    :param seeds: Seeds per cell
    :return: List of run descriptions (dicts), in execution order
    """
    base = (10, 3) if fast else BASE
    sizes = ((50, 1), (6, 10)) if fast else SIZES
    cells = []
    for fam in FAMILIES:
        cells.append(dict(sweep="S1", family=fam, n_s=base[0], P=base[1]))
    for rho in RHOS:
        cells.append(dict(sweep="S2", family="gauss", rho=rho, n_s=base[0], P=base[1]))
    for fam in SIZE_FAMILIES:
        for n_s, P in sizes:
            cells.append(dict(sweep="S3", family=fam, n_s=n_s, P=P))
    for snr, noise in NOISE_CELLS:
        cells.append(dict(sweep="S4", family="gauss", snr=snr, noise=noise,
                          n_s=base[0], P=base[1]))
    runs = []
    for cell in cells:
        for seed in range(1 if fast else seeds):
            run = dict(rho=0.0, snr=1.0, noise="gauss")
            run.update(cell)
            run["seed"] = seed
            runs.append(run)
    return runs


def source_of(run: dict) -> SourceSpec:
    return SourceSpec(family=run["family"], n_samples=int(run["n_s"]),
                      n_features=int(run["P"]), rho=float(run["rho"]),
                      snr=float(run["snr"]), noise=run["noise"])


def run_key(run: dict) -> str:
    """:return: Stable identity of a run, comparable with a row read back from CSV"""
    return (f"{run['sweep']}_{run['family']}_rho{float(run['rho']):g}_ns{int(run['n_s'])}"
            f"_P{int(run['P'])}_snr{float(run['snr']):g}_{run['noise']}_s{int(run['seed'])}")


def _seed_entropy(run: dict) -> list:
    """
    SeedSequence entropy for a run: the seed index plus a hash-free encoding of the cell, so
    each cell's datasets are distinct and every run is reproducible on its own.
    """
    fam = list(FAMILIES).index(run["family"])
    return [2026, int(run["seed"]), fam, int(round(float(run["rho"]) * 100)),
            int(run["n_s"]), int(run["P"]), int(round(float(run["snr"]) * 100)),
            0 if run["noise"] == "gauss" else 1]


def estimate_step_seconds(lam: int, n_s: int, P: int) -> float:
    """:return: Estimated seconds per scheme step (model above; --dry-run only)"""
    t_ref, lam_ref, work_ref = _STEP_REF
    work = (P + 1) * n_s * (P + 1)
    return t_ref * (lam / lam_ref) ** 1.52 * work / work_ref


def measure_run(run: dict, steps_override=None, traj_dir: Path = None) -> dict:
    """
    One full run: upload, train through the scheme, verify bit-exactness against plaintext
    integer GD, run the float baselines, compute the metrics.

    :param run: Run description from build_grid
    :param steps_override: Cap T (smoke runs)
    :param traj_dir: Where to write the per-step curves
    :return: Result row
    """
    spec = source_of(run)
    schedule, design = design_schedule(spec, steps=steps_override)
    n_s, P = spec.n_samples, spec.n_features
    row = dict(run, key=run_key(run), n=spec.n, status="ok", T=schedule.steps,
               T_design=design["T_design"], eta=schedule.eta, kappa_pop=design["kappa_pop"],
               spread=design["spread"], power=power_source())
    wall0 = time.time()

    up = upload(spec, _seed_entropy(run))
    plan, data = up.plan, up.data
    row.update(q=plan["q"], log2_q=plan["log2_q"], native=plan["native"], m=plan["m"],
               lam=plan["lam"], h=plan["h"], H_p_X=plan["H_p_X"], max_H=plan["max_H"],
               m_source=plan["m_source"], max_R_exact=plan["entropy"].max_R_exact,
               eps_c=plan["eps_c"], eps_c_alt=plan["eps_c_alt"], eps_c_bits=plan["eps_c_bits"],
               t_encode=up.timings["encode"], t_store=up.timings["store"],
               t_build=up.timings["build"], t_setup=up.timings["setup"],
               t_construct_first=up.timings["construct_first"])

    # Encoded: through the scheme
    enc = train_linear_regression(up.client, up.server, schedule, n_s, P)
    # Plaintext integer GD: the same loop, the exact gradient from the codes
    ref = run_gd(schedule, n_s, P, integer_gradient_fn(data["X_codes"], data["y_codes"],
                                                       schedule))
    bit_exact = (np.array_equal(enc.w_codes, ref.w_codes) and np.array_equal(enc.w, ref.w)
                 and enc.grad == ref.grad)
    row["bit_exact"] = bit_exact
    if not bit_exact:
        row["status"] = "WRONG"

    # Unencoded baselines
    X, y = data["X_real"], data["y_real"]
    Xd, yd = data["X_codes"] / 2.0 ** spec.f_X, data["y_codes"] / 2.0 ** spec.f_X
    w_float, t_float = float_gd(X, y, schedule)
    w_fdeq, _ = float_gd(Xd, yd, schedule)

    # Metrics, all on the REAL-valued data and its own optimum, the same reference for all
    w_opt = least_squares_optimum(X, y, True)
    curves = dict(scheme=suboptimality(X, y, enc.w, True, w_opt),
                  float=suboptimality(X, y, w_float, True, w_opt),
                  float_deq=suboptimality(X, y, w_fdeq, True, w_opt))
    # The scheme against the optimum of the data it actually holds (the dequantised codes):
    # separates the weight-grid floor from the data-quantisation bias
    curves["scheme_vs_deq_opt"] = suboptimality(Xd, yd, enc.w, True)

    step_scheme = enc.t_eval + enc.t_decode
    cum_scheme = np.concatenate([[0.0], np.cumsum(step_scheme)])
    cum_float = np.concatenate([[0.0], np.cumsum(t_float)])
    for name in ("scheme", "float", "float_deq", "scheme_vs_deq_opt"):
        row[f"plateau_{name}"] = plateau(curves[name])
    for tol in TOLS:
        for name in ("scheme", "float", "float_deq"):
            tau = steps_to_tolerance(curves[name], tol)
            row[f"tau_{name}_{tol:g}"] = tau
        tau_s, tau_f = row[f"tau_scheme_{tol:g}"], row[f"tau_float_{tol:g}"]
        row[f"wall_scheme_{tol:g}"] = (None if tau_s is None
                                       else up.timings["setup"] + float(cum_scheme[tau_s]))
        row[f"wall_float_{tol:g}"] = None if tau_f is None else float(cum_float[tau_f])

    # Context
    Xa = augment(X, True)
    kappa_s, lam_max_s, lam_min_s = condition_number(Xa.T @ Xa / n_s)
    w_true, b_true, sigma_e = spec.weights
    M = spec.x_spec.max_int
    row.update(
        t_step_scheme=float(np.median(step_scheme[1:] if len(step_scheme) > 1 else step_scheme)),
        t_step_cold=float(step_scheme[0]),
        t_eval=float(np.median(enc.t_eval)), t_decode=float(np.median(enc.t_decode)),
        t_step_float=float(np.median(t_float)),
        t_train_scheme=float(cum_scheme[-1]), t_train_float=float(cum_float[-1]),
        kappa_sample=kappa_s, lam_max_sample=lam_max_s, lam_min_sample=lam_min_s,
        diverged=bool(schedule.eta * lam_max_s >= 2.0),
        sat_features=float(np.mean(np.abs(data["X_codes"]) == M)),
        sat_labels=float(np.mean(np.abs(data["y_codes"]) == M)),
        w_clip_steps=float(np.mean(np.any(np.abs(enc.w[1:, :P]) >= schedule.w_spec.clip,
                                          axis=1))),
        err_w_true=float(np.linalg.norm(enc.w[-1] - np.append(w_true, b_true))),
        err_w_opt=float(np.linalg.norm(enc.w[-1] - w_opt)),
        err_float_w_opt=float(np.linalg.norm(w_float[-1] - w_opt)),
        sigma_e=sigma_e, b_true=b_true, w_true=json.dumps([float(v) for v in w_true]),
        sleep_events=sleep_events(wall0, time.time()), wall_run=time.time() - wall0)
    if row["sleep_events"]:
        row["status"] = "slept" if row["status"] == "ok" else row["status"]

    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)
        with open(traj_dir / f"{row['key']}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["step", "scheme", "float", "float_deq", "scheme_vs_deq_opt",
                         "t_step_scheme"])
            for t in range(len(curves["scheme"])):
                wr.writerow([t] + [f"{curves[k][t]:.6e}" for k in
                                   ("scheme", "float", "float_deq", "scheme_vs_deq_opt")]
                            + [f"{step_scheme[t]:.6f}" if t < len(step_scheme) else ""])
    return row


def estimate_run_seconds(run: dict, cache: dict = None) -> float:
    """:return: Estimated seconds for one run (design T x modelled step cost + setup)"""
    cache = {} if cache is None else cache
    cell = run_key(dict(run, seed=0))
    if cell not in cache:
        spec = source_of(run)
        sch, _ = design_schedule(spec)
        plan = plan_context(spec)
        cache[cell] = sch.steps * estimate_step_seconds(plan["lam"], spec.n_samples,
                                                         spec.n_features) + 10
    return cache[cell]


def coverage_order(runs: list) -> list:
    """
    Execution order for a sweep that may be cut short. Seed 0 of every cell runs first,
    cheapest cell first; later seeds follow by (seed + 1) x estimated cost, so cheap cells
    collect seeds while expensive ones wait. Only the order changes: every run keeps its own
    seed, so results are identical to the grid order.

    :param runs: Runs from build_grid
    :return: The same runs, reordered
    """
    cache = {}
    return sorted(runs, key=lambda r: (int(r["seed"]) > 0,
                                       (int(r["seed"]) + 1) * estimate_run_seconds(r, cache)))


def dry_run(runs: list):
    """Prints the grid with T, m, lambda and an estimated duration, without uploading."""
    print(f"{'sweep':5} {'family':12} {'rho':>4} {'(n_s,P)':>8} {'snr':>5} {'noise':7} "
          f"{'T':>5} {'eta':>6} {'m':>4} {'lambda':>6} {'eps_c bits':>10} {'est min':>8}")
    total, seen = 0.0, {}
    for run in runs:
        cell = run_key(dict(run, seed=0))
        if cell not in seen:
            spec = source_of(run)
            sch, _ = design_schedule(spec)
            plan = plan_context(spec)
            secs = sch.steps * estimate_step_seconds(plan["lam"], spec.n_samples,
                                                      spec.n_features) + 10
            seen[cell] = (sch, plan, secs)
            print(f"{run['sweep']:5} {run['family']:12} {float(run['rho']):>4g} "
                  f"{'(%d,%d)' % (run['n_s'], run['P']):>8} {float(run['snr']):>5g} "
                  f"{run['noise']:7} {sch.steps:>5} {sch.eta:>6.3f} {plan['m']:>4} "
                  f"{plan['lam']:>6} {plan['eps_c_bits']:>10.4f} {secs / 60:>8.1f}")
        total += seen[cell][2]
    print(f"\n{len(runs)} runs in {len(seen)} cells; estimated {total / 3600:.1f} h "
          f"(model: 1.71 s/step at lambda 3081, x lambda^1.52, x query work; plus ~10 s "
          f"setup per run). Run --fast first to check the model on this machine.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 GD convergence sweep")
    parser.add_argument("--sweep", default="all", help="S1, S2, S3, S4 or all (comma separated)")
    parser.add_argument("--seeds", type=int, default=SEEDS, help="seeds per cell")
    parser.add_argument("--fast", action="store_true",
                        help="smoke grid: (10,3) base, two S3 sizes, 1 seed, T capped at 20")
    parser.add_argument("--resume", action="store_true",
                        help="keep finished runs from the existing CSV; rerun the rest and "
                             "any that slept, errored or came out WRONG")
    parser.add_argument("--dry-run", action="store_true", help="print the grid and estimate")
    parser.add_argument("--out", default=str(RESULTS), help="output directory")
    parser.add_argument("--order", choices=("grid", "coverage"), default="grid",
                        help="grid: cell by cell; coverage: one seed of every cell first, "
                             "cheapest first (for a run that may be cut short)")
    args = parser.parse_args()

    runs = build_grid(args.fast, args.seeds)
    if args.sweep != "all":
        wanted = {s.strip().upper() for s in args.sweep.split(",")}
        runs = [r for r in runs if r["sweep"] in wanted]
    if args.dry_run:
        dry_run(runs)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_fast" if args.fast else ""
    path, traj_dir = out / f"gd_sweep{tag}.csv", out / f"gd_traj{tag}"
    (out / f"gd_sweep_machine{tag}.json").write_text(json.dumps(machine_info(), indent=2))

    done = {}
    if args.resume and path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "ok":
                    done[r["key"]] = r
    todo = [r for r in runs if run_key(r) not in done]
    if args.order == "coverage":
        todo = coverage_order(todo)
    print(f"{len(todo)} of {len(runs)} runs to do; power: {power_source()}", flush=True)
    if power_source() == "battery":
        print("  WARNING: on battery. Timings will not be comparable; plug in for reported "
              "runs.", flush=True)

    t0 = time.time()
    for i, run in enumerate(todo, 1):
        try:
            row = measure_run(run, steps_override=20 if args.fast else None,
                              traj_dir=traj_dir)
        except Exception as exc:   # record and continue: one bad cell must not stop a night
            row = dict(run, key=run_key(run), status="error", error=repr(exc),
                       trace=traceback.format_exc(limit=3))
        done[row["key"]] = row
        write_rows([done[run_key(r)] for r in runs if run_key(r) in done], path)
        tau = row.get("tau_scheme_0.01")
        print(f"[{i}/{len(todo)}] {row['key']:52} {row['status']:6} "
              f"T={row.get('T', '-')} m={row.get('m', '-')} lam={row.get('lam', '-')} "
              f"tau(1e-2) scheme={tau} float={row.get('tau_float_0.01')} "
              f"step={row.get('t_step_scheme', float('nan')):.2f}s "
              f"elapsed={(time.time() - t0) / 60:.1f} min", flush=True)

    rows = [done[run_key(r)] for r in runs if run_key(r) in done]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"done: {len(rows)} runs, {len(bad)} not ok", flush=True)
    for r in bad:
        print("   ", r["key"], r["status"], r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
