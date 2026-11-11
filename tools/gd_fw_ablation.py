"""
File: gd_fw_ablation.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: How the encoded run's accuracy floor depends on the weight format f_w.

    The Phase 1 sweep shows the encoded run levelling off at a suboptimality of about
    1e-3 to 1e-2 while unencoded float GD goes on to machine precision. The sweep also checks,
    run by run, that the scheme's trajectory is BIT-IDENTICAL to plaintext integer GD on the
    same codes (`bit_exact` column). So the floor is a property of the fixed-point format, and
    it can be mapped in plaintext: this script reruns every finished sweep run's own dataset
    (same SeedSequence, same data stream) through plaintext integer GD with
    f_w in {4, 6, 8, 10}, nothing else changed.

    No scheme runs: the f_w = 4 rows reproduce the sweep's encoded plateau exactly (checked
    below, `matches_sweep`), which is what licenses reading the other rows as what the scheme
    would do. What a larger f_w costs the scheme is reported alongside from the field-size
    chain alone (q, whether it stays native, m, lambda) -- no timing is claimed for it.

    Output: results/gd_fw_ablation.csv (one row per run x f_w)

Usage (after the sweep; plaintext and light, but keep it off the timed sweep):
    .venv/bin/python tools/gd_fw_ablation.py
    .venv/bin/python tools/plot_gd.py          # adds gd_fig7_fw_floor
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from training import run_gd

from gd_reference import (integer_gradient_fn, least_squares_optimum, plateau,
                          steps_to_tolerance, suboptimality)
from gd_setup import design_schedule, plan_context
from gd_sweep import RESULTS, TOLS, _seed_entropy, source_of
from q_sweep import write_rows

F_WS = (4, 6, 8, 10)


def _num(r: dict, k: str):
    return float(r[k]) if r.get(k) not in (None, "", "None") else None


def ablate_run(run: dict, plans: dict) -> list:
    """
    :param run: A finished sweep row (strings, as read from CSV)
    :param plans: Cache of plan_context per (cell, f_w)
    :return: One row per f_w
    """
    spec = source_of(run)
    # Same data stream as gd_setup.upload: SeedSequence(entropy).spawn(3)[0]
    data_ss = np.random.SeedSequence(_seed_entropy(run)).spawn(3)[0]
    data = spec.sample(np.random.default_rng(data_ss))
    X, y = data["X_real"], data["y_real"]
    w_opt = least_squares_optimum(X, y, True)
    cell = run["key"].rsplit("_s", 1)[0]
    out = []
    for f_w in F_WS:
        schedule, _ = design_schedule(spec, f_w=f_w)
        t0 = time.perf_counter()
        traj = run_gd(schedule, spec.n_samples, spec.n_features,
                      integer_gradient_fn(data["X_codes"], data["y_codes"], schedule))
        curve = suboptimality(X, y, traj.w, True, w_opt)
        if (cell, f_w) not in plans:
            plans[(cell, f_w)] = plan_context(spec, f_w=f_w)
        plan = plans[(cell, f_w)]
        row = dict(key=run["key"], sweep=run["sweep"], family=run["family"], rho=run["rho"],
                   n_s=run["n_s"], P=run["P"], snr=run["snr"], noise=run["noise"],
                   seed=run["seed"], f_w=f_w, T=schedule.steps, eta=schedule.eta,
                   plateau=plateau(curve), log2_q=plan["log2_q"], native=plan["native"],
                   m=plan["m"], lam=plan["lam"], t_plain=time.perf_counter() - t0)
        for tol in TOLS:
            row[f"tau_{tol:g}"] = steps_to_tolerance(curve, tol)
        if f_w == 4:
            ref = _num(run, "plateau_scheme")
            row["matches_sweep"] = ref is not None and np.isclose(row["plateau"], ref,
                                                                   rtol=1e-9, atol=0)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="f_w ablation of the encoded floor")
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    res = Path(args.results)
    with open(res / "gd_sweep.csv") as fh:
        runs = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    print(f"{len(runs)} finished sweep runs x f_w in {F_WS}", flush=True)
    rows, plans, t0 = [], {}, time.time()
    for i, run in enumerate(runs, 1):
        rows.extend(ablate_run(run, plans))
        write_rows(rows, res / "gd_fw_ablation.csv")
        m4 = rows[-len(F_WS)]["matches_sweep"]
        print(f"[{i}/{len(runs)}] {run['key']:52} f_w=4 matches sweep: {m4}  "
              + "  ".join(f"f_w={r['f_w']}: {r['plateau']:.1e}" for r in rows[-len(F_WS):])
              + f"  ({time.time() - t0:.0f} s)", flush=True)
    bad = [r["key"] for r in rows if r["f_w"] == 4 and not r["matches_sweep"]]
    print(f"done; {len(bad)} runs where f_w = 4 does NOT reproduce the sweep"
          + (": " + ", ".join(bad) if bad else ""), flush=True)


if __name__ == "__main__":
    main()
