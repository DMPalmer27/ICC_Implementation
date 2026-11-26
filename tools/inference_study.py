"""
File: inference_study.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: The inference study behind the paper (docs/INFERENCE.md section 5). Every
    scored row goes through the real scheme: upload, one degree-1 query per (row, output),
    one decode, and the decoded integer is checked against the plaintext integer oracle.

    Experiments (rows of results/inference_<exp>.csv):
      S  synthetic breadth: feature family x task x P, at the default format
      F  format: f_X x f_w on a few sources, every cell through the scheme
      C  cost vs stored rows N: measured d = 1, and the d = 2 training-query plan beside it
      R  real data: diabetes (linear) and wine (softmax), over a format grid

    Seeds: run seed -> SeedSequence -> independent 'train', 'store', 'labels', 'scheme'
    streams, so the stored rows never overlap the training rows and changing one stream
    never moves another.

Usage:
    .venv/bin/python tools/inference_study.py --exp S [--fast] [--resume]
    .venv/bin/python tools/inference_study.py --exp all
"""

import _path  # noqa: F401  -- puts icc/ on sys.path; must precede the local imports

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from inference import descale, predict_scores
from quantize import FixedPointSpec

from inference_setup import (decisions, fit_model, integer_scores, plan_inference,
                             quantise_model, score_error_bound, task_metrics, upload)
from inference_source import TASKS, FeatureSource, TaskDesign

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N_TRAIN = 2000          # synthetic training rows; independent of every stored row
DEFAULT_F_X, DEFAULT_F_W, CLIP_X = 8, 8, 4.0


def evaluate(task: str, X_store: np.ndarray, y_store: np.ndarray, X_codes: np.ndarray,
             x_spec: FixedPointSpec, W: np.ndarray, b: np.ndarray, f_w: int, entropy,
             seed, y_scale: float = 1.0) -> dict:
    """
    One inference run on already-split data: quantise the model, plan, upload the stored
    rows, score every row through the scheme, and measure.

    :param task: 'linear', 'logistic' or 'softmax'
    :param X_store: REAL stored features (N, P), used only for the float reference and bound
    :param y_store: Labels of the stored rows, held by the client
    :param X_codes: Stored integer codes (N, P) -- the only thing uploaded
    :param x_spec: Storage format
    :param W: Float weights (K, P), fitted on training rows
    :param b: Float intercepts (K,)
    :param f_w: Weight fractional bits
    :param entropy: EntropyResult of the stored vector
    :param seed: Entropy for the 'scheme' stream (G, key)
    :param y_scale: Linear task: RMSE back to the label's units
    :return: dict of measurements (one CSV row, without the caller's identifiers)
    """
    N, P = X_codes.shape
    model = quantise_model(W, b, f_w)
    plan = plan_inference(N, P, x_spec, model["w_spec"], model["b_spec"], entropy)
    client, server, t = upload(X_codes, plan, x_spec, seed)
    Z_int, t_q = predict_scores(client, server, N, P, model["W_codes"], model["b_codes"],
                                x_spec.f)
    oracle = integer_scores(X_codes, model["W_codes"], model["b_codes"], x_spec.f)
    exact = int(np.sum(Z_int == oracle))

    Z_scheme = descale(Z_int, x_spec.f, f_w)
    Z_float = X_store @ W.T + b
    err = np.abs(Z_scheme - Z_float)
    clipped_rows = np.any(np.abs(X_store) > x_spec.clip, axis=1)
    bound = score_error_bound(X_store, x_spec, W, f_w)
    m_float = task_metrics(task, y_store, Z_float, y_scale)
    m_scheme = task_metrics(task, y_store, Z_scheme, y_scale)
    dec_f, dec_s = decisions(task, Z_float), decisions(task, Z_scheme)
    flips = int(np.sum(dec_f != dec_s)) if dec_f is not None else 0
    if dec_f is not None:
        # a decision can only flip where the float score is within the bound of a tie
        if task == "logistic":
            margin = np.abs(Z_float[:, 0])
            bnd = bound[:, 0]
        else:
            top = np.sort(Z_float, axis=1)
            margin = top[:, -1] - top[:, -2]
            bnd = 2 * bound.max(axis=1)
        flip_possible = int(np.sum(margin <= bnd))
    else:
        flip_possible = 0

    row = dict(
        N=N, P=P, n=N * P, K=W.shape[0], f_X=x_spec.f, clip_X=x_spec.clip, f_w=f_w,
        A_w=model["w_spec"].clip, A_b=model["b_spec"].clip,
        q=plan["q"], log2_q=round(plan["log2_q"], 3), native=plan["native"], B=plan["B"],
        method=plan["method"], H_p_X=round(plan["H_p_X"], 3), max_H=round(plan["max_H"], 4),
        h=round(plan["h"], 4), m=plan["m"], lam=plan["lam"], eps_c=plan["eps_c"],
        eps_c_alt=plan["eps_c_alt"], eps_c_bits=plan["eps_c_bits"],
        queries=N * W.shape[0], exact=exact, all_exact=exact == N * W.shape[0],
        clipped_values=int(np.sum(np.abs(X_store) > x_spec.clip)),
        rows_clipped=int(np.sum(clipped_rows)),
        err_max=float(err.max()),
        err_max_unclipped=float(err[~clipped_rows].max()) if np.any(~clipped_rows) else float("nan"), err_mean=float(err.mean()), bound_max=float(bound.max()),
        bound_mean=float(bound.mean()), bound_violations=int(np.sum(err > bound * (1 + 1e-12))),
        flips=flips, flip_possible=flip_possible,
        t_encode=t["encode"], t_store=t["store"], t_build=t["build"],
        t_evaluate=t_q["evaluate"], t_decode=t_q["decode"],
    )
    for k, v in m_float.items():
        row[f"float_{k}"] = v
    for k, v in m_scheme.items():
        row[f"scheme_{k}"] = v
    return row


def run_synthetic(family: str, rho: float, task: str, P: int, N: int, f_X: int, f_w: int,
                  seed: int) -> dict:
    """
    One synthetic run: train on N_TRAIN fresh rows, store N fresh rows, score through the scheme.

    :return: CSV row
    """
    source = FeatureSource(family, P, rho, f_X, CLIP_X)
    design = TaskDesign(task, source)
    ss = np.random.SeedSequence([seed, P, N, f_X, f_w, TASKS.index(task), int(rho * 100),
                                 sum(map(ord, family))])
    train_ss, store_ss, label_ss, scheme_ss = ss.spawn(4)
    lab = np.random.default_rng(label_ss)
    X_train = source.sample(np.random.default_rng(train_ss), N_TRAIN)
    y_train = design.labels(X_train, lab)
    W, b = fit_model(task, X_train, y_train)
    X_store = source.sample(np.random.default_rng(store_ss), N)
    y_store = design.labels(X_store, lab)
    X_codes = source.quantise(X_store)
    row = dict(source=source.name, family=family, rho=rho, task=task, seed=seed)
    row.update(evaluate(task, X_store, y_store, X_codes, source.x_spec, W, b, f_w,
                        source.entropies(N), scheme_ss))
    return row


def run_real(name: str, f_X: int, f_w: int, clip_X: float, seed: int = 0) -> dict:
    """
    One real-data run: fit on the training half, upload and score the stored half.

    :return: CSV row, with the provisional entropy diagnostics
    """
    from real_data import entropy_provisional, load, stored_codes
    prep = load(name)
    x_spec = FixedPointSpec(f_X, clip_X)
    entropy, diag = entropy_provisional(prep, len(prep.X_store), x_spec)
    W, b = fit_model(prep.task, prep.X_train, prep.y_train)
    row = dict(source=name, family=name, rho=float("nan"), task=prep.task, seed=seed,
               h_symbol_bits=round(diag["h_symbol_bits"], 4), log2_M=round(diag["log2_M"], 3),
               discrete=";".join(diag["discrete"]))
    row.update(evaluate(prep.task, prep.X_store, prep.y_store, stored_codes(prep, x_spec),
                        x_spec, W, b, f_w, entropy, np.random.SeedSequence([seed, f_X, f_w]),
                        prep.y_scale))
    return row


def degree2_plan(N: int, P: int, f_X: int) -> dict:
    """
    Cost-only comparison: what the TRAINING query (least-squares gradient, degree 2, labels
    stored too) would need on N i.i.d. Gaussian rows, at the Phase 1 training format with
    exact entropies (tools/gd_setup.plan_context). Nothing is uploaded.

    :return: dict with d2_log2_q, d2_n, d2_h, d2_m, d2_lam
    """
    from gd_setup import plan_context
    from synthetic_source import SourceSpec
    plan = plan_context(SourceSpec("gauss", n_samples=N, n_features=P, f_X=f_X, clip_X=CLIP_X))
    return dict(d2_log2_q=round(plan["log2_q"], 3), d2_n=N * (P + 1), d2_h=round(plan["h"], 4),
                d2_m=plan["m"], d2_lam=plan["lam"], d2_eps_c=plan["eps_c"])


FAMILIES_S = ["gauss", "uniform", "laplace", "student3", "exponential", "lognormal",
              "bimodal", "spiked", "discrete3"]


def grid(exp: str, fast: bool) -> list:
    """:return: List of run specs (dicts) for one experiment"""
    specs = []
    if exp == "S":
        sources = ([("gauss", 0.0), ("lognormal", 0.0), ("spiked", 0.0)] if fast else
                   [(f, 0.0) for f in FAMILIES_S] + [("gauss", 0.5), ("gauss", 0.9)])
        for (fam, rho) in sources:
            for task in TASKS:
                for P in ([2, 5] if fast else [2, 5, 10, 20]):
                    for seed in ([0] if fast else [0, 1, 2]):
                        specs.append(dict(exp="S", family=fam, rho=rho, task=task, P=P,
                                          N=25 if fast else 100, f_X=DEFAULT_F_X,
                                          f_w=DEFAULT_F_W, seed=seed))
    elif exp == "F":
        bits = [4, 8] if fast else [2, 4, 6, 8, 10, 12]
        for (fam, rho) in [("gauss", 0.0), ("lognormal", 0.0)]:
            for task in ["linear", "logistic"]:
                for f_X in bits:
                    for f_w in bits:
                        for seed in ([0] if fast else [0, 1]):
                            specs.append(dict(exp="F", family=fam, rho=rho, task=task, P=10,
                                              N=20 if fast else 50, f_X=f_X, f_w=f_w,
                                              seed=seed))
    elif exp == "C":
        for N in ([10, 25, 50] if fast else [10, 25, 50, 100, 200, 300, 442]):
            specs.append(dict(exp="C", family="gauss", rho=0.0, task="linear", P=10, N=N,
                              f_X=DEFAULT_F_X, f_w=DEFAULT_F_W, seed=0))
    elif exp == "R":
        for name in ["diabetes", "wine"]:
            for f_X in ([4, 8] if fast else [4, 6, 8]):
                for f_w in ([8] if fast else [4, 6, 8]):
                    for clip in ([4.0] if fast else [4.0, 8.0]):
                        specs.append(dict(exp="R", family=name, task=None, f_X=f_X, f_w=f_w,
                                          clip_X=clip, seed=0))
    else:
        raise ValueError(f"unknown experiment {exp!r}")
    return specs


KEY_FIELDS = ("exp", "family", "rho", "task", "P", "N", "f_X", "f_w", "clip_X", "seed")


def spec_key(d: dict) -> str:
    """:return: Identity of a run, comparable between a spec and a CSV row (all strings)"""
    out = []
    for k in KEY_FIELDS:
        v = d.get(k)
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            out.append("")
        else:
            try:
                out.append(f"{float(v):g}")
            except ValueError:
                out.append(str(v))
    return "|".join(out)


def run_spec(spec: dict) -> dict:
    """
    Runs one spec, recording its power source and wall-clock interval. Sleep/wake events
    are filled in afterwards by mark_sleep_events: reading the macOS power log takes ~4 s,
    longer than most runs.
    """
    from q_sweep import power_source
    power, start = power_source(), time.time()
    if spec["exp"] == "R":
        row = run_real(spec["family"], spec["f_X"], spec["f_w"], spec["clip_X"], spec["seed"])
    else:
        row = run_synthetic(spec["family"], spec["rho"], spec["task"], spec["P"], spec["N"],
                            spec["f_X"], spec["f_w"], spec["seed"])
        if spec["exp"] == "C":
            row.update(degree2_plan(spec["N"], spec["P"], spec["f_X"]))
    out = dict(exp=spec["exp"])
    out.update(row)
    end = time.time()
    out.update(power=power, t_start=round(start, 3), t_end=round(end, 3),
               wall=round(end - start, 3))
    return out


def mark_sleep_events(rows: list):
    """
    Fills 'sleep_events' for every row that lacks it, from ONE read of the macOS power log:
    the number of Sleep/Wake/DarkWake transitions inside the row's [t_start, t_end] (+-1 s).
    A run that straddles one is not a clean timing (CLAUDE.md, working preferences).

    :param rows: Result rows; updated in place
    """
    from q_sweep import _PMSET_EVENT
    pending = [r for r in rows if r.get("sleep_events") in (None, "")]
    if not pending:
        return
    stamps = []
    if sys.platform == "darwin":
        from datetime import datetime
        log = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True).stdout
        for line in log.splitlines()[-4000:]:
            match = _PMSET_EVENT.match(line)
            if match:
                stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
    for r in pending:
        a, b = float(r["t_start"]), float(r["t_end"])
        r["sleep_events"] = sum(a - 1 <= t <= b + 1 for t in stamps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", help="S, F, C, R or all")
    ap.add_argument("--fast", action="store_true", help="smoke-test grids (minutes)")
    ap.add_argument("--resume", action="store_true", help="skip runs already in the CSV")
    ap.add_argument("--dry-run", action="store_true", help="list the grid sizes and exit")
    args = ap.parse_args()

    from q_sweep import machine_info, write_rows
    exps = ["S", "F", "C", "R"] if args.exp == "all" else args.exp.split(",")
    suffix = "_fast" if args.fast else ""
    RESULTS.mkdir(exist_ok=True)
    for exp in exps:
        specs = grid(exp, args.fast)
        path = RESULTS / f"inference_{exp}{suffix}.csv"
        if args.dry_run:
            print(f"{exp}: {len(specs)} runs -> {path.relative_to(ROOT)}")
            continue
        rows = []
        if args.resume and path.exists():
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
        done = {spec_key(r) for r in rows}
        todo = [s for s in specs if spec_key(s) not in done]
        print(f"== {exp}: {len(todo)} to run ({len(specs) - len(todo)} already done)", flush=True)
        for i, spec in enumerate(todo, 1):
            row = run_spec(spec)
            row.update({k: spec[k] for k in ("clip_X",) if k in spec})
            rows.append(row)
            if i % 25 == 0 or i == len(todo):
                mark_sleep_events(rows)
            write_rows(rows, path)
            flag = "" if row["all_exact"] and not row["bound_violations"] else "  <-- CHECK"
            print(f"[{exp} {i}/{len(todo)}] {row['source']:>14s} {row['task']:8s} n={row['n']:5d} "
                  f"f={row['f_X']}/{row['f_w']} log2q={row['log2_q']:.1f} m={row['m']} "
                  f"exact={row['exact']}/{row['queries']} err={row['err_max']:.2e} "
                  f"bound={row['bound_max']:.2e} {row['wall']:.1f}s{flag}", flush=True)
        (RESULTS / f"inference_machine{suffix}.json").write_text(json.dumps(machine_info(), indent=2))


if __name__ == "__main__":
    main()
