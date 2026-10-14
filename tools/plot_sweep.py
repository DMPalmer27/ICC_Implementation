"""
File: plot_sweep.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Figures for the parameter sweep, from the raw CSVs tools/q_sweep.py saves.

    Kept separate from the sweep so the figures can be restyled and regenerated without
    re-running hours of measurement. Every figure is written twice: a vector PDF for the
    LaTeX write-up and a PNG for quick viewing. docs/Q_SWEEP.md interprets them.

    Feasibility is a policy, not a measurement, so the thresholds are flags rather than
    constants; the defaults are the ones docs/Q_SWEEP.md argues for.

Usage:
    .venv/bin/python tools/plot_sweep.py
    .venv/bin/python tools/plot_sweep.py --suffix _fast     # plot the smoke-run CSVs
"""

import _path  # noqa: F401  -- keeps the tools/ convention; icc/ is not needed here

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
NATIVE_BITS = 31.5

PHASES = [("t_construct", "field construction", "setup"),
          ("t_build", "build interpolation matrix M", "setup"),
          ("t_store", "shard to workers", "setup"),
          ("t_eval", "worker evaluation (per step)", "step"),
          ("t_decode", "decode (per step)", "step")]


def load(path: Path) -> list:
    """
    Reads a sweep CSV, converting numeric fields.

    :param path: CSV path
    :return: List of row dicts; missing files give an empty list
    """
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if v in ("", None):
                    out[k] = None
                    continue
                try:
                    out[k] = float(v) if any(c in v for c in ".eE") or k.startswith("t_") \
                        else int(v)
                except ValueError:
                    out[k] = {"True": True, "False": False}.get(v, v)
            rows.append(out)
    return rows


def native_line(ax, label=True):
    """
    Marks the galois native-arithmetic ceiling on an axis whose x is log2 q.

    :param ax: Matplotlib axis
    :param label: Whether to attach a legend label
    """
    ax.axvline(NATIVE_BITS, color="0.4", ls="--", lw=1,
               label="galois native limit (log2 q = 31.5)" if label else None)


def save(fig, out: Path, name: str):
    """
    Writes a figure as PDF and PNG.

    :param fig: Figure
    :param out: Output directory
    :param name: File stem
    """
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_vs_q(rows, out):
    """Experiment A: each phase against log2 q at fixed m -- pure per-operation cost."""
    rows = [r for r in rows if r["status"] == "ok"]
    ms = sorted({r["m"] for r in rows})
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        sub = sorted([r for r in rows if r["m"] == m], key=lambda r: r["log2_q"])
        x = [r["log2_q"] for r in sub]
        for key, label, kind in PHASES:
            ax.plot(x, [r[key] for r in sub], marker="o", ms=3,
                    ls="-" if kind == "step" else ":", label=label)
        native_line(ax, label=(m == ms[0]))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([16, 32, 64, 128, 256, 512])
        ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
        ax.set_xlabel("log2 q  (field size in bits)")
        ax.set_title(f"m = {m}, λ = {sub[0]['lam']:,} (fixed across q)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("seconds (warm)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("A. Cost of each phase vs field size, problem size held fixed", fontsize=11)
    save(fig, out, "fig1_phase_vs_q")


def fig_cost_vs_lambda(rows, out):
    """Experiment B: per-step and setup cost against lambda at three field sizes."""
    rows = [r for r in rows if r["status"] == "ok"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for bits in sorted({r["log2_q"] for r in rows}):
        sub = sorted([r for r in rows if r["log2_q"] == bits], key=lambda r: r["lam"])
        lam = np.array([r["lam"] for r in sub], dtype=float)
        tag = f"log2 q = {bits:.0f}" + (" (native)" if bits <= NATIVE_BITS else " (object)")
        for ax, key in zip(axes, ("t_step", "t_setup")):
            y = np.array([r[key] for r in sub])
            big = lam >= 400
            slope = np.polyfit(np.log(lam[big]), np.log(y[big]), 1)[0] if big.sum() >= 2 else np.nan
            ax.plot(lam, y, marker="o", ms=3, label=f"{tag}, slope {slope:.2f}")
    for ax, title in zip(axes, ("per training step (eval + decode)",
                                "setup, once per upload (construct + store + build M)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("λ = C(m+2, 2) workers")
        ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("B. Scaling with problem size (log-log slope = growth exponent in λ)",
                 fontsize=11)
    save(fig, out, "fig2_cost_vs_lambda")


def fig_m_lambda_vs_q(rows, out):
    """Theory: Theorem 1 key length and worker count against log2 q, for quantised data."""
    rows = [r for r in rows if r["kind"] == "m_vs_q"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for n in sorted({r["n"] for r in rows}):
        sub = sorted([r for r in rows if r["n"] == n], key=lambda r: r["log2_q"])
        x = [r["log2_q"] for r in sub]
        line, = axes[0].plot(x, [r["m"] for r in sub], label=f"n = {n}")
        axes[0].axhline(sub[0]["m_cap"], color=line.get_color(), ls=":", lw=0.8)
        axes[1].plot(x, [r["lam"] for r in sub], color=line.get_color(), label=f"n = {n}")
    for ax in axes:
        native_line(ax, label=(ax is axes[0]))
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512])
        ax.set_xticklabels(["32", "64", "128", "256", "512"])
        ax.set_xlabel("log2 q")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("key length m  (dotted: cap n + p)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_ylabel("λ workers")
    axes[1].set_yscale("log")
    fig.suptitle("Theorem 1: m grows with q for quantised data (f_X = 7), but is capped "
                 "at ≈ n + p", fontsize=11)
    save(fig, out, "fig3_m_lambda_vs_q")


def fig_q_required(rows, out):
    """Theory: the field size a storage format requires, against the native and 64-bit lines."""
    rows = [r for r in rows if r["kind"] == "q_vs_format"]
    if not rows:
        return
    ns_list = sorted({r["n_s"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, f_w in zip(axes, (0, 4, 8)):
        for n_s in ns_list:
            sub = sorted([r for r in rows if r["n_s"] == n_s and r["f_w"] == f_w],
                         key=lambda r: r["f_X"])
            ax.plot([r["f_X"] for r in sub], [r["log2_q"] for r in sub], marker=".",
                    label=f"n_s = {n_s}")
        ax.axhline(NATIVE_BITS, color="0.4", ls="--", lw=1, label="native limit (31.5)")
        ax.axhline(63, color="0.6", ls="-.", lw=1, label="64-bit hardware (63)")
        ax.set_title(f"f_w = {f_w}")
        ax.set_xlabel("f_X  (fractional bits of stored data)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("required log2 q  (no wraparound, P = 3, d = 2)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Field size a quantisation format needs: log2 q ≈ 1 + log2 n_s + "
                 "2(f_X + log2 A_X) + f_w + log2(P·A_w + 1)", fontsize=10)
    save(fig, out, "fig4_q_required")


def fig_feasibility(rows, out, step_cap, setup_cap):
    """Experiment C: which (n, f_X, f_w) formats are feasible, single prime vs CRT."""
    rows = [r for r in rows]
    if not rows:
        return
    ns_vals = sorted({r["n_s"] for r in rows})
    fx_vals = sorted({r["f_X"] for r in rows})
    fw_vals = sorted({r["f_w"] for r in rows})
    fig, axes = plt.subplots(len(fw_vals), 2, figsize=(12, 3.3 * len(fw_vals)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    vmin, vmax = -2, math.log10(step_cap * 10)
    for i, f_w in enumerate(fw_vals):
        for j, backend in enumerate(("single", "crt")):
            ax = axes[i][j]
            grid = np.full((len(ns_vals), len(fx_vals)), np.nan)
            # One row per format. The CRT panel shows the CRT row; a format that fits one
            # native prime has none, because CRT would be the same computation, so it shows
            # the single-prime row there.
            by_key = {(r["n_s"], r["f_X"], r.get("backend")): r for r in rows if r["f_w"] == f_w}
            cells = [by_key.get((ns, fx, backend)) or by_key.get((ns, fx, "single"))
                     for ns in ns_vals for fx in fx_vals]
            for r in [c for c in cells if c is not None]:
                a, b = ns_vals.index(r["n_s"]), fx_vals.index(r["f_X"])
                ok = r["status"] == "ok"
                if ok:
                    grid[a, b] = math.log10(max(r["t_step"], 1e-2))
                feasible = ok and r["t_step"] <= step_cap and r["t_setup"] <= setup_cap
                status = r["status"]
                txt = (f"λ={r['lam']:,}\n{r['t_step']:.1f}s/step\nsetup {r['t_setup']:.0f}s"
                       + (f"\nL={r['L']}" if r.get("L", 1) > 1 else "")
                       if ok else f"λ={r.get('lam') or '?'}\n{status.split(':')[0]}")
                ax.text(b, a, txt + ("" if feasible else "\n✗ infeasible"),
                        ha="center", va="center", fontsize=6,
                        color="white" if ok and grid[a, b] < (vmin + vmax) / 2 else "black")
            ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(fx_vals)))
            ax.set_xticklabels([f"f_X={v}" for v in fx_vals], fontsize=8)
            ax.set_yticks(range(len(ns_vals)))
            ax.set_yticklabels([f"n={v * 4}\n(n_s={v})" for v in ns_vals], fontsize=8)
            ax.set_title(f"{'single prime' if backend == 'single' else 'CRT regime (a)'}, "
                         f"f_w = {f_w}", fontsize=10)
    fig.suptitle(f"C. Feasibility of quantisation formats (feasible: ≤ {step_cap:.0f} s per "
                 f"step, ≤ {setup_cap / 60:.0f} min setup; colour = log10 s per step)",
                 fontsize=10)
    save(fig, out, "fig5_feasibility")


def fig_crt_vs_single(rows, out):
    """Experiment C: single big prime vs CRT over native primes, at the same format and m."""
    ok = [r for r in rows if r["status"] == "ok"]
    pairs = []
    for r in ok:
        if r.get("backend") != "crt":
            continue
        twin = [s for s in ok if s.get("backend") == "single" and s["n_s"] == r["n_s"]
                and s["f_X"] == r["f_X"] and s["f_w"] == r["f_w"]]
        if twin:
            pairs.append((twin[0], r))
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, key, title in zip(axes, ("t_step", "t_setup"),
                              ("per training step", "setup, once per upload")):
        for n_s in sorted({p[0]["n_s"] for p in pairs}):
            sub = sorted([p for p in pairs if p[0]["n_s"] == n_s], key=lambda p: p[0]["log2_q"])
            x = [p[0]["log2_q"] for p in sub]
            line, = ax.plot(x, [p[0][key] for p in sub], marker="o", ms=3,
                            label=f"single prime, n = {4 * n_s}")
            ax.plot(x, [p[1][key] for p in sub], marker="s", ms=3, ls="--",
                    color=line.get_color(), label=f"CRT (a), n = {4 * n_s}")
            for p in sub:
                ax.annotate(f"L={p[1]['L']}", (p[0]["log2_q"], p[1][key]), fontsize=6,
                            textcoords="offset points", xytext=(3, -8))
        ax.set_yscale("log")
        ax.set_xlabel("log2 q required by the format  (single prime)")
        ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("CRT regime (a): L native primes vs one big prime, same format and same "
                 "(jointly sized) m", fontsize=10)
    save(fig, out, "fig6_crt_vs_single")


def fig_construction(rows_d, rows_a, out):
    """Experiment D: verified galois.GF(q) construction against the bypass the sweep uses."""
    if not rows_d:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    d = sorted(rows_d, key=lambda r: r["log2_q"])
    xs = [r["log2_q"] for r in d if r["status"] == "ok"]
    ax.plot(xs, [r["t_construct_verified"] for r in d if r["status"] == "ok"], marker="o",
            label="galois.GF(q), verified (default)")
    for r in d:
        if r["status"] != "ok":
            ax.scatter([r["log2_q"]], [120], marker="x", color="red", zorder=5)
            ax.annotate(f"{r['status']}", (r["log2_q"], 120), fontsize=7, color="red",
                        textcoords="offset points", xytext=(-55, -14))
    a = sorted({(r["log2_q"], r["t_construct"]) for r in rows_a
                if r["status"] == "ok" and r["m"] == min(x["m"] for x in rows_a)})
    ax.plot([p[0] for p in a], [p[1] for p in a], marker="s", ls="--",
            label="with primitive_element given, verify=False")
    native_line(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("log2 q")
    ax.set_ylabel("seconds")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title("D. Field construction (the only step that can hang for huge q)", fontsize=10)
    save(fig, out, "fig7_construction")


def main():
    parser = argparse.ArgumentParser(description="Figures for the q sweep")
    parser.add_argument("--suffix", default="", help="'_fast' to plot the smoke-run CSVs")
    parser.add_argument("--step-cap", type=float, default=60.0,
                        help="feasible if a training step takes at most this many seconds")
    parser.add_argument("--setup-cap", type=float, default=1200.0,
                        help="feasible if setup (once per upload) takes at most this long")
    args = parser.parse_args()

    out = RESULTS / f"figures{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    s = args.suffix
    rows = {e: load(RESULTS / f"q_sweep_{e}{s}.csv") for e in "TABCD"}
    machine = RESULTS / f"q_sweep_machine{s}.json"
    if machine.exists():
        print("machine:", json.loads(machine.read_text()).get("cpu", "?"))

    fig_phase_vs_q(rows["A"], out)
    fig_cost_vs_lambda(rows["B"], out)
    fig_m_lambda_vs_q(rows["T"], out)
    fig_q_required(rows["T"], out)
    fig_feasibility(rows["C"], out, args.step_cap, args.setup_cap)
    fig_crt_vs_single(rows["C"], out)
    fig_construction(rows["D"], rows["A"], out)


if __name__ == "__main__":
    main()
