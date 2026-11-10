"""
File: plot_gd.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Figures for the Phase 1 convergence sweep (docs/GD_LINEAR_REGRESSION.md
    section 10.7), from results/gd_sweep.csv and results/gd_traj/. Kept separate from the
    sweep so figures can be restyled without re-running it. Each figure is written as a
    vector PDF for the LaTeX write-up and a PNG for quick viewing.

    Encoding is fixed across every figure so they read as one set:
      blue   = encoded (through the scheme)
      orange = unencoded float GD on the real-valued data
      aqua   = float GD on the dequantised codes (where shown)
    The first three slots of the reference palette are the ones validated for all-pairs use
    (scatter, small multiples). Distributions are never coloured -- there are too many to
    tell apart by hue -- they are separated by panel or labelled directly.

Usage:
    .venv/bin/python tools/plot_gd.py
    .venv/bin/python tools/plot_gd.py --suffix _fast      # the smoke run
"""

import _path  # noqa: F401  -- keeps the tools/ convention

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_sweep import load  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
ENC, FLT, DEQ = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
FAMILY_ORDER = ("gauss", "uniform", "bimodal", "discrete3", "laplace", "student3",
                "spiked", "exponential", "lognormal")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
})


def save(fig, out: Path, name: str):
    """Writes a figure as PDF and PNG."""
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def ok_rows(rows):
    return [r for r in rows if r.get("status") == "ok"]


def cell_of(r) -> str:
    """:return: Run key without the seed"""
    return r["key"].rsplit("_s", 1)[0]


def by_cell(rows) -> dict:
    out = {}
    for r in rows:
        out.setdefault(cell_of(r), []).append(r)
    return out


def load_traj(traj_dir: Path, key: str) -> dict:
    path = traj_dir / f"{key}.csv"
    if not path.exists():
        return {}
    with open(path) as fh:
        rd = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rd]) for k in ("scheme", "float", "float_deq")}


def _floor(v, eps=1e-12):
    return np.maximum(v, eps)


# ─────────────────────────────────────────────────────────────────────────────

def fig_curves(rows, traj_dir, out):
    """1. Suboptimality vs step per family (S1): encoded vs unencoded, seeds faint."""
    s1 = [r for r in rows if r["sweep"] == "S1"]
    fams = [f for f in FAMILY_ORDER if any(r["family"] == f for r in s1)]
    if not fams:
        return
    cols = 3
    fig, axes = plt.subplots(math.ceil(len(fams) / cols), cols, figsize=(9, 2.4 * math.ceil(len(fams) / cols)),
                             sharey=True, squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        curves = [load_traj(traj_dir, r["key"]) for r in s1 if r["family"] == fam]
        curves = [c for c in curves if c]
        for c in curves:
            ax.plot(_floor(c["float"]), color=FLT, lw=0.8, alpha=0.35)
            ax.plot(_floor(c["scheme"]), color=ENC, lw=0.8, alpha=0.35)
        if curves:
            T = min(len(c["scheme"]) for c in curves)
            ax.plot(np.median([_floor(c["float"][:T]) for c in curves], axis=0), color=FLT,
                    label="unencoded (float)")
            ax.plot(np.median([_floor(c["scheme"][:T]) for c in curves], axis=0), color=ENC,
                    label="encoded (scheme)")
        ax.set_yscale("log")
        ax.set_title(fam, color=INK)
        ax.axhline(1e-2, color=MUTED, lw=0.8, ls=":")
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("(L − L*) / (L₀ − L*)")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    axes[0][0].legend(loc="lower left", frameon=False)
    fig.suptitle(f"Suboptimality by feature distribution, (n_s, P) = ({s1[0]['n_s']}, "
                 f"{s1[0]['P']}); median of seeds, seeds faint; dotted = 1e-2", fontsize=9,
                 color=INK)
    save(fig, out, "gd_fig1_curves_by_family")


def fig_tau_vs_kappa(rows, out, tol="0.01"):
    """2. Steps to tolerance vs sample condition number, every run, encoded vs unencoded."""
    pts = [(r["kappa_sample"], r.get(f"tau_scheme_{tol}"), r.get(f"tau_float_{tol}"), r)
           for r in rows if r.get("kappa_sample")]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    k = np.array([p[0] for p in pts])
    tf = [p[2] for p in pts]
    ts = [p[1] for p in pts]
    reached_f = [i for i, v in enumerate(tf) if v is not None]
    reached_s = [i for i, v in enumerate(ts) if v is not None]
    ax.scatter(k[reached_f] * 0.97, [max(tf[i], 0.8) for i in reached_f], s=22, color=FLT,
               edgecolor="white", linewidth=0.8, label="unencoded (float)", zorder=3)
    ax.scatter(k[reached_s] * 1.03, [max(ts[i], 0.8) for i in reached_s], s=22, color=ENC,
               edgecolor="white", linewidth=0.8, label="encoded (scheme)", zorder=3)
    miss = [i for i, v in enumerate(ts) if v is None]
    if miss:
        top = max([v for v in ts + tf if v is not None] + [1]) * 1.6
        ax.scatter(k[miss], [top] * len(miss), marker="^", s=24, color=ENC, alpha=0.6,
                   label="encoded: not reached in T", zorder=3)
    kk = np.geomspace(max(k.min(), 1), k.max() * 1.2, 50)
    ax.plot(kk, kk * math.log(1 / float(tol)), color=MUTED, lw=1, ls="--",
            label="κ · ln(1/tol)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample condition number κ of [X, 1]ᵀ[X, 1]")
    ax.set_ylabel(f"steps to suboptimality ≤ {tol}")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Convergence speed is set by conditioning, not by encoding", color=INK)
    save(fig, out, "gd_fig2_steps_vs_kappa")


def _cell_label(rs):
    r = rs[0]
    parts = [r["family"]]
    if r.get("rho"):
        parts.append(f"ρ={r['rho']:g}")
    if (r["n_s"], r["P"]) != (25, 3):
        parts.append(f"({r['n_s']},{r['P']})")
    if r.get("snr") not in (1, 1.0, None) or r.get("noise") != "gauss":
        parts.append(f"snr {r['snr']:g} {r['noise']}")
    return " ".join(parts)


def _dot_rows(ax, cells, fields, log=True):
    labels = []
    for y, (cell, rs) in enumerate(cells):
        labels.append(_cell_label(rs))
        for field, color, label in fields:
            vals = [r[field] for r in rs if r.get(field) is not None]
            if not vals:
                continue
            ax.scatter(vals, [y] * len(vals), s=12, color=color, alpha=0.4, linewidth=0)
            ax.scatter([float(np.median(vals))], [y], s=36, color=color, edgecolor="white",
                       linewidth=1.2, zorder=3, label=label if y == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)


def fig_plateau(rows, out):
    """3. Final suboptimality (the floor) per cell: what fixed point costs in accuracy."""
    cells = sorted(by_cell(rows).items(), key=lambda kv: (kv[1][0]["sweep"], kv[0]))
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(cells) + 1.2))
    _dot_rows(ax, cells, [("plateau_float", FLT, "unencoded (float)"),
                          ("plateau_float_deq", DEQ, "float on quantised data"),
                          ("plateau_scheme", ENC, "encoded (scheme)")])
    ax.set_xlabel("suboptimality at the end of the run (median of last 10% of steps)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Accuracy floor: quantisation of the weights dominates", color=INK)
    save(fig, out, "gd_fig3_plateau")


def fig_wallclock(rows, out, tol="0.01"):
    """4. Wall-clock to tolerance per cell, encoded (incl. setup) vs unencoded."""
    cells = sorted(by_cell(rows).items(), key=lambda kv: (kv[1][0]["sweep"], kv[0]))
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(cells) + 1.2))
    _dot_rows(ax, cells, [(f"wall_float_{tol}", FLT, "unencoded (float)"),
                          (f"wall_scheme_{tol}", ENC, "encoded: setup + steps")])
    ax.set_xlabel(f"seconds to suboptimality ≤ {tol} (log scale)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Wall-clock to converge: the scheme's overhead", color=INK)
    save(fig, out, "gd_fig4_wallclock")


def fig_cost_by_family(rows, out):
    """5. Per-step cost and worker count by family (S1): low entropy costs more."""
    s1 = [r for r in rows if r["sweep"] == "S1"]
    fams = [f for f in FAMILY_ORDER if any(r["family"] == f for r in s1)]
    if not fams:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, field, xlabel in ((axes[0], "lam", "workers λ"),
                              (axes[1], "t_step_scheme", "seconds per encoded step")):
        for y, fam in enumerate(fams):
            vals = [r[field] for r in s1 if r["family"] == fam]
            ax.barh(y, float(np.median(vals)), color=ENC, height=0.6)
            ax.text(float(np.median(vals)), y, f" {np.median(vals):,.4g}", va="center",
                    color=INK, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(fams)))
    axes[0].set_yticklabels([f"{f}  (h={np.median([r['h'] for r in s1 if r['family'] == f]):.2f})"
                             for f in fams])
    axes[0].invert_yaxis()
    fig.suptitle(f"Cost of protecting each distribution at ({s1[0]['n_s']}, {s1[0]['P']}): "
                 f"lower entropy h → larger key m → more workers", fontsize=9, color=INK)
    save(fig, out, "gd_fig5_cost_by_family")


def fig_size_noise(rows, out, tol="0.01"):
    """6. Steps to tolerance across sizes (S3 + the S1 base) and across SNR (S4)."""
    size_rows = [r for r in rows if r["sweep"] == "S3"
                 or (r["sweep"] == "S1" and r["family"] in ("gauss", "exponential", "spiked"))]
    noise_rows = [r for r in rows if r["sweep"] == "S4"
                  or (r["sweep"] == "S1" and r["family"] == "gauss")]
    if not size_rows and not noise_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    sizes = sorted({(r["n_s"], r["P"]) for r in size_rows}, key=lambda s: (s[1], -s[0]))
    fams = [f for f in ("gauss", "exponential", "spiked") if any(r["family"] == f for r in size_rows)]
    width = 0.8 / max(len(fams), 1)
    for i, fam in enumerate(fams):
        for j, s in enumerate(sizes):
            for field, color, dx in ((f"tau_float_{tol}", FLT, -0.1), (f"tau_scheme_{tol}", ENC, 0.1)):
                vals = [r[field] for r in size_rows if r["family"] == fam
                        and (r["n_s"], r["P"]) == s and r.get(field) is not None]
                if vals:
                    x = j + (i - (len(fams) - 1) / 2) * width + dx * width
                    ax.scatter([x] * len(vals), vals, s=12, color=color, alpha=0.5, linewidth=0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"({a},{b})\nn={a * (b + 1)}" for a, b in sizes])
    ax.set_yscale("log")
    ax.set_ylabel(f"steps to ≤ {tol}")
    ax.set_title("By size (n_s, P); within each size, left to right: " + " / ".join(fams),
                 color=INK)
    ax.scatter([], [], color=ENC, label="encoded")
    ax.scatter([], [], color=FLT, label="unencoded")
    ax.legend(frameon=False)

    ax = axes[1]
    for noise, marker in (("gauss", "o"), ("laplace", "s")):
        for field, color in (("plateau_float", FLT), ("plateau_scheme", ENC)):
            pts = [(r["snr"], r[field]) for r in noise_rows if r["noise"] == noise
                   and r.get(field) is not None]
            if pts:
                snrs = sorted({p[0] for p in pts})
                med = [np.median([v for s, v in pts if s == x]) for x in snrs]
                ax.plot(snrs, med, color=color, marker=marker, ms=5, lw=1.5,
                        ls="-" if noise == "gauss" else "--",
                        label=f"{'encoded' if color == ENC else 'unencoded'}, {noise} noise")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel("final suboptimality")
    if noise_rows:
        ax.set_title(f"By label noise, Gaussian features ({noise_rows[0]['n_s']}, "
                     f"{noise_rows[0]['P']})", color=INK)
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "gd_fig6_size_noise")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 GD figures")
    parser.add_argument("--suffix", default="", help="'_fast' for the smoke run")
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    res = Path(args.results)
    rows = ok_rows(load(res / f"gd_sweep{args.suffix}.csv"))
    if not rows:
        print("no ok rows to plot")
        return
    out = res / f"figures{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    traj_dir = res / f"gd_traj{args.suffix}"
    print(f"{len(rows)} runs -> {out}")
    fig_curves(rows, traj_dir, out)
    fig_tau_vs_kappa(rows, out)
    fig_plateau(rows, out)
    fig_wallclock(rows, out)
    fig_cost_by_family(rows, out)
    fig_size_noise(rows, out)


if __name__ == "__main__":
    main()
