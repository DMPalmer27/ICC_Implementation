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
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")   # keeps figure-level legends
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
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
        for tol in (1e-1, 1e-2):
            ax.axhline(tol, color=MUTED, lw=0.8, ls=":")
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("(L − L*) / (L₀ − L*)")
    for ax in axes[-1]:
        ax.set_xlabel("step")
    axes[0][0].legend(loc="lower left", frameon=False)
    fig.suptitle(f"Suboptimality by feature distribution, (n_s, P) = ({s1[0]['n_s']}, "
                 f"{s1[0]['P']}); median of seeds, seeds faint; dotted = 1e-1, 1e-2", fontsize=9,
                 color=INK)
    save(fig, out, "gd_fig1_curves_by_family")


def _tau_panel(ax, pts, tol):
    k = np.array([p[0] for p in pts])
    ts = [p[1] for p in pts]
    tf = [p[2] for p in pts]
    top = max([v for v in ts + tf if v is not None] + [1]) * 2.5
    for vals, color, dx, label in ((tf, FLT, 0.97, "unencoded (float)"),
                                   (ts, ENC, 1.03, "encoded (scheme)")):
        hit = [i for i, v in enumerate(vals) if v is not None]
        ax.scatter(k[hit] * dx, [max(vals[i], 0.8) for i in hit], s=22, color=color,
                   edgecolor="white", linewidth=0.8, label=label, zorder=3)
        miss = [i for i, v in enumerate(vals) if v is None]
        if miss:
            ax.scatter(k[miss] * dx, [top] * len(miss), marker="^", s=26, facecolor="white",
                       edgecolor=color, linewidth=1.2, zorder=3,
                       label=f"{label.split(' ')[0]}: not reached within T")
    kk = np.geomspace(max(k.min(), 1), k.max() * 1.2, 50)
    ax.plot(kk, kk * math.log(1 / tol), color=MUTED, lw=1, ls="--", label="κ · ln(1/tol)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.6, top * 2.2)
    ax.set_xlabel("sample condition number κ of [X, 1]ᵀ[X, 1]")
    ax.set_ylabel(f"steps to suboptimality ≤ {tol:g}")
    ax.set_title(f"tolerance {tol:g}", color=INK)


def fig_tau_vs_kappa(rows, out, tols=("0.1", "0.01")):
    """
    2. Steps to tolerance vs sample condition number, every run, encoded vs unencoded.
    Two tolerances: 1e-1 measures the descent alone; at 1e-2 the encoded run can sit on the
    floor set by the weight grid 2^-f_w (figure 3), which is a format effect, not encoding.
    """
    fig, axes = plt.subplots(1, len(tols), figsize=(4.6 * len(tols), 4.2), sharey=True,
                             squeeze=False)
    any_pts = False
    for ax, tol in zip(axes[0], tols):
        pts = [(r["kappa_sample"], r.get(f"tau_scheme_{tol}"), r.get(f"tau_float_{tol}"))
               for r in rows if r.get("kappa_sample")]
        if pts:
            any_pts = True
            _tau_panel(ax, pts, float(tol))
    if not any_pts:
        plt.close(fig)
        return
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Steps to converge vs conditioning; hollow triangles = not reached within the "
                 "run", fontsize=9, color=INK)
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


def _dot_rows(ax, cells, fields, log=True, floor=None, miss_x=None, invert=True):
    """
    One row per cell: seeds faint, median solid.

    :param floor: Values below it are drawn at it (log axes cannot show 0)
    :param miss_x: If given, a run with no value (tolerance not reached) is drawn there as a
        hollow triangle; otherwise it is left out
    :param invert: Put the first cell at the top (once per shared y axis)
    """
    labels, seen = [], set()
    for y, (cell, rs) in enumerate(cells):
        labels.append(_cell_label(rs))
        for f_i, (field, color, label) in enumerate(fields):
            yy = y + (f_i - (len(fields) - 1) / 2) * 0.18   # equal values stay visible
            vals = [r[field] for r in rs if r.get(field) is not None]
            if floor is not None:
                vals = [max(v, floor) for v in vals]
            n_miss = sum(1 for r in rs if r.get(field) is None)
            if miss_x is not None and n_miss:
                ax.scatter([miss_x] * n_miss, [yy] * n_miss, marker="^", s=26,
                           facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
            if not vals:
                continue
            ax.scatter(vals, [yy] * len(vals), s=12, color=color, alpha=0.4, linewidth=0)
            ax.scatter([float(np.median(vals))], [yy], s=36, color=color, edgecolor="white",
                       linewidth=1.2, zorder=3, label=label if label not in seen else None)
            seen.add(label)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if invert:
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
                          ("plateau_scheme", ENC, "encoded (scheme)")], floor=1e-12)
    ax.axvspan(1e-13, 1.5e-12, color="#e4e3df", lw=0, zorder=0)
    ax.set_xlim(left=1e-13)
    ax.set_xlabel("suboptimality at the end of the run (median of last 10% of steps)\n"
                  "grey band: ≤ 1e-12, converged to floating-point precision")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 0.0))
    ax.set_title("Accuracy floor: the encoded run sits on the weight grid 2^-f_w",
                 color=INK)
    save(fig, out, "gd_fig3_plateau")


def fig_wallclock(rows, out, tols=("0.1", "0.01")):
    """4. Wall-clock to tolerance per cell, encoded (incl. setup) vs unencoded."""
    cells = sorted(by_cell(rows).items(), key=lambda kv: (kv[1][0]["sweep"], kv[0]))
    if not cells:
        return
    fig, axes = plt.subplots(1, len(tols), figsize=(4.4 * len(tols) + 1.4,
                                                    0.24 * len(cells) + 1.4), sharey=True)
    vals = [r[f"wall_{n}_{t}"] for r in rows for n in ("float", "scheme") for t in tols
            if r.get(f"wall_{n}_{t}") is not None]
    miss_x = max(vals) * 4 if vals else 1
    for i, (ax, tol) in enumerate(zip(axes, tols)):
        _dot_rows(ax, cells, [(f"wall_float_{tol}", FLT, "unencoded (float)"),
                              (f"wall_scheme_{tol}", ENC, "encoded: setup + steps")],
                  miss_x=miss_x, invert=i == 0)
        ax.set_xlabel(f"seconds to suboptimality ≤ {float(tol):g}")
        ax.set_title(f"tolerance {float(tol):g}", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Wall-clock to converge: the scheme's overhead (log scale; hollow triangle = "
                 "not reached within the run)", fontsize=9, color=INK)
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


def fig_size_noise(rows, out, tol="0.1"):
    """6. Steps to tolerance across sizes (S3 + the S1 base) and across SNR (S4)."""
    size_rows = [r for r in rows if r["sweep"] == "S3"
                 or (r["sweep"] == "S1" and r["family"] in ("gauss", "exponential", "spiked"))]
    noise_rows = [r for r in rows if r["sweep"] == "S4"
                  or (r["sweep"] == "S1" and r["family"] == "gauss")]
    if not size_rows and not noise_rows:
        return
    size_cells = sorted(by_cell(size_rows).items(),
                        key=lambda kv: ((kv[1][0]["n_s"] * (kv[1][0]["P"] + 1)),
                                        kv[1][0]["P"], kv[1][0]["family"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.4, 0.24 * len(size_cells) + 1.4)))
    ax = axes[0]
    if size_cells:
        taus = [r[f] for r in size_rows for f in (f"tau_float_{tol}", f"tau_scheme_{tol}")
                if r.get(f) is not None]
        _dot_rows(ax, size_cells, [(f"tau_float_{tol}", FLT, "unencoded (float)"),
                                   (f"tau_scheme_{tol}", ENC, "encoded (scheme)")],
                  miss_x=max(taus + [1]) * 3)
        ax.set_yticklabels([f"{rs[0]['family']} ({rs[0]['n_s']},{rs[0]['P']})  "
                            f"n={rs[0]['n_s'] * (rs[0]['P'] + 1)}" for _, rs in size_cells])
        ax.set_xlabel(f"steps to suboptimality ≤ {float(tol):g} (hollow = not reached)")
        ax.legend(frameon=False, loc="lower right")
    ax.set_title("By size (n_s, P), sorted by n = n_s(P+1) data symbols", color=INK)

    ax = axes[1]
    for noise, marker in (("gauss", "o"), ("laplace", "s")):
        for field, color in (("plateau_float", FLT), ("plateau_scheme", ENC)):
            pts = [(r["snr"], max(r[field], 1e-12)) for r in noise_rows
                   if r["noise"] == noise and r.get(field) is not None]
            if pts:
                snrs = sorted({p[0] for p in pts})
                med = [np.median([v for s, v in pts if s == x]) for x in snrs]
                ax.plot(snrs, med, color=color, marker=marker, ms=5, lw=1.5,
                        ls="-" if noise == "gauss" else "--",
                        label=f"{'encoded' if color == ENC else 'unencoded'}, {noise} noise")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signal-to-noise ratio")
    ax.set_ylabel("final suboptimality (clipped at 1e-12)")
    if noise_rows:
        ax.set_title(f"By label noise, Gaussian features ({noise_rows[0]['n_s']}, "
                     f"{noise_rows[0]['P']})", color=INK)
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "gd_fig6_size_noise")


FW_RAMP = ("#a9c8f0", "#6aa1e6", "#2a78d6", "#174a8c")   # one hue, light -> dark = more bits


def fig_fw_floor(rows, ablation, out):
    """
    7. The encoded floor vs the weight format f_w (tools/gd_fw_ablation.py, plaintext integer
    GD, bit-identical to the scheme at f_w = 4). Two floors: the weight grid 2^-f_w, which
    more bits lowers, and the data's own quantisation and clipping (float GD on the stored
    codes, green), which no choice of f_w can beat.
    """
    if not ablation:
        return
    deq = {r["key"]: r.get("plateau_float_deq") for r in rows}
    cells = {}
    for a in ablation:
        cells.setdefault(a["key"].rsplit("_s", 1)[0], []).append(a)
    order = sorted(cells.items(), key=lambda kv: (kv[1][0]["sweep"], kv[0]))
    f_ws = sorted({a["f_w"] for a in ablation})
    fig, ax = plt.subplots(figsize=(6.8, 0.26 * len(order) + 1.4))
    labels = []
    for y, (cell, rs) in enumerate(order):
        labels.append(_cell_label(rs))
        for i, f_w in enumerate(f_ws):
            vals = [max(a["plateau"], 1e-12) for a in rs if a["f_w"] == f_w]
            if vals:
                ax.scatter([float(np.median(vals))], [y], s=34, color=FW_RAMP[i % 4],
                           edgecolor="white", linewidth=1, zorder=3,
                           label=f"encoded, f_w = {f_w}" if y == 0 else None)
        dq = [max(deq[a["key"]], 1e-12) for a in rs if a["f_w"] == f_ws[0]
              and deq.get(a["key"]) is not None]
        if dq:
            ax.scatter([float(np.median(dq))], [y], marker="|", s=140, color=DEQ,
                       linewidth=2, zorder=2,
                       label="data floor: float GD on the stored codes" if y == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("suboptimality at the end of the run (median over seeds)")
    base = [a for a in ablation if a["family"] == "gauss" and a["sweep"] == "S1"]
    cost = "; ".join(f"f_w={f}: log2 q {np.median([a['log2_q'] for a in base if a['f_w'] == f]):.1f}"
                     f", λ {int(np.median([a['lam'] for a in base if a['f_w'] == f])):,}"
                     for f in f_ws) if base else ""
    ax.set_title("Encoded floor vs weight bits f_w (plaintext integer GD = the scheme, "
                 "bit for bit)", color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=7)
    if cost:
        fig.text(0.5, -0.07, f"Cost at (25, 3) gauss — {cost}  (native field ends at "
                 f"log2 q ≈ 31.5)", ha="center", fontsize=7, color=MUTED)
    save(fig, out, "gd_fig7_fw_floor")


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
    abl = res / f"gd_fw_ablation{args.suffix}.csv"
    if abl.exists():
        fig_fw_floor(rows, load(abl), out)


if __name__ == "__main__":
    main()
