"""
File: plot_inference.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Figures for the inference paper (docs/INFERENCE.md), from
    results/inference_{S,F,C,R}.csv. Kept separate from the study so figures can be restyled
    without re-running it. Each figure is written as a vector PDF for LaTeX and a PNG.

    Encoding is fixed across every figure, as in tools/plot_gd.py:
      blue   = through the scheme
      orange = float reference (unencoded model on the real-valued data)
      grey   = bounds and guide lines
    Figure 1 recomputes the real-data scores from the plaintext INTEGER ORACLE rather than
    re-running the scheme; this is the scheme's output exactly, because every R run in
    results/inference_R.csv decoded every score equal to that oracle (column all_exact),
    which the script checks before plotting.

Usage:
    .venv/bin/python tools/plot_inference.py
    .venv/bin/python tools/plot_inference.py --suffix _fast      # the smoke run
"""

import _path  # noqa: F401  -- puts icc/ on sys.path

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_sweep import load  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
ENC, FLT, GUIDE = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
FAMILY_ORDER = ("gauss", "uniform", "bimodal", "discrete3", "laplace", "student3",
                "spiked", "exponential", "lognormal")
TASK_LABEL = {"linear": "linear", "logistic": "logistic", "softmax": "softmax (K=3)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
})


def save(fig, out: Path, name: str):
    """Writes a figure as PDF and PNG."""
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def truthy(v) -> bool:
    return v in (True, 1, "True", "true", "1")


def source_label(r) -> str:
    rho = r.get("rho")
    if rho not in (None, 0, 0.0) and not (isinstance(rho, float) and math.isnan(rho)):
        return f"{r['family']} ρ={rho:g}"
    return r["family"]


def source_order(labels) -> list:
    def key(s):
        fam = s.split(" ")[0]
        return (FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, s)
    return sorted(set(labels), key=key)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: real data, scheme vs float, per stored row
# ─────────────────────────────────────────────────────────────────────────────

def fig_real_scatter(R: list, out: Path, f_X=8, f_w=8, clip=4.0):
    from inference_setup import fit_model, integer_scores, quantise_model
    from quantize import FixedPointSpec
    from real_data import load as load_real, stored_codes

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, name in zip(axes, ("diabetes", "wine")):
        row = [r for r in R if r["family"] == name and r["f_X"] == f_X and r["f_w"] == f_w
               and float(r["clip_X"]) == clip]
        if not row or not truthy(row[0]["all_exact"]):
            ax.set_visible(False)
            continue
        prep = load_real(name)
        x_spec = FixedPointSpec(f_X, clip)
        W, b = fit_model(prep.task, prep.X_train, prep.y_train)
        model = quantise_model(W, b, f_w)
        Zi = integer_scores(stored_codes(prep, x_spec), model["W_codes"], model["b_codes"], f_X)
        Zs = np.asarray(Zi, dtype=float) / 2.0 ** (f_X + f_w)
        Zf = prep.X_store @ W.T + b
        if name == "diabetes":
            ys, yf = Zs[:, 0] * prep.y_scale, Zf[:, 0] * prep.y_scale
            ax.scatter(yf, ys, s=10, color=ENC, alpha=0.7, lw=0)
            lo, hi = min(yf.min(), ys.min()), max(yf.max(), ys.max())
            ax.plot([lo, hi], [lo, hi], color=GUIDE, lw=1)
            ax.set_xlabel("float prediction (disease progression units)")
            ax.set_ylabel("scheme prediction")
            ax.set_title(f"diabetes (linear), {len(ys)} stored rows\nR² {row[0]['scheme_r2']:.4f}"
                         f" scheme vs {row[0]['float_r2']:.4f} float\nmax |Δ| "
                         f"{np.abs(ys - yf).max():.2f} units, {row[0]['clipped_values']} clipped")
        else:
            for k, mk in zip(range(Zs.shape[1]), "o^s"):
                ax.scatter(Zf[:, k], Zs[:, k], s=10, color=ENC, alpha=0.6, lw=0, marker=mk,
                           label=f"class {k} logit")
            lo, hi = min(Zf.min(), Zs.min()), max(Zf.max(), Zs.max())
            ax.plot([lo, hi], [lo, hi], color=GUIDE, lw=1)
            ax.set_xlabel("float logit")
            ax.set_ylabel("scheme logit")
            ax.legend(frameon=False, loc="upper left")
            ax.set_title(f"wine (softmax), {len(Zs)} stored rows\naccuracy "
                         f"{row[0]['scheme_acc']:.3f} scheme vs {row[0]['float_acc']:.3f} float\n"
                         f"decision flips: {row[0]['flips']}, {row[0]['clipped_values']} clipped")
    fig.suptitle(f"Real data through the scheme (f_X = {f_X}, f_w = {f_w}, clip ±{clip:g}σ)",
                 fontsize=9)
    save(fig, out, "fig1_real_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: measured error vs the analytic bound (every S run)
# ─────────────────────────────────────────────────────────────────────────────

def fig_error_bound(S: list, out: Path):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    clipped = [r for r in S if (r["clipped_values"] or 0) > 0]
    clean = [r for r in S if (r["clipped_values"] or 0) == 0]
    for rows, col, lab in ((clean, ENC, "no feature clipped"),
                           (clipped, FLT, "some feature clipped")):
        if rows:
            ax.scatter([r["bound_max"] for r in rows], [r["err_max"] for r in rows], s=12,
                       color=col, alpha=0.7, lw=0, label=f"{lab} ({len(rows)} runs)")
    vals = [r["bound_max"] for r in S] + [r["err_max"] for r in S]
    lo, hi = min(v for v in vals if v > 0) / 2, max(vals) * 2
    ax.plot([lo, hi], [lo, hi], color=GUIDE, lw=1, label="error = bound")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("analytic bound, max over the run's scores")
    ax.set_ylabel("measured |z_scheme − z_float|, max")
    viol = sum(r["bound_violations"] or 0 for r in S)
    ax.set_title(f"Score error never exceeds the bound ({viol} violations, "
                 f"{sum(r['queries'] for r in S):,} scores)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "fig2_error_vs_bound")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: format grid (F)
# ─────────────────────────────────────────────────────────────────────────────

def _grid(rows, value):
    fx = sorted({r["f_X"] for r in rows})
    fw = sorted({r["f_w"] for r in rows})
    A = np.full((len(fw), len(fx)), np.nan)
    for i, w in enumerate(fw):
        for j, x in enumerate(fx):
            v = [value(r) for r in rows if r["f_X"] == x and r["f_w"] == w]
            v = [u for u in v if u is not None and not (isinstance(u, float) and math.isnan(u))]
            if v:
                A[i, j] = float(np.mean(v))
    return fx, fw, A


def _heat(ax, fx, fw, A, title, fmt, cmap="Blues", vmin=None, vmax=None):
    im = ax.imshow(A, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(fx)), fx), ax.set_yticks(range(len(fw)), fw)
    ax.set_xlabel("f_X (data fractional bits)"), ax.set_ylabel("f_w (weight fractional bits)")
    ax.grid(False)
    ax.set_title(title)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isnan(A[i, j]):
                rgb = im.cmap(im.norm(A[i, j]))[:3]
                dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
                ax.text(j, i, fmt(A[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
    return im


def fig_format(F: list, out: Path):
    for fam in ("gauss", "lognormal"):
        rows = [r for r in F if r["family"] == fam]
        if not rows:
            continue
        lin = [r for r in rows if r["task"] == "linear"]
        log = [r for r in rows if r["task"] == "logistic"]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
        fx, fw, A = _grid(lin, lambda r: math.log10(r["err_max"]) if r["err_max"] else None)
        _heat(axes[0], fx, fw, A, "linear: log10 max score error", lambda v: f"{v:.1f}",
              cmap="Blues_r")
        fx, fw, A = _grid(lin, lambda r: r["scheme_r2"] - r["float_r2"])
        _heat(axes[1], fx, fw, A, "linear: R² scheme − float", lambda v: f"{v:+.3f}",
              cmap="RdBu", vmin=-max(1e-9, np.nanmax(np.abs(A))), vmax=max(1e-9, np.nanmax(np.abs(A))))
        fx, fw, A = _grid(log, lambda r: 100.0 * r["flips"] / r["N"])
        _heat(axes[2], fx, fw, A, "logistic: % decisions flipped", lambda v: f"{v:.1f}",
              cmap="Oranges")
        fig.suptitle(f"Fixed-point format, {fam} features, P = {rows[0]['P']}, "
                     f"N = {rows[0]['N']} stored rows (mean of seeds)", fontsize=9)
        save(fig, out, f"fig3_format_{fam}")

    # field size and key length across the grid (independent of the family's task metric)
    rows = [r for r in F if r["family"] == "gauss" and r["task"] == "linear"]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
        fx, fw, A = _grid(rows, lambda r: r["log2_q"])
        _heat(axes[0], fx, fw, A, "log2 q", lambda v: f"{v:.0f}")
        fx, fw, A = _grid(rows, lambda r: r["m"] / r["n"])
        _heat(axes[1], fx, fw, A, "m / n (key length per stored value)", lambda v: f"{v:.2f}")
        fig.suptitle("Cost of the format: field size and key length (gauss, linear)", fontsize=9)
        save(fig, out, "fig3c_format_cost")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: key length across sources (S)
# ─────────────────────────────────────────────────────────────────────────────

def fig_key_length(S: list, out: Path):
    rows = [r for r in S if r["task"] == "linear"]
    labels = source_order(source_label(r) for r in rows)
    Ps = sorted({r["P"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for k, P in enumerate(Ps):
        y = []
        for lab in labels:
            v = [r["m"] / r["n"] for r in rows if source_label(r) == lab and r["P"] == P]
            y.append(np.mean(v) if v else np.nan)
        axes[0].plot(range(len(labels)), y, marker="o", ms=4, lw=1,
                     color=ENC, alpha=0.35 + 0.65 * k / max(1, len(Ps) - 1), label=f"P = {P}")
    axes[0].set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axes[0].set_ylabel("m / n")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1, color=GUIDE, lw=1)
    axes[0].set_title("Key length per stored value (N = 100 rows, f_X = f_w = 8)")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    for lab in labels:
        sel = [r for r in rows if source_label(r) == lab]
        axes[1].scatter([r["h"] for r in sel], [r["m"] / r["n"] for r in sel], s=12, lw=0,
                        color=ENC, alpha=0.7)
    hs = np.linspace(0, 1, 50)
    axes[1].plot(hs, 1 - hs, color=GUIDE, lw=1,
                 label="m/n = 1 − h  (p, log(1/ε), max_R terms omitted)")
    axes[1].set_xlabel("h = H_p(X) / n  (entropy per value, log_q units)")
    axes[1].set_ylabel("m / n")
    axes[1].set_title("Theorem 1: m ≈ n(1 − h)")
    axes[1].legend(frameon=False)
    save(fig, out, "fig4_key_length")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: task metric, scheme vs float, across sources (S)
# ─────────────────────────────────────────────────────────────────────────────

def fig_metrics(S: list, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    spec = (("linear", "r2", "R²"), ("logistic", "acc", "accuracy"),
            ("softmax", "acc", "accuracy"))
    for ax, (task, met, lab) in zip(axes, spec):
        rows = [r for r in S if r["task"] == task]
        if not rows:
            ax.set_visible(False)
            continue
        f = np.array([r[f"float_{met}"] for r in rows], dtype=float)
        s = np.array([r[f"scheme_{met}"] for r in rows], dtype=float)
        clipped = np.array([(r["clipped_values"] or 0) > 0 for r in rows])
        ax.scatter(f[~clipped], s[~clipped], s=12, lw=0, color=ENC, alpha=0.7,
                   label="no feature clipped")
        ax.scatter(f[clipped], s[clipped], s=12, lw=0, color=FLT, alpha=0.7,
                   label="some feature clipped")
        lo, hi = min(f.min(), s.min()), max(f.max(), s.max())
        ax.plot([lo, hi], [lo, hi], color=GUIDE, lw=1)
        ax.set_xlabel(f"float {lab}"), ax.set_ylabel(f"scheme {lab}")
        ax.set_title(f"{TASK_LABEL[task]}: max |Δ| = {np.abs(s - f).max():.4f} "
                     f"({len(rows)} runs)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Task metrics through the scheme vs the float model, all synthetic sources",
                 fontsize=9)
    save(fig, out, "fig5_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: cost vs stored size (C), degree 1 vs degree 2
# ─────────────────────────────────────────────────────────────────────────────

def _plain_log(ax):
    """Log axes with plain-number major ticks and no minor labels."""
    from matplotlib.ticker import NullFormatter, FuncFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())


def fig_cost(C: list, GD: list, out: Path):
    C = sorted(C, key=lambda r: r["n"])
    if not C:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    n = [r["n"] for r in C]
    axes[0].plot(n, [r["lam"] for r in C], marker="o", ms=4, color=ENC,
                 label="inference, d = 1 (measured)")
    if C[0].get("d2_lam") is not None:
        axes[0].plot([r["d2_n"] for r in C], [r["d2_lam"] for r in C], marker="s", ms=4,
                     color=FLT, ls="--", label="training gradient, d = 2 (plan)")
    if GD:
        axes[0].scatter([r["n"] for r in GD], [r["lam"] for r in GD], s=14, color=FLT,
                        marker="x", label="training, d = 2 (measured, Phase 1)")
    axes[0].set_xscale("log"), axes[0].set_yscale("log")
    axes[0].set_xlabel("stored values n"), axes[0].set_ylabel("workers λ (= download size)")
    axes[0].set_title("Workers: λ = m + 1 at d = 1, C(m+2, 2) at d = 2")
    axes[0].legend(frameon=False, loc="upper left")
    _plain_log(axes[0])
    phases = (("t_store", "store (shares)"), ("t_build", "build M (once)"),
              ("t_evaluate", "evaluate all queries"), ("t_decode", "decode all scores"))
    for (k, lab), a in zip(phases, (1.0, 0.75, 0.5, 0.3)):
        axes[1].plot(n, [max(r[k], 1e-4) for r in C], marker="o", ms=4, color=ENC, alpha=a,
                     label=lab)
    axes[1].set_xscale("log"), axes[1].set_yscale("log")
    axes[1].set_xlabel("stored values n  (N rows × P = 10)"), axes[1].set_ylabel("seconds")
    axes[1].set_title(f"Measured time, d = 1, log2 q ≈ {np.median([r['log2_q'] for r in C]):.0f}")
    axes[1].legend(frameon=False, loc="upper left")
    _plain_log(axes[1])
    save(fig, out, "fig6_cost")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: real data across formats (R)
# ─────────────────────────────────────────────────────────────────────────────

def fig_real_format(R: list, out: Path):
    if not R:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    clips = sorted({float(r["clip_X"]) for r in R})
    fws = sorted({r["f_w"] for r in R})
    for name, met, ax, lab in (("diabetes", "r2", axes[0], "R²"),
                               ("wine", "acc", axes[1], "accuracy")):
        rows = [r for r in R if r["family"] == name]
        for ci, clip in enumerate(clips):
            for wi, fw in enumerate(fws):
                sel = sorted([r for r in rows if float(r["clip_X"]) == clip and r["f_w"] == fw],
                             key=lambda r: r["f_X"])
                if not sel:
                    continue
                ax.plot([r["f_X"] for r in sel], [r[f"scheme_{met}"] for r in sel],
                        marker="o" if ci == 0 else "s", ms=4, lw=1,
                        ls="-" if ci == 0 else "--", color=ENC,
                        alpha=0.35 + 0.65 * wi / max(1, len(fws) - 1),
                        label=f"clip ±{clip:g}σ, f_w = {fw}")
        if rows:
            ax.axhline(rows[0][f"float_{met}"], color=FLT, lw=1.5, label="float model")
        ax.set_xlabel("f_X"), ax.set_ylabel(f"scheme {lab}")
        ax.set_title(f"{name}: {lab} through the scheme")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    for name, mk in (("diabetes", "o"), ("wine", "^")):
        sel = sorted([r for r in R if r["family"] == name and float(r["clip_X"]) == clips[0]
                      and r["f_w"] == fws[-1]], key=lambda r: r["f_X"])
        if sel:
            axes[2].plot([r["f_X"] for r in sel], [r["m"] / r["n"] for r in sel], marker=mk,
                         ms=5, color=ENC, label=f"{name} (n = {sel[0]['n']})")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("f_X"), axes[2].set_ylabel("m / n")
    axes[2].set_title("Key length, provisional entropy bound")
    axes[2].legend(frameon=False)
    save(fig, out, "fig7_real_format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="e.g. _fast for the smoke run")
    args = ap.parse_args()
    out = RESULTS / "figures" / f"inference{args.suffix}"
    out.mkdir(parents=True, exist_ok=True)
    S, F, C, R = (load(RESULTS / f"inference_{e}{args.suffix}.csv") for e in "SFCR")
    GD = [r for r in load(RESULTS / "gd_sweep.csv") if r.get("status") == "ok"]
    not_exact = [r for r in S + F + C + R if not truthy(r["all_exact"])]
    if not_exact:
        print(f"  WARNING: {len(not_exact)} runs did not decode every score exactly")
    if R:
        fig_real_scatter(R, out)
        fig_real_format(R, out)
    if S:
        fig_error_bound(S, out)
        fig_key_length(S, out)
        fig_metrics(S, out)
    if F:
        fig_format(F, out)
    if C:
        fig_cost(C, GD, out)
    if not args.suffix:
        import shutil
        paper = RESULTS.parent / "paper" / "figures"
        paper.mkdir(parents=True, exist_ok=True)
        for pdf in out.glob("*.pdf"):
            shutil.copy2(pdf, paper / pdf.name)
        print(f"  copied PDFs -> {paper}")


if __name__ == "__main__":
    main()
