"""Paper figures 6-8 (draft_v1) from committed report.json files only - no new runs.
Also copies the existing overnight figures (fig1-5) into paper/figs so the paper is self-contained.

    python -m viz.paper_figures

Writes paper/figs/*.png and paper/figs/figure_data_paper.csv (every plotted number, with its source file).
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots (validated with the dataviz validator, light mode: all checks pass; contrast WARN -> direct labels)
C_NONE, C_MFG, C_CBF, C_BOTH = "#eb6834", "#6250d6", "#eda100", "#1baf7a"
C_FIXED, C_SIGMA = "#2a78d6", "#eb6834"

A27 = "results/amend27_ladder_layers/report.json"
A38 = "results/amend38_settled_recompute/report.json"
A35 = "results/amend35_sigma_cap/report.json"


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9); ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def bar_group(ax, xs, groups, rec, fig_id, src):
    """groups: list of (label, color, [(mean, lo, hi) per x])."""
    k = len(groups); w = 0.8 / k
    for gi, (lab, col, vals) in enumerate(groups):
        pos = np.arange(len(xs)) + (gi - (k - 1) / 2) * w
        m = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        ax.bar(pos, m, width=w - 0.02, color=col, edgecolor="white", linewidth=1, label=lab, zorder=2)
        ax.errorbar(pos, m, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        for p, v in zip(pos, m):
            ax.text(p, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
        for x, v in zip(xs, vals):
            rec.append((fig_id, lab, x, v[0], v[1], v[2], src))
    ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels(xs); ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def mci(r):
    return (r["mean"], r["ci95"][0], r["ci95"][1])


def fig_layers(rec):
    a27, a38 = load(A27)["rows"], load(A38)["A2"]
    Ns = ["100", "150", "200"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [3, 3, 1.3]})
    for ax, est, title in ((axs[0], "learned", "learned v3 (Config G, safe@T)"),
                           (axs[1], "noise_only", "oracle = delay removed (Config G, safe@T)")):
        ss = lambda arm: [mci(a27[N][arm]["safe_success"]) for N in Ns]  # noqa: E731
        both = "learned_pooled_d0.70" if est == "learned" else "noise_only_d0.70"
        groups = [("none (stale to both)", C_NONE, ss("real_d0.70")),
                  ("MFG-only", C_MFG, ss(f"split_{est}-real_d0.70")),      # split_<MFG src>-<CBF src>
                  ("CBF-only", C_CBF, ss(f"split_real-{est}_d0.70")),
                  ("both layers", C_BOTH, ss(both))]
        bar_group(ax, [f"N={N}" for N in Ns], groups, rec, f"fig6_layers_G_{est}", A27)
        style(ax, "", "safe success"); ax.set_title(title, fontsize=9, color=INK)
    ax = axs[2]
    s = lambda arm: [mci(a38[arm]["safe_settled_5s"])]  # noqa: E731
    groups = [("none (stale to both)", C_NONE, s("real")),
              ("MFG-only", C_MFG, s("split_noise_only-real (MFG clean, CBF delayed)")),
              ("CBF-only", C_CBF, s("split_real-noise_only (MFG delayed, CBF clean)")),
              ("both layers", C_BOTH, s("noise_only"))]
    bar_group(ax, ["N=100"], groups, rec, "fig6_layers_S_oracle_settled", A38)
    style(ax, "", ""); ax.set_title("oracle, Config S,\nsafe settled 5 s", fontsize=9, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig6_layer_split.png"), dpi=200)
    plt.close(fig)


def fig_rim(rec):
    B = load(A38)["B"]
    classes = ["oscillating", "parked_MFG_stalled", "parked_CBF_blocked", "parked_other", "in_transit", "other"]
    names = {"oscillating": "oscillating", "parked_MFG_stalled": "parked, MFG-stalled",
             "parked_CBF_blocked": "parked, CBF-blocked", "parked_other": "parked, other",
             "in_transit": "in transit", "other": "other"}
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, arm, title in ((axs[0], "fresh", "exact positions (fresh), n=40"),
                           (axs[1], "noise_only", "sensor noise (noise_only), n=100")):
        ag = B[arm]["agents"]
        for yi, c in enumerate(classes):
            d = [a["d_T"] for a in ag if a["class"] == c]
            y = yi + rng.uniform(-0.18, 0.18, len(d))
            ax.scatter(d, y, s=14, color=C_FIXED, edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(0.62, yi, f"n={len(d)}", va="center", fontsize=7, color=MUTED)
            for a in ag:
                if a["class"] == c:
                    rec.append(("fig7_rim", arm, c, a["seed"], a["agent"], a["d_T"], A38))
        ax.axvline(0, color=INK, lw=1); ax.axvline(0.30, color=MUTED, lw=1, ls="--")
        ax.text(0.0, -0.75, " rim", fontsize=7, color=INK)
        ax.text(0.30, -0.75, " +0.30 m", fontsize=7, color=MUTED)
        ax.set_xlim(-0.65, 0.75); ax.set_ylim(-1.0, len(classes) - 0.5); ax.set_yticks(range(len(classes))); ax.set_yticklabels([names[c] for c in classes])
        ax.invert_yaxis(); style(ax, "distance to rim at T [m] (>0 outside)", "")
        ax.grid(True, axis="x", color=GRID, lw=0.6); ax.grid(False, axis="y")
        ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_rim_nonsettlers.png"), dpi=200); plt.close(fig)


def fig_sigma(rec):
    R = load(A35)["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)
    fixed = ["fixed0.70", "fixed0.85", "fixed1.00", "fixed1.20"]
    sig = ["sigma_a0.1", "sigma_a0.05", "sigma_a0.01", "sigma_bin_a0.05"]
    slab = {"sigma_a0.1": "α 0.10", "sigma_a0.05": "α 0.05", "sigma_a0.01": "α 0.01", "sigma_bin_a0.05": "primary"}
    for ax, N in ((axs[0], "150"), (axs[1], "200")):
        def xy(a):
            r = R[N][a]; return mci(r["completeness"]), mci(r["safe_success"])
        pts = [xy(a) for a in fixed]
        ax.plot([p[0][0] for p in pts], [p[1][0] for p in pts], color=C_FIXED, lw=2, zorder=2,
                label="constant margin d_safe")
        for a, (x, y) in zip(fixed, pts):
            ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                        fmt="o", ms=6, color=C_FIXED, mec="white", mew=1, elinewidth=0.8, capsize=2, zorder=3)
            off = (-26, -12) if (N == "150" and a == "fixed0.70") else (4, -10)
            ax.annotate(a.replace("fixed", ""), (x[0], y[0]), xytext=off, textcoords="offset points",
                        fontsize=7, color=INK)
            rec.append(("fig8_sigma", N, a, x[0], y[0], "", A35))
        for a in sig:
            x, y = xy(a)
            prim = a == "sigma_bin_a0.05"
            ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                        fmt="D" if prim else "o", ms=8 if prim else 6, color=C_SIGMA,
                        mfc=C_SIGMA if prim else "white", mec=C_SIGMA if not prim else "white", mew=1.2,
                        elinewidth=0.8, capsize=2, zorder=4, label="σ-CBF (capped), primary" if prim else None)
            if prim:
                ax.annotate("σ primary", (x[0], y[0]), xytext=(-62, 6), textcoords="offset points", fontsize=7,
                            color=INK)
            rec.append(("fig8_sigma", N, a, x[0], y[0], "", A35))
        style(ax, "completeness@T", "safe success@T" if N == "150" else "")
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_title(f"N={N}", fontsize=9, color=INK); ax.set_xlim(0.2, 0.95); ax.set_ylim(-0.03, 0.62)
    axs[0].plot([], [], "o", color=C_SIGMA, mfc="white", label="σ-CBF, α 0.10 / 0.05 / 0.01")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(OUT, "fig8_sigma_frontier.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(ROOT, "results", "overnight_figures")
    for f in os.listdir(src):
        if f.endswith(".png"):
            shutil.copy(os.path.join(src, f), os.path.join(OUT, f))
    rec = []
    fig_layers(rec); fig_rim(rec); fig_sigma(rec)
    with open(os.path.join(OUT, "figure_data_paper.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["figure", "a", "b", "c", "d", "e", "source"]); w.writerows(rec)


if __name__ == "__main__":
    main()
