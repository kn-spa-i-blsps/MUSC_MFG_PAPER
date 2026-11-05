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
A39S = "results/amend39_configS_band/report.json"
A39G = "results/amend39_settled_regen/figure_data_amend39.csv"
LEGACY = "results/overnight_figures/figure_data.csv"
C_REAL, C_LEARN, C_NOISE = "#eb6834", "#2a78d6", "#1baf7a"   # validated light: all checks pass, contrast WARN -> direct labels
HEAD_W, HEAD_B = "5", "0.30"   # paper headline metric: settled 5 s, 0.30 m band (prereg amend39 addendum 1)


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
    a39s = load(A39S)["arms"]
    s = lambda arm: [mci(a39s[arm][f"safe_settled_{HEAD_W}s_b{HEAD_B}"])]  # noqa: E731
    groups = [("none (stale to both)", C_NONE, s("real")),
              ("MFG-only", C_MFG, s("split_noise_only-real (MFG clean, CBF delayed)")),
              ("CBF-only", C_CBF, s("split_real-noise_only (MFG delayed, CBF clean)")),
              ("both layers", C_BOTH, s("noise_only"))]
    bar_group(ax, ["N=100"], groups, rec, "fig6_layers_S_oracle_settled_band", A39S)
    style(ax, "", ""); ax.set_title("oracle, Config S, safe\nsettled 5 s + 0.30 m band", fontsize=9, color=INK)
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


def g39():
    """{(cell, metric): (mean, lo, hi)} from the amend39 report CSV, or None if amend39 has not run."""
    p = os.path.join(ROOT, A39G)
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        out[(r["cell"], r["metric"])] = (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def legacy():
    out = {}
    for r in csv.DictReader(open(os.path.join(ROOT, LEGACY))):
        m = float(r["mean"])
        out[(r["figure"], r["metric"], r["series"], float(r["x"]))] = (m, m, m)
    return out


def fig_staleness(rec):
    """Fig. 1 (RA-L): (a) safe success vs N, (b) ever-collided vs T_frame for the N-sweep and the fixed-N t_slot sweep
    (the decomposition), (c) safe success vs T_frame at fixed N=100. amend39 settled+band if available, else legacy @T."""
    G = g39()
    Ns, TF = [25, 50, 100, 150, 200], [0.25, 0.5, 0.75, 1.0]
    ts = {0.25: "ts2.50ms", 0.5: "ts5.00ms", 0.75: "ts7.50ms", 1.0: "ts10.00ms"}
    arms = [("real_d0.55", "real, d 0.55", C_REAL, "--", "s"), ("real_d0.70", "real, d 0.70", C_REAL, "-", "o"),
            ("learned_pooled_d0.70", "learned v3", C_LEARN, "-", "D"), ("noise_only_d0.70", "delay removed", C_NOISE, "-", "^")]
    if G is not None:
        hm, hb = f"safe_settled_{HEAD_W}s_b{HEAD_B}", "safe_settled_5s_b0.00"
        src, ylab = A39G, "safe settled"

        def cx(arm, N, m):
            stg, a = ("A", "amend24") if arm.startswith("noise") else ("A", "amend22")
            return G.get((f"{stg}__{a}__N{N:03d}_{arm}", m))

        def tx(arm, tf, m):
            return G.get((f"B__amend26__{ts[tf]}_{arm}", m))
    else:
        L = legacy(); hm, hb = "safe_success", None
        src, ylab = LEGACY, "safe success @T (legacy)"
        lname = {"real_d0.55": "real_d0.55", "real_d0.70": "real_d0.70", "learned_pooled_d0.70": "learned",
                 "noise_only_d0.70": "noise_only"}

        def cx(arm, N, m):
            return L.get(("fig1_crossover", m, lname[arm], float(N)))

        def tx(arm, tf, m):
            return L.get(("fig5_tslot", m, lname[arm], float(tf)))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35))

    def line(ax, xs, vals, col, ls, mk, lab, fid, hollow=False):
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if not pts:
            return
        x = [p[0] for p in pts]; m = [p[1][0] for p in pts]
        err = [[p[1][0] - p[1][1] for p in pts], [p[1][2] - p[1][0] for p in pts]]
        ax.errorbar(x, m, yerr=err, color=col, ls=ls, lw=2 if not hollow else 1.2, marker=mk, ms=6,
                    mfc="white" if hollow else col, mec=col if hollow else "white", mew=1.2 if hollow else 1,
                    elinewidth=0.8, capsize=2, zorder=3, label=lab)
        for p in pts:
            rec.append((fid, lab, p[0], p[1][0], p[1][1], p[1][2], src))
    ax = axs[0]
    for arm, lab, col, ls, mk in arms:
        line(ax, Ns, [cx(arm, N, hm) for N in Ns], col, ls, mk, lab, "fig1a_crossover")
        if hb:
            line(ax, Ns, [cx(arm, N, hb) for N in Ns], col, ":", mk, None, "fig1a_crossover_b0", hollow=True)
    style(ax, "N  (T_frame = N·5 ms)", ylab); ax.set_title("(a) N-sweep", fontsize=9, color=INK)
    ax.set_xticks(Ns); ax.set_ylim(-0.04, 1.08)
    ax = axs[1]
    line(ax, [N * 0.005 for N in Ns], [cx("learned_pooled_d0.70", N, "ever_collided") for N in Ns], C_LEARN, "-", "D",
         "learned, N-sweep", "fig1b_ever_Nsweep")
    line(ax, TF, [tx("learned_pooled_d0.70", t, "ever_collided") for t in TF], C_LEARN, "--", "D",
         "learned, N=100 fixed", "fig1b_ever_tslot", hollow=True)
    line(ax, [N * 0.005 for N in Ns], [cx("noise_only_d0.70", N, "ever_collided") for N in Ns], C_NOISE, "-", "^",
         "delay removed", "fig1b_ever_noise")
    style(ax, "T_frame [s]", "ever collided"); ax.set_title("(b) collisions track T_frame",
                                                                                fontsize=9, color=INK)
    ax.set_ylim(-0.04, 1.08); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axs[2]
    for arm, lab, col, ls, mk in arms[:3]:
        line(ax, TF, [tx(arm, t, hm) for t in TF], col, ls, mk, lab, "fig1c_tslot")
        if hb:
            line(ax, TF, [tx(arm, t, hb) for t in TF], col, ":", mk, None, "fig1c_tslot_b0", hollow=True)
    style(ax, "T_frame [s], N=100 fixed", ""); ax.set_title("(c) pure staleness", fontsize=9, color=INK)
    ax.set_xticks(TF); ax.set_ylim(-0.04, 1.08)
    h, l = axs[0].get_legend_handles_labels()
    if hb:
        h.append(plt.Line2D([], [], color=MUTED, ls=":", marker="o", mfc="white", mec=MUTED)); l.append("same, no band (b=0)")
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(OUT, "fig1_staleness.png"), dpi=300)
    plt.close(fig)
    return G is not None


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(ROOT, "results", "overnight_figures")
    for f in os.listdir(src):
        if f.endswith(".png"):
            shutil.copy(os.path.join(src, f), os.path.join(OUT, f))
    rec = []
    fig_layers(rec); fig_rim(rec); fig_sigma(rec)
    print("fig1_staleness source:", "amend39 settled+band" if fig_staleness(rec) else "LEGACY @T (amend39 not run)")
    with open(os.path.join(OUT, "figure_data_paper.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["figure", "a", "b", "c", "d", "e", "source"]); w.writerows(rec)


if __name__ == "__main__":
    main()
