"""Overnight paper figures (CONFIRMED cells only, see logs/overnight/review_amend*.md). Reads committed report.json files,
writes PNG + the plotted numbers as CSV to results/overnight_figures/.

    python -m viz.overnight_figures
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.smoke import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "results", "overnight_figures")
# reference categorical palette (dataviz skill, light mode), fixed slot per entity
C = {"real_d0.55": "#2a78d6", "real_d0.70": "#eb6834", "learned": "#1baf7a", "noise_only": "#eda100",
     "learned_old": "#e87ba4", "cv": "#6250d6"}
LABEL = {"real_d0.55": "real (no comp.), d_safe 0.55", "real_d0.70": "real (no comp.), d_safe 0.70",
         "learned": "learned v3 (pooled), d_safe 0.70", "noise_only": "delay removed (noise_only), d_safe 0.70",
         "learned_old": "learned, N=100-only predictor", "cv": "constant-velocity"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def series(ax, x, rows, key, ent, ms=6):
    m = [r[key]["mean"] for r in rows]
    lo = [r[key]["mean"] - r[key]["ci95"][0] for r in rows]
    hi = [r[key]["ci95"][1] - r[key]["mean"] for r in rows]
    ax.errorbar(x, m, yerr=[lo, hi], color=C[ent], lw=2, marker="o", ms=ms, capsize=3, label=LABEL[ent],
                markeredgecolor="white", markeredgewidth=1.2)
    return m


def fig_crossover(rec):
    a22, a24 = load("results/amend22_N_crossover_pooled/report.json"), load("results/amend24_mechanism_control/report.json")
    Ns = [25, 50, 100, 150, 200]
    arms = {"real_d0.55": [a22["rows"][str(N)]["real_d0.55"] for N in Ns],
            "real_d0.70": [a22["rows"][str(N)]["real_d0.70"] for N in Ns],
            "learned": [a22["rows"][str(N)]["learned_pooled_d0.70"] for N in Ns],
            "noise_only": [a24["rows"][str(N)]["noise_only"] for N in Ns]}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, yl in zip(axs, ("safe_success", "ever_collided", "completeness"),
                           ("safe-success (reached goal, never d < 0.30 m)", "ever d < 0.30 m (fraction of agents)",
                            "completeness (fraction in goal at end)")):
        for ent, rows in arms.items():
            m = series(ax, Ns, rows, key, ent)
            for N, v in zip(Ns, m):
                rec.append(("fig1_crossover", key, ent, N, v))
        style(ax, "swarm size N  (T_frame = N · 5 ms)", yl); ax.set_ylim(-0.03, 1.05); ax.set_xticks(Ns)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle("N crossover on a fixed arena (43.4 × 19.1 m, gate 2.31 m), 10 seeds, 95% seed-bootstrap CI",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, "fig1_N_crossover.png"), dpi=160); plt.close(fig)


def fig_mechanism(rec):
    ol3 = load("results/overnight_p1/openloop_v3.json")["per_N"]
    ol1 = load("results/overnight_p1/openloop_v1.json")["per_N"]
    Ns = [25, 50, 100, 150, 200]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ent, src, k in (("real_d0.55", ol3, "real"), ("cv", ol3, "cv"), ("learned_old", ol1, "learned_old"),
                        ("learned", ol3, "learned_new")):
        v = [src[str(N)][k]["mse_near1m_m2"] for N in Ns]
        ax.plot(Ns, v, color=C[ent], lw=2, marker="o", ms=6, markeredgecolor="white", markeredgewidth=1.2,
                label={"real_d0.55": "stale (real, no compensation)", "cv": "constant-velocity",
                       "learned_old": "learned, N=100-only predictor", "learned": "learned v3, pooled over N"}[ent])
        for N, y in zip(Ns, v):
            rec.append(("fig2_openloop_mse", "mse_near1m_m2", ent, N, y))
    ax.set_yscale("log"); ax.set_xticks(Ns)
    style(ax, "swarm size N  (T_frame = N · 5 ms)", "open-loop x̂ MSE, pairs < 1 m  [m², log]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Estimation error vs N (held-out real-arm histories)", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_mechanism_openloop.png"), dpi=160); plt.close(fig)


def fig_margin(rec):
    a23 = load("results/amend23_margin_pooled/report.json")
    ds = [0.55, 0.70, 0.85, 1.00]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, N in zip(axs, ("100", "150")):
        for key, col, lab in (("safe_success", C["learned"], "safe-success"), ("ever_collided", C["real_d0.70"],
                                                                                "ever d < 0.30 m"),
                              ("completeness", C["real_d0.55"], "completeness")):
            rows = [a23["rows"][N][f"{d:.2f}"] for d in ds]
            m = [r[key]["mean"] for r in rows]
            ax.errorbar(ds, m, yerr=[[r[key]["mean"] - r[key]["ci95"][0] for r in rows],
                                     [r[key]["ci95"][1] - r[key]["mean"] for r in rows]],
                        color=col, lw=2, marker="o", ms=6, capsize=3, label=lab, markeredgecolor="white",
                        markeredgewidth=1.2)
            for d, v in zip(ds, m):
                rec.append(("fig3_margin", key, f"N{N}", d, v))
        style(ax, "fixed CBF margin d_safe [m]", "fraction of agents" if N == "100" else "")
        ax.set_title(f"learned v3, N = {N}", color=INK, fontsize=10); ax.set_xticks(ds); ax.set_ylim(-0.03, 1.05)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_margin_tradeoff.png"), dpi=160); plt.close(fig)


def fig_sweep(rec, rep, name, xkey, xlabel, fname, title):
    if not os.path.exists(os.path.join(ROOT, rep)):
        return
    r = load(rep)
    tags = list(r["rows"])
    x = [xkey(t) for t in tags]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, yl in zip(axs, ("safe_success", "ever_collided"), ("safe-success", "ever d < 0.30 m")):
        for arm, ent in (("real_d0.55", "real_d0.55"), ("real_d0.70", "real_d0.70"), ("learned_pooled_d0.70", "learned")):
            m = series(ax, x, [r["rows"][t][arm] for t in tags], key, ent)
            for xx, v in zip(x, m):
                rec.append((name, key, ent, xx, v))
        style(ax, xlabel, yl); ax.set_ylim(-0.03, 1.05); ax.set_xticks(x)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, fname), dpi=160); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = []
    fig_crossover(rec)
    fig_mechanism(rec)
    fig_margin(rec)
    fig_sweep(rec, "results/amend25_gate_sweep/report.json", "fig4_gate", lambda t: float(t[1:]),
              "gate width [m]  (N = 100)", "fig4_density_gate_sweep.png", "Operating density via gate width, N = 100")
    fig_sweep(rec, "results/amend26_tslot_sweep/report.json", "fig5_tslot", lambda t: 100 * float(t[2:-2]) / 1000,
              "T_frame = 100 · t_slot [s]  (N = 100, fixed geometry)", "fig5_staleness_tslot_sweep.png",
              "Pure staleness: TDMA slot length at fixed N and density")
    with open(os.path.join(OUT, "figure_data.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["figure", "metric", "series", "x", "mean"]); w.writerows(rec)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
