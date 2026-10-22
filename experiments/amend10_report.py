"""Amendment-10 facts: learned arm with the V_goal attractor at the far edge of the goal disc (far backstop),
otherwise amend9, vs the amendment-9 and amendment-6 learned arms (paired by seed). Each arm's facts are computed
on its own scenario (r_goal, disc centre). No verdict is computed.

    python -m experiments.amend10_report --config experiments/ladder_fargoal_amend10.yaml \
        --amend9 experiments/ladder_fargoal_amend9.yaml --amend6 experiments/ladder_fargoal.yaml [--viz]

Fill-order facts (amend10 prereg): the disc is cut into three strips of equal width along x (near = inlet side,
x - x_c < -r/3; mid; far, x - x_c > r/3) and an inner disc r <= r_goal/2. Occupancy of a region = agents in it /
(RHO_GOAL(arm) * region area), RHO_GOAL(arm) = N / (pi r_goal^2), i.e. 1.0 = the region holds its share of a
uniformly filled disc.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from env.rollout import summarize
from experiments.amend7_report import disc_facts, load
from experiments.amend8_report import KEYS, METRIC_KEYS
from experiments.amend9_report import extra_facts
from experiments.ladder import setup
from experiments.smoke import ROOT, build_scenario
from viz import render

REGIONS = ("near", "mid", "far", "inner")
CURVE_DT = 1.0                  # s, sampling of the completeness / occupancy curves
COLORS = {"amend10": "#2a78d6", "amend9": "#eb6834", "amend6": "#1baf7a"}   # fixed categorical order
SNAP_T = [15.0, 25.0, 35.0, 50.0, 70.0]


def region_masks(xz, sc):
    """xz (..., 2) -> dict of boolean masks (...) for the four fill regions."""
    dx = xz[..., 0] - sc.goal[0]
    r = np.hypot(dx, xz[..., 1] - sc.goal[1])
    ind = r <= sc.r_goal
    t = sc.r_goal / 3.0
    return {"near": ind & (dx < -t), "mid": ind & (np.abs(dx) <= t), "far": ind & (dx > t),
            "inner": r <= sc.r_goal / 2.0}


def region_capacity(sc, step=0.01):
    xs = np.arange(-sc.r_goal, sc.r_goal, step) + step / 2
    X, Z = np.meshgrid(xs + sc.goal[0], xs + sc.goal[1], indexing="ij")
    rho = sc.N / (np.pi * sc.r_goal ** 2)
    return {k: float(m.sum() * step ** 2 * rho) for k, m in region_masks(np.stack([X, Z], -1), sc).items()}


def fill_facts(L, sc, dt):
    pos = L["pos"]
    cap = region_capacity(sc)
    masks = region_masks(pos, sc)
    stride = int(round(CURVE_DT / dt))
    n_last = int(10.0 / dt)
    dx = pos[..., 0] - sc.goal[0]
    r = np.hypot(dx, pos[..., 1] - sc.goal[1])
    ind = r <= sc.r_goal
    out = {"region_capacity_agents": cap}
    for k in REGIONS:
        occ = masks[k].sum(1) / cap[k]
        out[f"occ_{k}_final"] = float(occ[-1])
        out[f"occ_{k}_mean_last_10s"] = float(occ[-n_last:].mean())
        out[f"occ_{k}_curve"] = occ[::stride].tolist()
    out["completeness_curve"] = ind.mean(1)[::stride].tolist()
    out["indisc_centroid_dx_final_m"] = float(dx[-1][ind[-1]].mean()) if ind[-1].any() else None
    out["indisc_centroid_dx_mean_last_10s_m"] = float(np.nanmean(
        [dx[t][ind[t]].mean() if ind[t].any() else np.nan for t in range(len(pos) - n_last, len(pos))]))
    out["outside_final_upstream"] = int((~ind[-1] & (dx[-1] < 0)).sum())
    out["outside_final_downstream"] = int((~ind[-1] & (dx[-1] >= 0)).sum())
    return out


FILL_KEYS = ([f"occ_{k}_{w}" for k in REGIONS for w in ("final", "mean_last_10s")]
             + ["indisc_centroid_dx_final_m", "indisc_centroid_dx_mean_last_10s_m",
                "outside_final_upstream", "outside_final_downstream"])


def fill_figure(path, arms, out, seeds, positions):
    """Row 1: completeness curves (mean over seeds, thin per-seed). Row 2: near/mid/far/inner occupancy, amend10 vs
    amend9 (small multiples, one shared y). Rows 3-4: seed-0 positions at SNAP_T for amend9 and amend10."""
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.0, 0.9, 1.0, 1.0], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    for tag, _, _, _ in arms:
        cs = [np.asarray(out[tag][s]["completeness_curve"]) for s in seeds]
        n = min(len(c) for c in cs)
        t = np.arange(n) * CURVE_DT
        for c in cs:
            ax.plot(t, c[:n], color=COLORS[tag], lw=0.6, alpha=0.35)
        m = np.mean([c[:n] for c in cs], 0)
        ax.plot(t, m, color=COLORS[tag], lw=2, label=f"{tag} (final {m[-1]:.2f})")
    ax.set_xlabel("t [s]"); ax.set_ylabel("completeness (own r_goal)"); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25); ax.legend(loc="upper left", frameon=False)
    ax.set_title("completeness: mean over 5 seeds (thick), per seed (thin)")
    for j, k in enumerate(REGIONS):
        a = fig.add_subplot(gs[1, j])
        for tag in ("amend10", "amend9"):
            cs = [np.asarray(out[tag][s][f"occ_{k}_curve"]) for s in seeds]
            n = min(len(c) for c in cs)
            a.plot(np.arange(n) * CURVE_DT, np.mean([c[:n] for c in cs], 0), color=COLORS[tag], lw=2, label=tag)
        a.axhline(1.0, color="0.5", lw=0.8, ls=":")
        a.set_ylim(0, 1.6); a.set_title(f"occupancy: {k}", fontsize=10); a.grid(alpha=0.25)
        a.set_xlabel("t [s]")
        if j == 0:
            a.set_ylabel("agents / uniform share"); a.legend(frameon=False, fontsize=8)
        else:
            a.set_yticklabels([])
    ax_n = fig.add_subplot(gs[1, 4]); ax_n.axis("off")
    ax_n.text(0, 0.5, "strips of width 2r/3 along x\nnear = inlet side (gate)\nfar = downstream side\n"
              "inner = r <= r_goal/2\n1.0 = uniformly filled disc", fontsize=9, va="center")
    for i, tag in enumerate(("amend9", "amend10")):
        sc, pos, dt = positions[tag]
        for j, ts in enumerate(SNAP_T):
            a = fig.add_subplot(gs[2 + i, j])
            render._draw_scene(a, sc)
            for x in (-1, 1):
                a.axvline(sc.goal[0] + x * sc.r_goal / 3, color="0.7", lw=0.6, ls=":")
            k = min(int(round(ts / dt)), len(pos) - 1)
            p = pos[k]
            ind = np.hypot(*(p - np.asarray(sc.goal)).T) <= sc.r_goal
            a.scatter(p[~ind, 0], p[~ind, 1], s=7, color="0.45")
            a.scatter(p[ind, 0], p[ind, 1], s=7, color=COLORS[tag])
            a.set_xlim(sc.wall_x - 1.0, sc.arena[0])
            a.set_title(f"{tag} seed 0, t={k * dt:.0f}s, in disc {ind.sum()}", fontsize=9)
    fig.suptitle("amend10 (V_goal attractor at far disc edge, x) vs amend9 (attractor at disc centre): fill order")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--amend9", required=True)
    ap.add_argument("--amend6", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    arms = []
    for tag, path in (("amend10", a.config), ("amend9", a.amend9), ("amend6", a.amend6)):
        c = yaml.safe_load(open(path))
        arms.append((tag, c, os.path.join(ROOT, "results", c["run_name"], "learned"), build_scenario(c, c["N"])))
    cfg = arms[0][1]
    dt = cfg["dt_exec"]
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    keys = KEYS + ["completeness_at_57.8s", "agents_in_disc_mean_last_10s", "horizon_s"] + FILL_KEYS
    out = {"seeds": a.seeds, "attractor": {t: list(getattr(sc, "attractor", sc.goal)) for t, _, _, sc in arms},
           "goal": {t: list(sc.goal) for t, _, _, sc in arms}, "r_goal": {t: sc.r_goal for t, _, _, sc in arms}}
    positions = {}
    for tag, c, dr, sc in arms:
        out[tag] = {}
        for s in a.seeds:
            m = json.load(open(os.path.join(dr, f"metrics_seed{s}.json")))
            L = load(os.path.join(dr, f"logs_seed{s}.npz"))
            L["_d_safe"] = c["cbf"]["d_safe"]
            f = {k: m.get(k) for k in METRIC_KEYS}
            f.update(disc_facts(L, sc, dt))
            f.update(extra_facts(L, sc, dt))
            f.update(fill_facts(L, sc, dt))
            out[tag][s] = f
            if s == 0 and tag in ("amend9", "amend10"):
                positions[tag] = (sc, np.asarray(L["pos"]), dt)
            del L
            print(tag, s, {k: f[k] for k in ("completeness_final", "collision_pair_steps", "coll_pair_steps_in_disc",
                                             "p5_sep_worst_step_m", "occ_near_final", "occ_mid_final",
                                             "occ_far_final", "occ_inner_final")}, flush=True)
    summ = {}
    for k in keys:
        for tag, _, _, _ in arms:
            v = np.array([out[tag][s][k] if out[tag][s][k] is not None else np.nan for s in a.seeds], float)
            summ.setdefault(k, {})[tag] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                                           "sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
                                           "per_seed": v.tolist()}
        n = np.array(summ[k]["amend10"]["per_seed"])
        for ref in ("amend9", "amend6"):
            d = n - np.array(summ[k][ref]["per_seed"])
            summ[k][f"paired_diff_amend10_minus_{ref}_mean"] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    for k in ("in_disc_depth_band_counts", "all_depth_band_counts"):
        summ[k] = {tag: np.sum([out[tag][s][k] for s in a.seeds], 0).tolist() for tag, _, _, _ in arms}
    out["summary"] = summ
    json.dump(out, open(os.path.join(run_dir, "amend10_report.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

    if a.viz:
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        fill_figure(os.path.join(viz_dir, "fill_order_amend10_vs_amend9.png"), arms, out, a.seeds, positions)
        strip = lambda L: {k: v for k, v in L.items() if k not in ("xhat_rel", "cbf_nb_idx", "held")}
        sc_, grid, sol, rcfg, _ = setup(cfg)
        logs = strip(load(os.path.join(run_dir, "learned", "logs_seed0.npz")))
        _, series = summarize(logs, sc_, dt)
        render.animate(os.path.join(viz_dir, "anim_learned_seed0.gif"), sc_, logs, series, sol, dt_exec=dt,
                       title="amend10 far-goal N=100 learned, attractor at far disc edge, d_safe=0.55, seed=0")
        render.snapshots(os.path.join(viz_dir, "snapshots_learned_seed0.png"), sc_, logs, series,
                         cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon], dt_exec=dt)
        panel = {"amend10": (logs, series)}
        for tag, c, dr, sc in arms[1:]:
            z = strip(load(os.path.join(dr, "logs_seed0.npz")))
            panel[tag] = (z, summarize(z, sc, dt)[1])
        render.summary_panel(os.path.join(viz_dir, "summary_panel_amend10_vs_amend9_amend6_learned.png"), sc_,
                             panel, dt_exec=dt)
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
