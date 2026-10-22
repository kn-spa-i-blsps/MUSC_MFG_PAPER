"""Ladder facts (prereg amend6): per-arm table (mean ± sd, per seed), paired bootstrap over seeds of
arm − real, and viz of chosen arms (seed 0). No verdict is computed.

    python -m experiments.ladder_report --config experiments/ladder_fargoal.yaml [--viz learned real]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import yaml

from env.rollout import summarize
from experiments.ladder import setup
from experiments.smoke import ROOT
from viz import render

ARMS = ["real", "cv", "learned", "smith"]
KEYS = ["completeness_final", "arrived_frac_max", "t_first_90pct", "collision_pair_steps", "collision_steps",
        "min_sep_overall_m", "p5_sep_worst_step_m", "p5_sep_median_over_time_m", "cbf_intervention_rate",
        "cbf_intervention_mean_mps", "slack_active_rate", "slack_max", "slack_hard_max", "mean_speed_mps",
        "closedloop_xhat_mse_near", "closedloop_xhat_mse_all", "n_outside_r_goal_final",
        "frac_left_goal_after_arrival", "crossed_wall_final", "qp_nan_fallbacks"]


def boot_ci(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, d.size), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--viz", nargs="*", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    M = {}
    for arm in ARMS:
        rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run_dir, arm, "metrics_seed*.json")))]
        M[arm] = {r["seed"]: r for r in rows}
    seeds = sorted(set.intersection(*[set(v) for v in M.values()]))
    out = {"seeds": seeds, "arms": {}, "paired_vs_real": {}}
    for arm in ARMS:
        st = {}
        for k in KEYS:
            v = np.array([np.nan if M[arm][s].get(k) is None else M[arm][s][k] for s in seeds], float)
            st[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                     "per_seed": v.tolist()}
        out["arms"][arm] = st
    for arm in ARMS[1:]:
        pv = {}
        for k in ["collision_pair_steps", "completeness_final", "p5_sep_worst_step_m", "min_sep_overall_m",
                  "cbf_intervention_rate", "slack_active_rate"]:
            d = np.array([M[arm][s][k] - M["real"][s][k] for s in seeds], float)
            lo, hi = boot_ci(d)
            pv[k] = {"mean_diff": float(d.mean()), "ci95": [lo, hi], "per_seed": d.tolist()}
        r = np.array([M["real"][s]["collision_pair_steps"] for s in seeds], float)
        c = np.array([M[arm][s]["collision_pair_steps"] for s in seeds], float)
        pv["collision_reduction_vs_real_of_means"] = float(1.0 - c.mean() / r.mean())
        out["paired_vs_real"][arm] = pv
    ol = os.path.join(run_dir, "openloop_mse.json")
    out["openloop_mse"] = json.load(open(ol)) if os.path.exists(ol) else None
    json.dump(out, open(os.path.join(run_dir, "ladder_report.json"), "w"), indent=1)
    show = ["completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
            "cbf_intervention_rate", "slack_active_rate", "closedloop_xhat_mse_near", "n_outside_r_goal_final"]
    for arm in ARMS:
        print(arm, {k: (round(out["arms"][arm][k]["mean"], 4), round(out["arms"][arm][k]["sd"], 4)) for k in show})
    for arm in ARMS[1:]:
        print("vs real", arm, json.dumps(out["paired_vs_real"][arm]))

    if a.viz:
        sc, grid, sol, rcfg, _ = setup(cfg)
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        panel = {}
        for arm in a.viz:
            z = dict(np.load(os.path.join(run_dir, arm, "logs_seed0.npz")))
            logs = {k: v for k, v in z.items() if k not in ("t", "arrived_frac")}
            _, series = summarize(logs, sc, cfg["dt_exec"])
            panel[arm] = (logs, series)
            render.animate(os.path.join(viz_dir, f"anim_{arm}_seed0.gif"), sc, logs, series, sol,
                           dt_exec=cfg["dt_exec"], title=f"ladder far-goal N=100 arm={arm} seed=0")
            render.snapshots(os.path.join(viz_dir, f"snapshots_{arm}_seed0.png"), sc, logs, series,
                             cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon],
                             dt_exec=cfg["dt_exec"])
        render.summary_panel(os.path.join(viz_dir, "summary_panel_" + "_".join(a.viz) + ".png"), sc, panel,
                             dt_exec=cfg["dt_exec"])
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
