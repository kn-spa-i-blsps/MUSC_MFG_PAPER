"""Amendment-8 facts: learned arm with d_safe reverted to 0.40 (flat goal + hold-on-arrival kept) vs the
amendment-6 learned arm and the amendment-7 run (paired by seed). Same per-seed facts as amend7_report.
No verdict is computed.

    python -m experiments.amend8_report --config experiments/ladder_fargoal_amend8.yaml \
        --amend6 results/ladder_fargoal_N100/learned --amend7 results/ladder_fargoal_amend7_N100/learned [--viz]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from env.rollout import summarize
from experiments.amend7_report import disc_facts, load
from experiments.ladder import setup
from experiments.smoke import ROOT, build_scenario
from viz import render

KEYS = ["completeness_final", "t_first_90pct", "n_outside_r_goal_final", "agents_in_disc_mean_t_ge_28s",
        "collision_pair_steps", "coll_pair_steps_in_disc", "min_sep_overall_m", "p5_sep_worst_step_m",
        "cbf_intervention_rate", "slack_active_rate", "slack_active_rate_in_disc", "slack_active_rate_outside",
        "in_disc_coll_qp_feasible_both", "mean_speed_mps", "closedloop_xhat_mse_near"]
METRIC_KEYS = ["completeness_final", "t_first_90pct", "collision_pair_steps", "min_sep_overall_m",
               "p5_sep_worst_step_m", "slack_active_rate", "cbf_intervention_rate", "mean_speed_mps",
               "closedloop_xhat_mse_near", "n_outside_r_goal_final"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--amend6", required=True)
    ap.add_argument("--amend7", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = build_scenario(cfg, cfg["N"])
    dt = cfg["dt_exec"]
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    arms = (("amend8", os.path.join(run_dir, "learned"), cfg["cbf"]["d_safe"]),
            ("amend6", os.path.join(ROOT, a.amend6), 0.40),
            ("amend7", os.path.join(ROOT, a.amend7), 0.55))
    out = {"seeds": a.seeds}
    for tag, dr, d_safe in arms:
        out[tag] = {}
        for s in a.seeds:
            m = json.load(open(os.path.join(dr, f"metrics_seed{s}.json")))
            L = load(os.path.join(dr, f"logs_seed{s}.npz"))
            L["_d_safe"] = d_safe
            f = {k: m.get(k) for k in METRIC_KEYS}
            f.update(disc_facts(L, sc, dt))
            out[tag][s] = f
            del L
            print(tag, s, {k: f[k] for k in ("completeness_final", "collision_pair_steps",
                                             "coll_pair_steps_in_disc", "p5_sep_worst_step_m")}, flush=True)
    summ = {}
    for k in KEYS:
        for tag, _, _ in arms:
            v = np.array([out[tag][s][k] if out[tag][s][k] is not None else np.nan for s in a.seeds], float)
            summ.setdefault(k, {})[tag] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                                           "sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
                                           "per_seed": v.tolist()}
        n = np.array(summ[k]["amend8"]["per_seed"])
        for ref in ("amend6", "amend7"):
            d = n - np.array(summ[k][ref]["per_seed"])
            summ[k][f"paired_diff_amend8_minus_{ref}_mean"] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    for k in ("in_disc_depth_band_counts", "all_depth_band_counts"):
        summ[k] = {tag: np.sum([out[tag][s][k] for s in a.seeds], 0).tolist() for tag, _, _ in arms}
    out["summary"] = summ
    json.dump(out, open(os.path.join(run_dir, "amend8_report.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

    if a.viz:
        sc_, grid, sol, rcfg, _ = setup(cfg)
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        strip = lambda L: {k: v for k, v in L.items() if k not in ("xhat_rel", "cbf_nb_idx", "held")}
        logs = strip(load(os.path.join(run_dir, "learned", "logs_seed0.npz")))
        _, series = summarize(logs, sc_, dt)
        render.animate(os.path.join(viz_dir, "anim_learned_seed0.gif"), sc_, logs, series, sol, dt_exec=dt,
                       title="amend8 far-goal N=100 learned d_safe=0.40 flat goal + hold, seed=0")
        render.snapshots(os.path.join(viz_dir, "snapshots_learned_seed0.png"), sc_, logs, series,
                         cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon], dt_exec=dt)
        panel = {"amend8": (logs, series)}
        for tag, dr, _ in arms[1:]:
            z = strip(load(os.path.join(dr, "logs_seed0.npz")))
            panel[tag] = (z, summarize(z, sc_, dt)[1])
        render.summary_panel(os.path.join(viz_dir, "summary_panel_amend8_vs_amend6_amend7_learned.png"), sc_,
                             panel, dt_exec=dt)
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
