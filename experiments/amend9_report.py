"""Amendment-9 facts: learned arm with the sparse goal (spacing 0.80 m, r_goal 4.20 m, taller downstream chamber),
conical V_goal, no hold, d_safe 0.55, vs the amendment-6 learned arm and the amendment-8 run (paired by seed).
Each arm's facts are computed on its own scenario (r_goal, goal). No verdict is computed.

    python -m experiments.amend9_report --config experiments/ladder_fargoal_amend9.yaml \
        --amend6 experiments/ladder_fargoal.yaml --amend8 experiments/ladder_fargoal_amend8.yaml [--viz]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from env.rollout import summarize
from experiments.amend7_report import disc_facts, load
from experiments.amend8_report import KEYS, METRIC_KEYS
from experiments.ladder import setup
from experiments.smoke import ROOT, build_scenario
from viz import render

T_REF = 57.774782049039636      # amend6/8 horizon: completeness at this time is reported for all arms
LAST_S = 10.0                   # extra steady-state window: the final 10 s of each run


def extra_facts(L, sc, dt):
    r = np.linalg.norm(L["pos"] - np.asarray(sc.goal)[None, None], axis=-1)
    ind = r <= sc.r_goal
    i_ref = min(int(round(T_REF / dt)), ind.shape[0] - 1)
    return {"completeness_at_57.8s": float(ind[i_ref].mean()),
            "agents_in_disc_mean_last_10s": float(ind[-int(LAST_S / dt):].sum(1).mean()),
            "horizon_s": float(ind.shape[0] * dt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--amend6", required=True, help="amend6 config (results/<run_name>/learned)")
    ap.add_argument("--amend8", required=True, help="amend8 config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    arms = []
    for tag, path in (("amend9", a.config), ("amend6", a.amend6), ("amend8", a.amend8)):
        c = yaml.safe_load(open(path))
        arms.append((tag, c, os.path.join(ROOT, "results", c["run_name"], "learned")))
    cfg = arms[0][1]
    dt = cfg["dt_exec"]
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    keys = KEYS + ["completeness_at_57.8s", "agents_in_disc_mean_last_10s", "horizon_s"]
    out = {"seeds": a.seeds}
    for tag, c, dr in arms:
        sc = build_scenario(c, c["N"])
        out[tag] = {}
        for s in a.seeds:
            m = json.load(open(os.path.join(dr, f"metrics_seed{s}.json")))
            L = load(os.path.join(dr, f"logs_seed{s}.npz"))
            L["_d_safe"] = c["cbf"]["d_safe"]
            f = {k: m.get(k) for k in METRIC_KEYS}
            f.update(disc_facts(L, sc, dt))
            f.update(extra_facts(L, sc, dt))
            out[tag][s] = f
            del L
            print(tag, s, {k: f[k] for k in ("completeness_final", "collision_pair_steps",
                                             "coll_pair_steps_in_disc", "p5_sep_worst_step_m")}, flush=True)
    summ = {}
    for k in keys:
        for tag, _, _ in arms:
            v = np.array([out[tag][s][k] if out[tag][s][k] is not None else np.nan for s in a.seeds], float)
            summ.setdefault(k, {})[tag] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                                           "sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
                                           "per_seed": v.tolist()}
        n = np.array(summ[k]["amend9"]["per_seed"])
        for ref in ("amend6", "amend8"):
            d = n - np.array(summ[k][ref]["per_seed"])
            summ[k][f"paired_diff_amend9_minus_{ref}_mean"] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    for k in ("in_disc_depth_band_counts", "all_depth_band_counts"):
        summ[k] = {tag: np.sum([out[tag][s][k] for s in a.seeds], 0).tolist() for tag, _, _ in arms}
    out["summary"] = summ
    json.dump(out, open(os.path.join(run_dir, "amend9_report.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

    if a.viz:
        strip = lambda L: {k: v for k, v in L.items() if k not in ("xhat_rel", "cbf_nb_idx", "held")}
        sc_, grid, sol, rcfg, _ = setup(cfg)
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        logs = strip(load(os.path.join(run_dir, "learned", "logs_seed0.npz")))
        _, series = summarize(logs, sc_, dt)
        render.animate(os.path.join(viz_dir, "anim_learned_seed0.gif"), sc_, logs, series, sol, dt_exec=dt,
                       title="amend9 far-goal N=100 learned sparse goal (0.80 m) d_safe=0.55, seed=0")
        render.snapshots(os.path.join(viz_dir, "snapshots_learned_seed0.png"), sc_, logs, series,
                         cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon], dt_exec=dt)
        # time series of the reference arms are computed on their own scenario (r_goal 3.15)
        panel = {"amend9": (logs, series)}
        for tag, c, dr in arms[1:]:
            z = strip(load(os.path.join(dr, "logs_seed0.npz")))
            panel[tag] = (z, summarize(z, build_scenario(c, c["N"]), dt)[1])
        render.summary_panel(os.path.join(viz_dir, "summary_panel_amend9_vs_amend6_amend8_learned.png"), sc_,
                             panel, dt_exec=dt)
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
