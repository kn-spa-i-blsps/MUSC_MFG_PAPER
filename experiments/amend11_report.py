"""Amendment-11 facts: the full ladder {real, cv, learned, smith} on the amendment-9 safe config (sparse goal,
d_safe 0.55, conical V_goal to the disc centre, no hold). Per-arm table (mean ± sd, per seed), paired bootstrap
over seeds of arm − real (amend6 statistics), open-loop MSE (amend6 definition, this geometry), and viz of the
best and worst arm by the rule fixed in the amendment. No verdict is computed.

    python -m experiments.amend11_report --config experiments/ladder_fargoal_amend11.yaml \
        --amend9 experiments/ladder_fargoal_amend9.yaml [--viz]

Leading safety metric (amend11): worst-step p5 NN separation and the deep-collision rate, i.e. true pair-steps with
d < 0.10 m per control step (overall and in the disc), always read next to completeness.
Best arm: among the arms whose mean completeness is >= mean completeness of real − 0.03 (the amend6 tolerance), the
highest mean worst-step p5 (ties: lower deep-collision rate). Worst arm: over all arms, the lowest mean worst-step p5
(ties: higher deep-collision rate).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from env.rollout import summarize
from experiments.amend7_report import BANDS, disc_facts, load
from experiments.amend9_report import extra_facts
from experiments.ladder import setup
from experiments.ladder_report import boot_ci
from experiments.smoke import ROOT, build_scenario
from viz import render

ARMS = ["real", "cv", "learned", "smith"]
METRIC_KEYS = ["completeness_final", "t_first_90pct", "n_outside_r_goal_final", "collision_pair_steps",
               "min_sep_overall_m", "p5_sep_worst_step_m", "cbf_intervention_rate", "slack_active_rate",
               "mean_speed_mps", "closedloop_xhat_mse_near", "closedloop_xhat_mse_all", "qp_nan_fallbacks"]
DERIVED = ["coll_pair_steps_in_disc", "deep_pair_steps", "deep_pair_steps_in_disc", "deep_rate_per_step",
           "deep_rate_per_step_in_disc", "slack_active_rate_in_disc", "slack_active_rate_outside",
           "in_disc_coll_qp_feasible_both", "agents_in_disc_mean_t_ge_28s", "agents_in_disc_mean_last_10s",
           "completeness_at_57.8s"]
PAIRED = ["completeness_final", "collision_pair_steps", "coll_pair_steps_in_disc", "deep_pair_steps",
          "p5_sep_worst_step_m", "min_sep_overall_m", "cbf_intervention_rate", "slack_active_rate"]
COMPL_TOL = 0.03


def pick_best_worst(summ):
    m = lambda arm, k: summ[k][arm]["mean"]
    ok = [a for a in ARMS if m(a, "completeness_final") >= m("real", "completeness_final") - COMPL_TOL]
    best = max(ok, key=lambda a: (m(a, "p5_sep_worst_step_m"), -m(a, "deep_rate_per_step")))
    worst = min(ARMS, key=lambda a: (m(a, "p5_sep_worst_step_m"), -m(a, "deep_rate_per_step")))
    return best, worst, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--amend9", required=True, help="amend9 config: its learned arm is the reproducibility check")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dt = cfg["dt_exec"]
    sc = build_scenario(cfg, cfg["N"])
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    out = {"seeds": a.seeds, "bands_m": BANDS, "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for s in a.seeds:
            m = json.load(open(os.path.join(run_dir, arm, f"metrics_seed{s}.json")))
            L = load(os.path.join(run_dir, arm, f"logs_seed{s}.npz"))
            L["_d_safe"] = cfg["cbf"]["d_safe"]
            f = {k: m.get(k) for k in METRIC_KEYS}
            f.update(disc_facts(L, sc, dt))
            f.update(extra_facts(L, sc, dt))
            n_steps = L["pos"].shape[0]
            f["deep_pair_steps"] = f["all_depth_band_counts"][0]
            f["deep_pair_steps_in_disc"] = f["in_disc_depth_band_counts"][0]
            f["deep_rate_per_step"] = f["deep_pair_steps"] / n_steps
            f["deep_rate_per_step_in_disc"] = f["deep_pair_steps_in_disc"] / n_steps
            out["arms"][arm][s] = f
            del L
            print(arm, s, {k: f[k] for k in ("completeness_final", "collision_pair_steps", "coll_pair_steps_in_disc",
                                             "deep_pair_steps", "p5_sep_worst_step_m")}, flush=True)
    summ = {}
    for k in METRIC_KEYS + DERIVED:
        for arm in ARMS:
            v = np.array([np.nan if out["arms"][arm][s][k] is None else out["arms"][arm][s][k] for s in a.seeds], float)
            summ.setdefault(k, {})[arm] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                                           "sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
                                           "per_seed": v.tolist()}
    for k in ("in_disc_depth_band_counts", "all_depth_band_counts"):
        summ[k] = {arm: np.sum([out["arms"][arm][s][k] for s in a.seeds], 0).tolist() for arm in ARMS}
    out["summary"] = summ
    pv = {}
    for arm in ARMS[1:]:
        pv[arm] = {}
        for k in PAIRED:
            d = np.array([out["arms"][arm][s][k] - out["arms"]["real"][s][k] for s in a.seeds], float)
            lo, hi = boot_ci(d)
            pv[arm][k] = {"mean_diff": float(d.mean()), "ci95": [lo, hi], "per_seed": d.tolist()}
        r, c = summ["collision_pair_steps"]["real"]["mean"], summ["collision_pair_steps"][arm]["mean"]
        pv[arm]["collision_reduction_vs_real_of_means"] = float(1.0 - c / r) if r > 0 else None
    out["paired_vs_real"] = pv
    ol = os.path.join(run_dir, "openloop_mse.json")
    out["openloop_mse"] = json.load(open(ol)) if os.path.exists(ol) else None
    # reproducibility: the learned arm here vs the amend9 learned arm (identical config and seeds)
    c9 = yaml.safe_load(open(a.amend9))
    rep = {}
    for s in a.seeds:
        m9 = json.load(open(os.path.join(ROOT, "results", c9["run_name"], "learned", f"metrics_seed{s}.json")))
        rep[s] = {k: [out["arms"]["learned"][s][k], m9.get(k)] for k in
                  ("completeness_final", "collision_pair_steps", "p5_sep_worst_step_m", "min_sep_overall_m")}
    out["learned_vs_amend9_learned_[amend11,amend9]"] = rep
    best, worst, ok = pick_best_worst(summ)
    out["viz_choice"] = {"best": best, "worst": worst, "eligible_for_best": ok}
    json.dump(out, open(os.path.join(run_dir, "amend11_report.json"), "w"), indent=1, default=float)
    print(json.dumps({"summary": summ, "paired_vs_real": pv, "viz_choice": out["viz_choice"],
                      "openloop_mse": out["openloop_mse"], "repro": rep}, indent=1, default=float))

    if a.viz:
        strip = lambda L: {k: v for k, v in L.items() if k not in ("xhat_rel", "cbf_nb_idx", "held", "t", "arrived_frac")}
        sc_, grid, sol, rcfg, _ = setup(cfg)
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        panel = {}
        for tag, arm in (("best", best), ("worst", worst)):
            if arm in panel:
                continue
            logs = strip(load(os.path.join(run_dir, arm, "logs_seed0.npz")))
            _, series = summarize(logs, sc_, dt)
            panel[arm] = (logs, series)
            render.animate(os.path.join(viz_dir, f"anim_{arm}_seed0.gif"), sc_, logs, series, sol, dt_exec=dt,
                           title=f"amend11 ladder (safe config) arm={arm} [{tag}] seed=0")
            render.snapshots(os.path.join(viz_dir, f"snapshots_{arm}_{tag}_seed0.png"), sc_, logs, series,
                             cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon], dt_exec=dt)
        for arm in ARMS:                                   # time series of every arm on the panel
            if arm not in panel:
                logs = strip(load(os.path.join(run_dir, arm, "logs_seed0.npz")))
                panel[arm] = (logs, summarize(logs, sc_, dt)[1])
        render.summary_panel(os.path.join(viz_dir, f"summary_panel_all_arms_first_{best}.png"), sc_, panel,
                             dt_exec=dt)
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
