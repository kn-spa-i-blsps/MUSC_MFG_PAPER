"""Amendment-7 facts: learned arm with d_safe 0.55 + flat-bottomed goal + hold-on-arrival vs the
amendment-6 learned arm (paired by seed), in-disc collisions, depth, QP infeasibility, worst-case p5,
and the direct per-pair attribution from the logged x_hat. No verdict is computed.

    python -m experiments.amend7_report --config experiments/ladder_fargoal_amend7.yaml \
        --baseline results/ladder_fargoal_N100/learned [--viz]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from control.params import D_COLLISION
from env.rollout import summarize
from experiments.ladder import setup
from experiments.smoke import ROOT, build_scenario
from viz import render

BANDS = [0.0, 0.10, 0.20, 0.25, 0.30]
SLACK_ON = 1e-4


def disc_facts(L, sc, dt):
    """In-disc collision facts from TRUE positions; per-pair attribution when x_hat was logged."""
    pos = L["pos"]
    T, N, _ = pos.shape
    iu, ju = np.triu_indices(N, 1)
    r = np.linalg.norm(pos - np.asarray(sc.goal)[None, None], axis=-1)
    ind = r <= sc.r_goal
    act = L["slack_max"] > SLACK_ON
    d = np.linalg.norm(pos[:, iu] - pos[:, ju], axis=-1)
    coll = d < D_COLLISION
    cd = coll & ind[:, iu] & ind[:, ju]
    tc, pc = np.nonzero(cd)
    a, b = iu[pc], ju[pc]
    out = {
        "coll_pair_steps": int(coll.sum()),
        "coll_pair_steps_in_disc": int(cd.sum()),
        "share_in_disc": float(cd.sum() / max(coll.sum(), 1)),
        "in_disc_depth_band_counts": np.histogram(d[cd], BANDS)[0].tolist(),
        "in_disc_d_quantiles_[.1,.5,.9]": np.quantile(d[cd], [.1, .5, .9]).tolist() if cd.any() else None,
        "all_depth_band_counts": np.histogram(d[coll], BANDS)[0].tolist(),
        "slack_active_rate_in_disc": float(act[ind].mean()),
        "slack_active_rate_outside": float(act[~ind].mean()),
        "in_disc_coll_qp_feasible_both": float((~(act[tc, a] | act[tc, b])).mean()) if cd.any() else None,
        "coll_steady_share_t_ge_28s": float((tc * dt >= 28.0).mean()) if cd.any() else None,
        "agents_in_disc_mean_t_ge_28s": float(ind[int(28.0 / dt):].sum(1).mean()),
    }
    if "held" in L:
        out["held_rate_in_disc"] = float(L["held"][ind].mean())
    if "xhat_rel" in L:
        out["attribution_all"] = attribution(L, coll, d, act, iu, ju)
        out["attribution_in_disc"] = attribution(L, cd, d, act, iu, ju)
        onset = coll.copy()
        onset[1:] &= ~coll[:-1]
        on_prev = np.zeros_like(onset)
        on_prev[:-1] = onset[1:]                      # the step BEFORE each onset
        out["attribution_step_before_onset"] = attribution(L, on_prev, d, act, iu, ju)
    return out


def attribution(L, mask, d, act, iu, ju, d_safe=None):
    d_safe = L["_d_safe"] if d_safe is None else d_safe
    tc, pc = np.nonzero(mask)
    if tc.size == 0:
        return {"n": 0}
    a, b = iu[pc], ju[pc]
    xr = L["xhat_rel"]
    dh_ab = np.linalg.norm(xr[tc, a, b].astype(np.float32), axis=-1)   # a's perceived distance to b
    dh_ba = np.linalg.norm(xr[tc, b, a].astype(np.float32), axis=-1)
    dmin = np.minimum(dh_ab, dh_ba)
    infeas = act[tc, a] | act[tc, b]
    est = dmin >= d_safe
    nb = L["cbf_nb_idx"].astype(np.int64)
    a_has_b = (nb[tc, a] == b[:, None]).any(1)
    b_has_a = (nb[tc, b] == a[:, None]).any(1)
    err = np.r_[dh_ab, dh_ba] - np.r_[d[tc, pc], d[tc, pc]]
    return {
        "n": int(tc.size),
        "estimate_error_dhat_both_ge_dsafe": float(est.mean()),
        "estimate_error_and_qp_feasible": float((est & ~infeas).mean()),
        "perceived_inside_barrier_qp_feasible": float((~est & ~infeas).mean()),
        "qp_infeasible_either": float(infeas.mean()),
        "qp_infeasible_and_perceived_inside": float((~est & infeas).mean()),
        "partner_not_in_cbf_set_either": float((~a_has_b | ~b_has_a).mean()),
        "dhat_min_quantiles_[.1,.25,.5,.75,.9]": np.quantile(dmin, [.1, .25, .5, .75, .9]).tolist(),
        "dhat_minus_d_quantiles_[.1,.25,.5,.75,.9]": np.quantile(err, [.1, .25, .5, .75, .9]).tolist(),
    }


def load(path):
    z = dict(np.load(path))
    return {k: v for k, v in z.items() if k not in ("t", "arrived_frac")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = build_scenario(cfg, cfg["N"])
    dt = cfg["dt_exec"]
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    base_dir = os.path.join(ROOT, a.baseline)
    keys = ["completeness_final", "t_first_90pct", "collision_pair_steps", "min_sep_overall_m",
            "p5_sep_worst_step_m", "slack_active_rate", "cbf_intervention_rate", "mean_speed_mps",
            "closedloop_xhat_mse_near", "n_outside_r_goal_final"]
    out = {"seeds": a.seeds, "new": {}, "baseline": {}}
    for tag, dr, d_safe in (("new", os.path.join(run_dir, "learned"), cfg["cbf"]["d_safe"]),
                            ("baseline", base_dir, 0.40)):
        for s in a.seeds:
            m = json.load(open(os.path.join(dr, f"metrics_seed{s}.json")))
            L = load(os.path.join(dr, f"logs_seed{s}.npz"))
            L["_d_safe"] = d_safe
            f = {k: m.get(k) for k in keys}
            f.update(disc_facts(L, sc, dt))
            out[tag][s] = f
            print(tag, s, {k: f[k] for k in ("completeness_final", "collision_pair_steps",
                                             "coll_pair_steps_in_disc", "p5_sep_worst_step_m")}, flush=True)
    summ = {}
    for k in keys + ["coll_pair_steps_in_disc", "slack_active_rate_in_disc", "in_disc_coll_qp_feasible_both"]:
        for tag in ("new", "baseline"):
            v = np.array([out[tag][s][k] if out[tag][s][k] is not None else np.nan for s in a.seeds], float)
            summ.setdefault(k, {})[tag] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                                           "per_seed": v.tolist()}
        n, b = (np.array(summ[k][t]["per_seed"]) for t in ("new", "baseline"))
        summ[k]["paired_diff_new_minus_baseline_mean"] = float(np.nanmean(n - b))
    out["summary"] = summ
    json.dump(out, open(os.path.join(run_dir, "amend7_report.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

    if a.viz:
        sc_, grid, sol, rcfg, _ = setup(cfg)
        viz_dir = render.ensure_dir(os.path.join(run_dir, "viz"))
        L = load(os.path.join(run_dir, "learned", "logs_seed0.npz"))
        logs = {k: v for k, v in L.items() if k not in ("xhat_rel", "cbf_nb_idx", "held")}
        _, series = summarize(logs, sc_, dt)
        render.animate(os.path.join(viz_dir, "anim_learned_seed0.gif"), sc_, logs, series, sol, dt_exec=dt,
                       title="amend7 far-goal N=100 learned d_safe=0.55 flat goal + hold, seed=0")
        render.snapshots(os.path.join(viz_dir, "snapshots_learned_seed0.png"), sc_, logs, series,
                         cfg.get("snapshot_times", [0, 5, 10, 15, 20, 30, 40]) + [grid.horizon], dt_exec=dt)
        z = load(os.path.join(base_dir, "logs_seed0.npz"))
        _, sb = summarize(z, sc_, dt)
        render.summary_panel(os.path.join(viz_dir, "summary_panel_amend7_vs_amend6_learned.png"), sc_,
                             {"amend7": (logs, series), "amend6": (z, sb)}, dt_exec=dt)
        print("[viz]", viz_dir)


if __name__ == "__main__":
    main()
