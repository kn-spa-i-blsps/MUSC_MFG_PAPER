"""Amendment-12 facts: per-agent safe success on the EXISTING amendment-11 ladder logs (no new runs).
arrived_i: in r_goal at the final state; collided_i: true distance < D_COLLISION to any other agent at any state of
the rollout (pos + pos_final). Safe success = arrived & ~collided. No verdict is computed.

    python -m experiments.amend12_report --config experiments/ladder_fargoal_amend11.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from control.params import D_COLLISION
from experiments.smoke import ROOT, build_scenario

ARMS = ["real", "cv", "learned", "smith"]
KEYS = ["safe_success", "arrived_safe", "arrived_collided", "not_arrived_safe", "not_arrived_collided",
        "ever_collided", "completeness_old"]


def agent_facts(z, sc):
    pos = np.concatenate([z["pos"], z["pos_final"][None]], 0)               # (T+1,N,2)
    T1, N, _ = pos.shape
    iu, ju = np.triu_indices(N, 1)
    collided = np.zeros(N, bool)
    pair_steps_T = 0
    for t0 in range(0, T1, 1000):
        p = pos[t0:t0 + 1000]
        c = np.linalg.norm(p[:, iu] - p[:, ju], axis=-1) < D_COLLISION     # (t,P)
        pair_steps_T += int(c[: max(0, min(1000, T1 - 1 - t0))].sum())       # exclude pos_final, as summarize
        hit = c.any(0)
        collided[iu[hit]] = True
        collided[ju[hit]] = True
    arrived = np.linalg.norm(pos[-1] - np.asarray(sc.goal), axis=-1) <= sc.r_goal
    f = {"safe_success": (arrived & ~collided).mean(), "arrived_safe": (arrived & ~collided).mean(),
         "arrived_collided": (arrived & collided).mean(), "not_arrived_safe": (~arrived & ~collided).mean(),
         "not_arrived_collided": (~arrived & collided).mean(), "ever_collided": collided.mean(),
         "completeness_old": arrived.mean()}
    return {k: float(v) for k, v in f.items()}, pair_steps_T, arrived, collided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = build_scenario(cfg, cfg["N"])
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    out = {"seeds": a.seeds, "D_COLLISION": D_COLLISION, "r_goal": float(sc.r_goal), "arms": {}, "per_agent": {}}
    for arm in ARMS:
        out["arms"][arm], out["per_agent"][arm] = {}, {}
        for s in a.seeds:
            z = np.load(os.path.join(run_dir, arm, f"logs_seed{s}.npz"))
            z = {"pos": z["pos"].astype(np.float64), "pos_final": z["pos_final"].astype(np.float64)}
            f, ps, arrived, collided = agent_facts(z, sc)
            m = json.load(open(os.path.join(run_dir, arm, f"metrics_seed{s}.json")))
            f["check_pair_steps_[recomputed,metrics]"] = [ps, m["collision_pair_steps"]]
            f["check_completeness_[recomputed,metrics]"] = [f["completeness_old"], m["completeness_final"]]
            out["arms"][arm][s] = f
            out["per_agent"][arm][s] = {"arrived": arrived.astype(int).tolist(),
                                        "collided": collided.astype(int).tolist()}
            print(arm, s, f, flush=True)
    summ = {}
    for k in KEYS:
        for arm in ARMS:
            v = np.array([out["arms"][arm][s][k] for s in a.seeds])
            summ.setdefault(k, {})[arm] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                           "per_seed": v.tolist()}
    out["summary"] = summ
    json.dump(out, open(os.path.join(run_dir, "amend12_safe_success.json"), "w"), indent=1)
    print(f"\n{'arm':8s} {'safe_success':>16s} {'ever_collided':>16s} {'completeness_old':>18s}")
    for arm in ARMS:
        c = lambda k: f"{summ[k][arm]['mean']:.3f} ± {summ[k][arm]['sd']:.3f}"
        print(f"{arm:8s} {c('safe_success'):>16s} {c('ever_collided'):>16s} {c('completeness_old'):>18s}")
    print("\nbreakdown (mean ± sd): arrived&safe / arrived&collided / not_arrived&safe / not_arrived&collided")
    for arm in ARMS:
        print(arm, " / ".join(f"{summ[k][arm]['mean']:.3f}±{summ[k][arm]['sd']:.3f}" for k in KEYS[1:5]))


if __name__ == "__main__":
    main()
