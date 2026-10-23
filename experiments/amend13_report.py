"""Amendment-13 facts: amendment-12 safe success as a function of the crash threshold c, on the EXISTING
amendment-11 ladder logs (no new runs). collided_i(c): per-agent min true separation over the rollout < c.
No verdict is computed.

    python -m experiments.amend13_report --config experiments/ladder_fargoal_amend11.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from experiments.smoke import ROOT, build_scenario

ARMS = ["real", "cv", "learned", "smith"]
THRESHOLDS = [0.30, 0.25, 0.20, 0.15, 0.10]
LEVELS = [0.5, 0.9]


def agent_minsep(pos):
    """Per-agent min distance to any other agent over all states. pos: (T,N,2)."""
    T, N, _ = pos.shape
    iu, ju = np.triu_indices(N, 1)
    ms = np.full(N, np.inf)
    for t0 in range(0, T, 1000):
        dmin = np.linalg.norm(pos[t0:t0 + 1000, iu] - pos[t0:t0 + 1000, ju], axis=-1).min(0)   # (P,)
        np.minimum.at(ms, iu, dmin)
        np.minimum.at(ms, ju, dmin)
    return ms


def mean_sd(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "per_seed": v.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = build_scenario(cfg, cfg["N"])
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    a12 = json.load(open(os.path.join(run_dir, "amend12_safe_success.json")))
    out = {"seeds": a.seeds, "thresholds_m": THRESHOLDS, "r_goal": float(sc.r_goal), "per_agent": {}, "summary": {},
           "grid_crossing": {}, "continuous_crossing": {}, "check_c0.30_vs_amend12": {}}
    for arm in ARMS:
        arr, ms = [], []
        for s in a.seeds:
            z = np.load(os.path.join(run_dir, arm, f"logs_seed{s}.npz"))
            pos = np.concatenate([z["pos"], z["pos_final"][None]], 0).astype(np.float64)
            arr.append(np.linalg.norm(pos[-1] - np.asarray(sc.goal), axis=-1) <= sc.r_goal)
            ms.append(agent_minsep(pos))
            print(arm, s, "done", flush=True)
        arr, ms = np.array(arr), np.array(ms)                                     # (S,N)
        out["per_agent"][arm] = {"arrived": arr.astype(int).tolist(), "minsep_m": ms.tolist()}
        ss = lambda c: (arr & ~(ms < c)).mean(1)
        out["summary"][arm] = {f"{c:.2f}": {"safe_success": mean_sd(ss(c)), "ever_collided": mean_sd((ms < c).mean(1))}
                               for c in THRESHOLDS}
        out["summary"][arm]["completeness_old"] = mean_sd(arr.mean(1))
        out["check_c0.30_vs_amend12"][arm] = [ss(0.30).tolist(),
                                              [a12["arms"][arm][str(s)]["safe_success"] for s in a.seeds]]
        out["grid_crossing"][arm] = {str(L): max([c for c in THRESHOLDS if ss(c).mean() >= L], default=None)
                                     for L in LEVELS}
        # continuous: mean safe success is a step function of c that jumps at the minsep values of arrived agents
        cand = np.unique(ms[arr])
        cont = {}
        for L in LEVELS:
            ok = [c for c in cand if ss(c).mean() >= L]                           # at c = minsep the agent is safe
            cont[str(L)] = float(max(ok)) if ok else None
        out["continuous_crossing"][arm] = cont
    json.dump(out, open(os.path.join(run_dir, "amend13_threshold_curve.json"), "w"), indent=1)

    for key in ("safe_success", "ever_collided"):
        print(f"\n{key} (mean ± sd)\n{'arm':8s}" + "".join(f"{f'c={c:.2f}':>16s}" for c in THRESHOLDS))
        for arm in ARMS:
            print(f"{arm:8s}" + "".join(f"{out['summary'][arm][f'{c:.2f}'][key]['mean']:>9.3f} ± "
                                        f"{out['summary'][arm][f'{c:.2f}'][key]['sd']:.3f}" for c in THRESHOLDS))
    print("\nper seed safe_success:")
    for arm in ARMS:
        print(arm, {f"{c:.2f}": out["summary"][arm][f"{c:.2f}"]["safe_success"]["per_seed"] for c in THRESHOLDS})
    print("\ncompleteness_old:", {arm: out["summary"][arm]["completeness_old"]["mean"] for arm in ARMS})
    print("grid crossing:", out["grid_crossing"])
    print("continuous crossing:", out["continuous_crossing"])
    print("check c=0.30 vs amend12:", out["check_c0.30_vs_amend12"])


if __name__ == "__main__":
    main()
