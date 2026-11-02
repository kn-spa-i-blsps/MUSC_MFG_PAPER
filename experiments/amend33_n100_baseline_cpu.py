"""Amendment-33: CPU baseline validation + layer ablations at N=100 on the locked amend11 config (d_safe 0.55,
seeds 0-4). The CPU-feasible counterpart of amend28 (whose amend21 geometry does not fit in 30 GB RAM).

New arms (this run):  fresh, fresh_nocbf, fresh_nomfg, noise_only  (+ real seed 0 = identity vs amend11)
Reused arms (amend11, same CPU, same config): real, cv, learned, smith (seeds 0-4)

    python -m experiments.amend33_n100_baseline_cpu write-cells
    python -m experiments.amend33_n100_baseline_cpu report
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os

import numpy as np
import yaml

from experiments.amend12_report import agent_facts
from experiments.smoke import ROOT, build_scenario

BASE = os.path.join(ROOT, "experiments", "ladder_fargoal_amend11.yaml")
CELL_DIR = os.path.join(ROOT, "experiments", "amend33_cells")
RUN = "amend33_n100_baseline_cpu"
RES = os.path.join(ROOT, "results", RUN)
A11 = os.path.join(ROOT, "results", "ladder_fargoal_amend11_N100")
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend33_n100_baseline_cpu.md")
SEEDS = [0, 1, 2, 3, 4]
NEW = {"fresh": "fresh", "fresh_nocbf": "fresh", "fresh_nomfg": "fresh", "noise_only": "noise_only"}
OLD = ("real", "cv", "learned", "smith")
IDENTITY_KEYS = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
                 "cbf_intervention_rate", "slack_active_rate", "integral_W_true")
EXTRA = ("collision_pair_steps", "integral_W_true", "makespan_s", "mean_time_to_goal_s", "mean_speed_mps",
         "cbf_intervention_rate", "p5_sep_worst_step_m")
N_BOOT = 10000


def cell_cfg(arm):
    c = copy.deepcopy(yaml.safe_load(open(BASE)))
    c["run_name"] = f"{RUN}/{arm}"
    if arm == "identity_real":
        return c                                   # byte-for-byte the amend11 config, log_xhat kept
    c["log_xhat"] = False                          # logging only (disk); no effect on dynamics
    if arm == "fresh_nocbf":
        c["cbf_on"] = False
    if arm == "fresh_nomfg":
        c["macro_cost"]["c_w"] = 0.0
        c["macro_cost"]["w_bar"] = 0.0
        c["control"]["c_dens"] = 0.0
    return c


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for arm in ("identity_real",) + tuple(NEW):
        with open(os.path.join(CELL_DIR, f"{arm}.yaml"), "w") as f:
            f.write(f"# AMEND33 cell {arm}: ladder_fargoal_amend11.yaml with run_name and the arm's keys changed\n")
            yaml.safe_dump(cell_cfg(arm), f, sort_keys=False)
    print("cells written to", CELL_DIR)


def arm_dir(arm):
    if arm in OLD:
        return os.path.join(A11, arm)
    if arm == "identity_real":
        return os.path.join(RES, "identity_real", "real")
    return os.path.join(RES, arm, NEW[arm])


def rows(arm, sc):
    out = []
    for s in SEEDS:
        p = os.path.join(arm_dir(arm), f"logs_seed{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        f, _, _, _ = agent_facts({"pos": z["pos"].astype(np.float64), "pos_final": z["pos_final"].astype(np.float64)}, sc)
        m = json.load(open(os.path.join(arm_dir(arm), f"metrics_seed{s}.json")))
        f.update({k: m.get(k) for k in EXTRA}, seed=s, completeness=m["completeness_final"])
        out.append(f)
    return out


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(0)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": len(x)}


def paired(ra, rb, key):
    """mean over common seeds of a - b, seed-level bootstrap CI."""
    da = {r["seed"]: r[key] for r in ra if r[key] is not None}
    db = {r["seed"]: r[key] for r in rb if r[key] is not None}
    common = sorted(set(da) & set(db))
    return boot_ci([da[s] - db[s] for s in common])


def cmd_report(a):
    cfg = yaml.safe_load(open(BASE))
    sc = build_scenario(cfg, cfg["N"])
    arms = tuple(NEW) + OLD
    R = {arm: rows(arm, sc) for arm in arms}
    keys = ("safe_success", "ever_collided", "completeness") + EXTRA
    out = {"aneks_sha256": hashlib.sha256(open(ANEKS, "rb").read()).hexdigest(), "seeds": SEEDS,
           "arms": {arm: {"n_seeds": len(R[arm]), **{k: boot_ci([r[k] for r in R[arm] if r[k] is not None]) for k in keys}}
                    for arm in arms}}
    # identity: current code, real seed 0, vs amend11 real seed 0
    pi = os.path.join(arm_dir("identity_real"), "metrics_seed0.json")
    if os.path.exists(pi):
        new = json.load(open(pi))
        old = json.load(open(os.path.join(A11, "real", "metrics_seed0.json")))
        out["identity"] = {"all_equal": all(new[k] == old[k] for k in IDENTITY_KEYS),
                           "keys": {k: [old[k], new[k]] for k in IDENTITY_KEYS}}
    q = {}
    for key in ("safe_success", "ever_collided", "completeness", "collision_pair_steps", "integral_W_true", "makespan_s"):
        q[key] = {"Q2_nocbf_minus_fresh": paired(R["fresh_nocbf"], R["fresh"], key),
                  "Q3_fresh_minus_nomfg": paired(R["fresh"], R["fresh_nomfg"], key),
                  "Q4_real_minus_noise_only": paired(R["real"], R["noise_only"], key),
                  "Q5_learned_minus_real": paired(R["learned"], R["real"], key),
                  "noise_only_minus_fresh": paired(R["noise_only"], R["fresh"], key)}
    out["contrasts"] = q
    ss = out["arms"]
    try:
        gap = ss["noise_only"]["safe_success"]["mean"] - ss["real"]["safe_success"]["mean"]
        out["Q5_recovered_fraction_safe_success"] = (ss["learned"]["safe_success"]["mean"] - ss["real"]["safe_success"]["mean"]) / gap if gap else None
    except (TypeError, KeyError):
        out["Q5_recovered_fraction_safe_success"] = None
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    print(f"{'arm':12s} {'n':>2s} {'safe_succ':>10s} {'ever_coll':>10s} {'compl':>7s} {'coll_ps':>9s} {'intW':>7s} {'makespan':>8s}")
    for arm in arms:
        r = ss[arm]
        if not r["n_seeds"]:
            print(f"{arm:12s}  0"); continue
        g = lambda k: r[k]["mean"] if r[k] else float("nan")
        print(f"{arm:12s} {r['n_seeds']:2d} {g('safe_success'):10.3f} {g('ever_collided'):10.3f} {g('completeness'):7.3f} "
              f"{g('collision_pair_steps'):9.0f} {g('integral_W_true'):7.1f} {g('makespan_s'):8.1f}")
    if "identity" in out:
        print("identity real seed0 vs amend11:", out["identity"]["all_equal"])
    for key in ("safe_success", "completeness"):
        for name, v in q[key].items():
            if v:
                print(f"{key:13s} {name:26s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("write-cells")
    sp.add_parser("report")
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
