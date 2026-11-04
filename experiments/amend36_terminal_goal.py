"""Amendment-36: MFG terminal cost = goal potential, N=100 amend11 config, delay-free arms, seeds 0-4 (CPU).
Stage 1, link 1 ("delay-free works"). References: amend33 rows (terminal cost 0), same seeds.

    python -m experiments.amend36_terminal_goal write-cells
    python -m experiments.amend36_terminal_goal report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import yaml

from experiments import amend33_n100_baseline_cpu as a33
from experiments.amend12_report import agent_facts
from experiments.smoke import ROOT, build_scenario

RUN = "amend36_terminal_goal"
RES = os.path.join(ROOT, "results", RUN)
CELL_DIR = os.path.join(ROOT, "experiments", "amend36_cells")
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend36_terminal_goal.md")
ARMS = {"fresh": "fresh", "noise_only": "noise_only", "fresh_nomfg": "fresh"}   # cell -> --arm
SEEDS = a33.SEEDS
LAST = 20                                    # 0.2 s at dt_exec 0.01


def cell_cfg(arm):
    c = a33.cell_cfg(arm)
    c["macro_cost"]["terminal"] = "goal"
    c["run_name"] = f"{RUN}/{arm}"
    return c


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for arm in ARMS:
        with open(os.path.join(CELL_DIR, f"{arm}.yaml"), "w") as f:
            f.write(f"# AMEND36 cell {arm}: amend33 cell + macro_cost.terminal goal, run_name\n")
            yaml.safe_dump(cell_cfg(arm), f, sort_keys=False)
    print("cells written to", CELL_DIR)


def rows(d, sc):
    out = []
    for s in SEEDS:
        p = os.path.join(d, f"logs_seed{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        pos, pT = z["pos"].astype(np.float64), z["pos_final"].astype(np.float64)
        f, _, _, _ = agent_facts({"pos": pos, "pos_final": pT}, sc)
        m = json.load(open(os.path.join(d, f"metrics_seed{s}.json")))
        f.update({k: m.get(k) for k in a33.EXTRA}, seed=s, completeness=m["completeness_final"])
        g = np.asarray(sc.goal)
        outT = np.linalg.norm(pT - g, axis=-1) > sc.r_goal
        in_before = np.linalg.norm(pos[-LAST] - g, axis=-1) <= sc.r_goal
        f["n_out_T"] = int(outT.sum())
        f["n_left_last_0p2s"] = int((outT & in_before).sum())
        out.append(f)
    return out


def cmd_report(a):
    cfg = yaml.safe_load(open(a33.BASE))
    sc = build_scenario(cfg, cfg["N"])
    new = {arm: rows(os.path.join(RES, arm, ARMS[arm]), sc) for arm in ARMS}
    old = {arm: rows(a33.arm_dir(arm), sc) for arm in ARMS}
    keys = ("safe_success", "ever_collided", "completeness", "n_out_T", "n_left_last_0p2s") + a33.EXTRA
    summ = lambda R: {"n_seeds": len(R), "per_seed": {k: [r[k] for r in R] for k in keys},
                      **{k: a33.boot_ci([r[k] for r in R if r[k] is not None]) for k in keys}}
    out = {"label": "REGISTERED (amend36)", "aneks_sha256": hashlib.sha256(open(ANEKS, "rb").read()).hexdigest(),
           "seeds": SEEDS, "new_terminal_goal": {k: summ(v) for k, v in new.items()},
           "old_amend33_terminal0": {k: summ(v) for k, v in old.items()}, "contrasts": {}, "link1": {}}
    for key in ("completeness", "safe_success", "collision_pair_steps", "makespan_s", "integral_W_true", "n_out_T"):
        out["contrasts"][key] = {f"{arm}_new_minus_old": a33.paired(new[arm], old[arm], key) for arm in ARMS}
        out["contrasts"][key]["fresh_minus_nomfg_new"] = a33.paired(new["fresh"], new["fresh_nomfg"], key)
    for arm in ("fresh", "noise_only"):
        R = new[arm]
        c = [r["completeness"] for r in R]
        coll0 = all(r["collision_pair_steps"] == 0 for r in R)
        d = out["contrasts"]["completeness"][f"{arm}_new_minus_old"]
        if len(R) == len(SEEDS) and coll0 and np.mean(c) >= 0.99 and min(c) >= 0.98:
            v = "HOLDS"
        elif len(R) == len(SEEDS) and coll0 and d and d["ci95"][0] > 0:
            v = "IMPROVED, NOT MET"
        else:
            v = "FAIL"
        out["link1"][arm] = {"verdict": v, "completeness_mean": float(np.mean(c)) if c else None,
                             "completeness_min": float(min(c)) if c else None, "collisions_zero_all_seeds": coll0,
                             "n_seeds": len(R)}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    g = lambda r, k: r[k]["mean"] if r[k] else float("nan")
    print(f"{'arm':22s} {'n':>2s} {'safe':>6s} {'ever':>6s} {'compl':>6s} {'out_T':>6s} {'left.2s':>7s} {'coll_ps':>8s} "
          f"{'makespan':>8s} {'intW':>7s}")
    for tag, blk in (("new", out["new_terminal_goal"]), ("old", out["old_amend33_terminal0"])):
        for arm, r in blk.items():
            print(f"{tag + ' ' + arm:22s} {r['n_seeds']:2d} {g(r, 'safe_success'):6.3f} {g(r, 'ever_collided'):6.3f} "
                  f"{g(r, 'completeness'):6.3f} {g(r, 'n_out_T'):6.1f} {g(r, 'n_left_last_0p2s'):7.1f} "
                  f"{g(r, 'collision_pair_steps'):8.0f} {g(r, 'makespan_s'):8.1f} {g(r, 'integral_W_true'):7.1f}")
    for key in ("completeness", "safe_success"):
        for name, v in out["contrasts"][key].items():
            if v:
                print(f"{key:13s} {name:26s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")
    print("link 1:", json.dumps(out["link1"]))
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "report"])
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
