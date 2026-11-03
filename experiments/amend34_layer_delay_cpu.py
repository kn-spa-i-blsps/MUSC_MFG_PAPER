"""Amendment-34 (EXPLORATORY): through which layer does TDMA delay do its damage? CPU, N=100, amend11 config, d_safe 0.55,
seeds 0-4. Arms (existing amend27 split code, split:<mfg>/<cbf>):
  A = split:real/noise_only  (delayed -> MFG, delay-free -> CBF)
  B = split:noise_only/real  (delay-free -> MFG, delayed -> CBF)
References: real = amend11 real, noise_only = amend33 noise_only (seeds 0-4); one-seed check via the split path.

    python -m experiments.amend34_layer_delay_cpu write-cells
    python -m experiments.amend34_layer_delay_cpu report
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os

import numpy as np
import yaml

from control.params import D_COLLISION
from experiments.smoke import ROOT, build_scenario

BASE = os.path.join(ROOT, "experiments", "ladder_fargoal_amend11.yaml")
CELL_DIR = os.path.join(ROOT, "experiments", "amend34_cells")
RUN = "amend34_layer_delay_cpu"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend34_layer_delay_cpu.md")
SEEDS = [0, 1, 2, 3, 4]
DIRS = {"real": os.path.join(ROOT, "results", "ladder_fargoal_amend11_N100", "real"),
        "noise_only": os.path.join(ROOT, "results", "amend33_n100_baseline_cpu", "noise_only", "noise_only"),
        "A_cbf_fresh": os.path.join(RES, "ab", "split_real-noise_only"),
        "B_mfg_fresh": os.path.join(RES, "ab", "split_noise_only-real")}
CHECK = {"real": (os.path.join(RES, "check_real", "split_real-real"), DIRS["real"]),
         "noise_only": (os.path.join(RES, "ab", "split_noise_only-noise_only"), DIRS["noise_only"])}
IDENTITY_KEYS = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
                 "cbf_intervention_rate", "slack_active_rate", "integral_W_true")
DT = 0.01
MOVE_WINDOW_S, MOVE_MIN_M, GATE_R_M = 5.0, 0.5, 3.0
N_BOOT = 10000


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for name, log_xhat in (("ab", False), ("check_real", True)):
        c = copy.deepcopy(yaml.safe_load(open(BASE)))
        c["run_name"] = f"{RUN}/{name}"
        c["log_xhat"] = log_xhat           # logging only; check_real keeps amend11's True
        with open(os.path.join(CELL_DIR, f"{name}.yaml"), "w") as f:
            f.write(f"# AMEND34 cell {name}: ladder_fargoal_amend11.yaml with run_name/log_xhat changed only\n")
            yaml.safe_dump(c, f, sort_keys=False)
    print("cells written to", CELL_DIR)


def seed_facts(d, s, sc):
    z = np.load(os.path.join(d, f"logs_seed{s}.npz"))
    pos = np.concatenate([z["pos"], z["pos_final"][None]], 0).astype(np.float64)   # (T+1,N,2)
    T1, N, _ = pos.shape
    iu, ju = np.triu_indices(N, 1)
    collided = np.zeros(N, bool)
    for t0 in range(0, T1, 1000):
        p = pos[t0:t0 + 1000]
        hit = (np.linalg.norm(p[:, iu] - p[:, ju], axis=-1) < D_COLLISION).any(0)
        collided[iu[hit]] = True
        collided[ju[hit]] = True
    m = json.load(open(os.path.join(d, f"metrics_seed{s}.json")))
    arrived = np.linalg.norm(pos[-1] - np.asarray(sc.goal), axis=-1) <= sc.r_goal
    w = int(round(MOVE_WINDOW_S / DT))
    moving = np.linalg.norm(pos[-1] - pos[-1 - w], axis=-1) >= MOVE_MIN_M
    near_gate = np.linalg.norm(pos[-1] - np.array([sc.wall_x, sc.gate_z]), axis=-1) <= GATE_R_M
    upstream = pos[-1, :, 0] < sc.wall_x
    na = ~arrived
    other = na & ~moving & ~near_gate
    return {"seed": s,
            "fleet_success": int(arrived.all() and m["collision_pair_steps"] == 0),
            "reached_goal": float(arrived.mean()),
            "drones_ever_collided": int(collided.sum()),
            "collision_pair_steps": int(m["collision_pair_steps"]),
            "makespan_s": m["makespan_s"], "t90_s": m["t_first_90pct"],
            "n_not_arrived": int(na.sum()),
            "na_still_moving": int((na & moving).sum()),
            "na_jammed_gate": int((na & ~moving & near_gate).sum()),
            "na_other": int(other.sum()),
            "na_other_upstream": int((other & upstream).sum()),
            "na_other_downstream": int((other & ~upstream).sum()),
            "check_completeness_equal": bool(abs(arrived.mean() - m["completeness_final"]) < 1e-12)}


def signflip(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    stats = [abs((np.array(sg) * d).mean()) for sg in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([x >= obs - 1e-12 for x in stats]))


def contrast(ra, rb, key):
    da = {r["seed"]: r[key] for r in ra if r[key] is not None}
    db = {r["seed"]: r[key] for r in rb if r[key] is not None}
    common = sorted(set(da) & set(db))
    if not common:
        return None
    x = np.array([da[s] - db[s] for s in common], float)
    b = x[np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return {"mean": float(x.mean()), "per_seed": x.tolist(), "n": len(x),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "signflip_p": signflip(x), "signflip_min_p": 2.0 / 2 ** len(x)}


def check(new_dir, ref_dir):
    if not os.path.exists(os.path.join(new_dir, "metrics_seed0.json")):
        return None
    n, r = (json.load(open(os.path.join(x, "metrics_seed0.json"))) for x in (new_dir, ref_dir))
    zn, zr = (np.load(os.path.join(x, "logs_seed0.npz")) for x in (new_dir, ref_dir))
    pos_eq = bool(np.array_equal(zn["pos"], zr["pos"]) and np.array_equal(zn["pos_final"], zr["pos_final"]))
    keys = {k: [r[k], n[k]] for k in IDENTITY_KEYS}
    return {"pos_bit_equal": pos_eq, "keys_equal": all(a == b for a, b in keys.values()), "keys": keys}


def cmd_report(a):
    cfg = yaml.safe_load(open(BASE))
    sc = build_scenario(cfg, cfg["N"])
    R = {arm: [seed_facts(d, s, sc) for s in SEEDS if os.path.exists(os.path.join(d, f"metrics_seed{s}.json"))]
         for arm, d in DIRS.items()}
    keys = ("fleet_success", "reached_goal", "drones_ever_collided", "collision_pair_steps", "t90_s", "makespan_s")
    out = {"label": "EXPLORATORY", "aneks_sha256": hashlib.sha256(open(ANEKS, "rb").read()).hexdigest(),
           "seeds": SEEDS, "check": {k: check(*v) for k, v in CHECK.items()}, "per_seed": R,
           "means": {arm: {k: (float(np.mean([r[k] for r in R[arm] if r[k] is not None]))
                               if any(r[k] is not None for r in R[arm]) else None) for k in keys} for arm in R}}
    out["contrasts"] = {f"{x}_minus_{y}": {k: contrast(R[x], R[y], k) for k in keys}
                        for x, y in (("A_cbf_fresh", "real"), ("A_cbf_fresh", "noise_only"),
                                     ("B_mfg_fresh", "real"), ("B_mfg_fresh", "noise_only"))}
    M = out["means"]
    rec = {}
    for arm in ("A_cbf_fresh", "B_mfg_fresh"):
        rec[arm] = {}
        for k in ("drones_ever_collided", "collision_pair_steps", "reached_goal", "fleet_success"):
            gap = M["noise_only"][k] - M["real"][k] if M[arm][k] is not None else None
            rec[arm][k] = (M[arm][k] - M["real"][k]) / gap if gap else None
    out["recovered_fraction"] = rec
    cA = out["contrasts"]["B_mfg_fresh_minus_real"]["reached_goal"]
    rA, rB = rec["A_cbf_fresh"], rec["B_mfg_fresh"]
    ok = lambda v: v is not None and v >= 0.9
    out["readings"] = {
        "A_close_to_noise_only_on_collisions": ok(rA["drones_ever_collided"]) and ok(rA["collision_pair_steps"]),
        "B_moves_reached_goal_toward_noise_only": bool(cA and len(cA["per_seed"]) == 5 and all(v > 0 for v in cA["per_seed"])),
        "neither_alone_recovers": not any(ok(r["drones_ever_collided"]) and ok(r["reached_goal"]) for r in (rA, rB))}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    print("EXPLORATORY. check:", json.dumps({k: (v and {"pos": v["pos_bit_equal"], "keys": v["keys_equal"]}) for k, v in out["check"].items()}))
    for arm in R:
        for r in R[arm]:
            print(f"{arm:12s} s{r['seed']} fleet={r['fleet_success']} reached={r['reached_goal']:.2f} "
                  f"ever_coll={r['drones_ever_collided']:3d} coll_ps={r['collision_pair_steps']:6d} "
                  f"t90={r['t90_s']} makespan={r['makespan_s']} not_arr={r['n_not_arrived']} "
                  f"(moving {r['na_still_moving']}, gate {r['na_jammed_gate']}, other {r['na_other']}: "
                  f"up {r['na_other_upstream']} / down {r['na_other_downstream']})")
    for name, c in out["contrasts"].items():
        for k, v in c.items():
            if v:
                print(f"{name:28s} {k:22s} {v['mean']:+10.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                      f"p_signflip={v['signflip_p']:.4f} per_seed={[round(x, 3) for x in v['per_seed']]}")
    print("recovered fraction:", json.dumps(rec))
    print("readings:", json.dumps(out["readings"]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("write-cells")
    sp.add_parser("report")
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
