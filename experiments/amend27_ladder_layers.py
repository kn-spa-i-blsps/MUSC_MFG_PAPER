"""Amendment-27: compensation-method ladder + MFG/CBF layer attribution on the amend21 crossover geometry (pooled v3),
precision=highest, seeds 0-9. Reads the amend22 (real d0.55/0.70, learned_pooled d0.70) and amend24 (noise_only,
fresh N150/200) rows from their committed report.json files (read-only) and adds:
  A  method ladder   cv, smith x d_safe {0.55, 0.70} x N {25..200}; fresh d0.70 x N {25, 50, 100}
  B  layer split     split:<mfg>/<cbf> x N {100, 150, 200}, d0.70: learned/real (compensation in the MFG layer only),
                     real/learned (CBF layer only), noise_only/real and real/noise_only (oracle per layer)
  I  identity gate   N=100 seed 0 d0.70: real and split:real/real vs the committed amend22 real_d0.70 seed 0 metrics

    python -m experiments.amend27_ladder_layers write-cells
    python -m experiments.amend27_ladder_layers identity
    python -m experiments.amend27_ladder_layers report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import IDENTITY_KEYS, METRICS, run_rows, safe_N, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend27_cells")
RUN = "amend27_ladder_layers"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend27_ladder_layers.md")
A22 = os.path.join(ROOT, "results", "amend22_N_crossover_pooled", "report.json")
A24 = os.path.join(ROOT, "results", "amend24_mechanism_control", "report.json")
NS = (25, 50, 100, 150, 200)
SPLIT_NS = (100, 150, 200)
SPLITS = ("learned/real", "real/learned", "noise_only/real", "real/noise_only")
SEEDS = list(range(10))
EXTRA = ("closedloop_xhat_mse_near_pairw", "near_pair_steps")
CELLS = ([(N, arm, d) for N in NS for arm in ("cv", "smith") for d in (0.55, 0.70)]
         + [(N, "fresh", 0.70) for N in (25, 50, 100)]
         + [(N, f"split:{s}", 0.70) for N in SPLIT_NS for s in SPLITS])


def tag(arm):
    return arm.replace(":", "_").replace("/", "-")


def cell_name(N, arm, d):
    return f"N{N:03d}_{tag(arm)}_d{d:.2f}"


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N, arm, d in CELLS + [(100, "split:real/real", 0.70), (100, "real", 0.70)]:
        c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
        c["cbf"]["d_safe"] = d
        c["run_name"] = f"{RUN}/N{N:03d}/{tag(arm)}_d{d:.2f}"
        c["amend27_arm"] = arm
        sub = "identity" if (arm in ("real", "split:real/real")) else ""
        os.makedirs(os.path.join(CELL_DIR, sub), exist_ok=True)
        p = os.path.join(CELL_DIR, sub, f"{cell_name(N, arm, d)}.yaml")
        with open(p, "w") as f:
            f.write(f"# AMEND27 cell {cell_name(N, arm, d)} (arm {arm}): amend21_cells/N{N:03d}_real.yaml with "
                    f"cbf.d_safe={d}, run_name and amend27_arm (read by the chain only) changed\n")
            yaml.safe_dump(c, f, sort_keys=False)
        print(p)


def rdir(N, arm, d):
    return os.path.join(ROOT, "results", RUN, f"N{N:03d}", f"{tag(arm)}_d{d:.2f}", tag(arm))


def cmd_identity(a):
    ref = json.load(open(os.path.join(ROOT, "results", "amend22_N_crossover_pooled", "N100", "real_d0.70", "real",
                                      "metrics_seed0.json")))
    out = {}
    for arm in ("real", "split:real/real"):
        m = json.load(open(os.path.join(rdir(100, arm, 0.70), "metrics_seed0.json")))
        out[arm] = {k: {"amend22": ref[k], "amend27": m[k], "equal": ref[k] == m[k]} for k in IDENTITY_KEYS}
        out[arm]["all_equal"] = all(v["equal"] for v in out[arm].values())
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "identity.json"), "w"), indent=1)
    print(json.dumps({k: v["all_equal"] for k, v in out.items()}))
    if not all(v["all_equal"] for v in out.values()):
        raise SystemExit(1)


def cmd_report(a):
    a22, a24 = json.load(open(A22)), json.load(open(A24))
    out = {"aneks_sha256": sha256(ANEKS), "matmul_precision": "highest", "seeds": a.seeds,
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "rows": {}, "diffs": {}, "safe_N": {}, "layer": {}}
    ps = lambda r, k: r[k]["per_seed"]
    for N in NS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"{cell_name(N, 'cv', 0.70)}.yaml"))), N)
        rows = {k: v for k, v in a22["rows"][str(N)].items()}
        rows.update({f"{k}_d0.70": v for k, v in a24["rows"].get(str(N), {}).items()})
        for N_, arm, d in CELLS:
            if N_ == N:
                rows[f"{tag(arm)}_d{d:.2f}"] = run_rows(rdir(N, arm, d), sc, a.seeds, 0.30)
        out["rows"][str(N)] = rows
        # Q2: learned v3 d0.70 vs every non-learned compensator at its better margin (by mean safe-success)
        L = rows["learned_pooled_d0.70"]
        dd = {}
        for m in ("real", "cv", "smith"):
            best = max((f"{m}_d0.55", f"{m}_d0.70"), key=lambda k: rows[k]["safe_success"]["mean"])
            dd[f"learned_minus_best_{m}"] = {"best_margin_cell": best, **{k: diff_ci(ps(L, k), ps(rows[best], k))
                                             for k in ("safe_success", "ever_collided", "completeness")}}
        out["diffs"][str(N)] = dd
    for arm in ("real_d0.55", "real_d0.70", "cv_d0.55", "cv_d0.70", "smith_d0.55", "smith_d0.70",
                "learned_pooled_d0.70", "noise_only_d0.70", "fresh_d0.70"):
        out["safe_N"][arm] = safe_N(out["rows"], arm) if all(arm in out["rows"][str(N)] for N in NS) else None
    # Q3: layer attribution, recovered fraction of the (both - none) gap, per learned and oracle
    for N in SPLIT_NS:
        r = out["rows"][str(N)]
        lay = {}
        for comp, both in (("learned", "learned_pooled_d0.70"), ("noise_only", "noise_only_d0.70")):
            none = "real_d0.70"
            for layer, cell in (("mfg_only", f"split_{comp}-real_d0.70"), ("cbf_only", f"split_real-{comp}_d0.70")):
                for k in ("safe_success", "ever_collided"):
                    gap = r[both][k]["mean"] - r[none][k]["mean"]
                    lay[f"{comp}_{layer}_{k}"] = {
                        "minus_none": diff_ci(ps(r[cell], k), ps(r[none], k)),
                        "minus_both": diff_ci(ps(r[cell], k), ps(r[both], k)),
                        "recovered_fraction_of_gap": (r[cell][k]["mean"] - r[none][k]["mean"]) / gap if abs(gap) > 1e-9 else None}
        out["layer"][str(N)] = lay
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N, rows in out["rows"].items():
        for arm, r in rows.items():
            print(f"N={N} {arm:26s} " + " ".join(f"{k}={fmt(r[k])}" for k in ("safe_success", "ever_collided", "completeness")))
    print("safe_N:", out["safe_N"])
    for N, dd in out["diffs"].items():
        for k, v in dd.items():
            print(f"N={N} {k} ({v['best_margin_cell']}): safe {v['safe_success']}")
    for N, lay in out["layer"].items():
        for k, v in lay.items():
            print(f"N={N} {k}: -none {v['minus_none']} -both {v['minus_both']} recovered {v['recovered_fraction_of_gap']}")
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "identity", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "identity": cmd_identity, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
