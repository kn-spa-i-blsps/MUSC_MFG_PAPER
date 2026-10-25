"""Amendment-24 (overnight, review_amend22 fix 1): mechanism control for the N crossover. `noise_only` (perception with
the TDMA delay removed, sensor noise kept = perfect delay compensation) at d_safe 0.70, N in {25..200}, seeds 0-9, plus
`fresh` (no delay, no noise) at N in {150, 200}; amend21 geometry + per-N caches, precision=highest. Compared with the
amend22 learned_pooled_d0.70 / real_d0.70 rows (report.json of amend22, read-only).

    python -m experiments.amend24_mechanism_control write-cells
    python -m experiments.amend24_mechanism_control report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import METRICS, run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend24_cells")
RUN = "amend24_mechanism_control"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend24_mechanism_control.md")
A22 = os.path.join(ROOT, "results", "amend22_N_crossover_pooled", "report.json")
CELLS = [(N, "noise_only") for N in (25, 50, 100, 150, 200)] + [(N, "fresh") for N in (150, 200)]
D_SAFE = 0.70
SEEDS = list(range(10))
EXTRA = ("closedloop_xhat_mse_near_pairw", "near_pair_steps")


def cell_name(N, arm):
    return f"N{N:03d}_{arm}_d{D_SAFE:.2f}"


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N, arm in CELLS:
        c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
        c["cbf"]["d_safe"] = D_SAFE
        c["run_name"] = f"{RUN}/N{N:03d}/{arm}_d{D_SAFE:.2f}"
        p = os.path.join(CELL_DIR, f"{cell_name(N, arm)}.yaml")
        with open(p, "w") as f:
            f.write(f"# AMEND24 cell {cell_name(N, arm)} (condition {arm}): amend21_cells/N{N:03d}_real.yaml with "
                    f"cbf.d_safe={D_SAFE} and run_name changed\n")
            yaml.safe_dump(c, f, sort_keys=False)
        print(p)


def cmd_report(a):
    a22 = json.load(open(A22))
    out = {"aneks_sha256": sha256(ANEKS), "matmul_precision": "highest", "seeds": a.seeds,
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "rows": {}, "diffs": {}}
    for N, arm in CELLS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"{cell_name(N, arm)}.yaml"))), N)
        r = run_rows(os.path.join(ROOT, "results", RUN, f"N{N:03d}", f"{arm}_d{D_SAFE:.2f}", arm), sc, a.seeds, 0.30,
                     EXTRA)
        out["rows"].setdefault(str(N), {})[arm] = r
        ref = a22["rows"][str(N)]
        out["diffs"].setdefault(str(N), {}).update({
            f"{arm}_minus_{b}_{k}": diff_ci(r[k]["per_seed"], ref[b][k]["per_seed"])
            for b in ("learned_pooled_d0.70", "real_d0.70") for k in ("safe_success", "ever_collided")})
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N, arms in out["rows"].items():
        for arm, r in arms.items():
            print(f"N={N} {arm:10s} " + " ".join(f"{k}={fmt(r[k])}" for k in METRICS[:5] + EXTRA[:1]))
    print("diffs:", json.dumps(out["diffs"], indent=1))
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
