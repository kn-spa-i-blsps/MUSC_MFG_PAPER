"""Amendment-32: window-matched classical baseline `cvls` (constant-velocity least-squares fit through the same
4-broadcast window the learned arm sees; compensation.perception) on the amend21 crossover geometry, d_safe {0.55, 0.70}
x N {25..200}, precision=highest, seeds 0-9. Learned v3 / real / noise_only rows are read from the committed amend27
report (per-seed values). Also writes the cvls margin per N (higher mean safe-success, tie -> 0.70) that amend29/30 use.

    python -m experiments.amend32_cvls write-cells
    python -m experiments.amend32_cvls report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import METRICS, SAFE_COMPL_MIN, SAFE_EVER_MAX, run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend32_cells")
RUN = "amend32_cvls"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend32_cvls.md")
A27 = os.path.join(ROOT, "results", "amend27_ladder_layers", "report.json")
NS = (25, 50, 100, 150, 200)
DS = (0.55, 0.70)
SEEDS = list(range(10))


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N in NS:
        for d in DS:
            c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
            c["cbf"]["d_safe"] = d
            c["run_name"] = f"{RUN}/N{N:03d}/cvls_d{d:.2f}"
            c["chain_arm"] = "cvls"
            p = os.path.join(CELL_DIR, f"N{N:03d}_cvls_d{d:.2f}.yaml")
            with open(p, "w") as f:
                f.write(f"# AMEND32 cell N{N:03d} cvls d{d:.2f}: amend21_cells/N{N:03d}_real.yaml with cbf.d_safe, "
                        f"run_name, chain_arm changed\n")
                yaml.safe_dump(c, f, sort_keys=False)
            print(p)


def cmd_report(a):
    a27 = json.load(open(A27))
    out = {"aneks_sha256": sha256(ANEKS), "matmul_precision": "highest", "seeds": a.seeds,
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "rows": {}, "diffs": {}, "margin": {}, "safe_N": None}
    ps = lambda r, k: r[k]["per_seed"]
    for N in NS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"N{N:03d}_cvls_d0.70.yaml"))), N)
        rows = {f"cvls_d{d:.2f}": run_rows(os.path.join(ROOT, "results", RUN, f"N{N:03d}", f"cvls_d{d:.2f}", "cvls"),
                                           sc, a.seeds, 0.30) for d in DS}
        r27 = a27["rows"][str(N)]
        best = "cvls_d0.55" if rows["cvls_d0.55"]["safe_success"]["mean"] > rows["cvls_d0.70"]["safe_success"]["mean"] \
            else "cvls_d0.70"
        out["margin"][str(N)] = float(best[-4:])
        out["rows"][str(N)] = rows
        out["diffs"][str(N)] = {f"learned_minus_{best}": {k: diff_ci(ps(r27["learned_pooled_d0.70"], k), ps(rows[best], k))
                                                         for k in ("safe_success", "ever_collided", "completeness")},
                                f"{best}_minus_real_best": {k: diff_ci(ps(rows[best], k), ps(max(
                                    (r27["real_d0.55"], r27["real_d0.70"]), key=lambda r: r["safe_success"]["mean"]), k))
                                    for k in ("safe_success", "ever_collided", "completeness")}}
    out["safe_N"] = None   # amend22 definition, at the better cvls margin per N; contiguous from N=25
    for N in NS:
        r = out["rows"][str(N)][f"cvls_d{out['margin'][str(N)]:.2f}"]
        if not (r["ever_collided"]["ci95"][1] <= SAFE_EVER_MAX and r["completeness"]["mean"] >= SAFE_COMPL_MIN):
            break
        out["safe_N"] = N
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N in NS:
        for arm, r in out["rows"][str(N)].items():
            print(f"N={N} {arm} safe={fmt(r['safe_success'])} ever={fmt(r['ever_collided'])} compl={fmt(r['completeness'])}"
                  f" xhat_mse_near={r['closedloop_xhat_mse_near']['mean']:.3f}")
        for k, v in out["diffs"][str(N)].items():
            s = v["safe_success"]
            print(f"N={N} {k}: safe {s['diff']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    print("cvls margins:", out["margin"], "safe_N (best margin per N):", out["safe_N"])
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
