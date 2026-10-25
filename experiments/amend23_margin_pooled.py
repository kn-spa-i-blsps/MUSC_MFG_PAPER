"""Amendment-23 (overnight Phase 2): fixed-margin re-tune with the pooled predictor v3 on the amend21 geometry,
precision=highest. learned_pooled d_safe in {0.55, 0.70, 0.85, 1.00} at N in NS; seeds 0-9; per-N caches = amend21 keys.
Margin selection rule (pre-registered): among margins with completeness mean >= 0.90, the one with the highest
safe-success mean; it replaces the incumbent 0.70 only if its safe-success difference vs 0.70 has a bootstrap CI
excluding 0, else 0.70 is kept ("not resolved").

    python -m experiments.amend23_margin_pooled write-cells
    python -m experiments.amend23_margin_pooled report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import METRICS, WEIGHTS, run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend23_cells")
RUN = "amend23_margin_pooled"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend23_margin_pooled.md")
NS = [100, 150]
MARGINS = [0.55, 0.70, 0.85, 1.00]
INCUMBENT = 0.70
SEEDS = list(range(10))
COMPL_MIN = 0.90
EXTRA = ("closedloop_xhat_mse_near_pairw", "near_pair_steps")


def cell_name(N, d):
    return f"N{N:03d}_learned_pooled_d{d:.2f}"


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N in NS:
        for d in MARGINS:
            c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
            c["cbf"]["d_safe"] = d
            c["run_name"] = f"{RUN}/N{N:03d}/d{d:.2f}"
            p = os.path.join(CELL_DIR, f"{cell_name(N, d)}.yaml")
            with open(p, "w") as f:
                f.write(f"# AMEND23 cell {cell_name(N, d)} (condition learned): amend21_cells/N{N:03d}_real.yaml with "
                        f"cbf.d_safe={d} and run_name changed\n")
                yaml.safe_dump(c, f, sort_keys=False)
            print(p)


def cmd_report(a):
    out = {"aneks_sha256": sha256(ANEKS), "weights": WEIGHTS, "weights_sha256": sha256(os.path.join(ROOT, WEIGHTS)),
           "matmul_precision": "highest", "seeds": a.seeds, "ci": "seed-level (cluster) bootstrap, 10k, 95%",
           "rows": {}, "selection": {}}
    for N in NS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"{cell_name(N, MARGINS[0])}.yaml"))), N)
        rows = {f"{d:.2f}": run_rows(os.path.join(ROOT, "results", RUN, f"N{N:03d}", f"d{d:.2f}", "learned"), sc,
                                     a.seeds, 0.30, EXTRA) for d in MARGINS}
        out["rows"][str(N)] = rows
        ok = [d for d in MARGINS if rows[f"{d:.2f}"]["completeness"]["mean"] >= COMPL_MIN]
        best = max(ok, key=lambda d: rows[f"{d:.2f}"]["safe_success"]["mean"]) if ok else None
        sel = {"eligible": ok, "best_mean": best}
        if best is not None and best != INCUMBENT:
            dd = diff_ci(rows[f"{best:.2f}"]["safe_success"]["per_seed"],
                         rows[f"{INCUMBENT:.2f}"]["safe_success"]["per_seed"])
            sel |= {"best_minus_incumbent": dd, "chosen": best if dd["ci95"][0] > 0 else INCUMBENT,
                    "resolved": dd["ci95"][0] > 0}
        else:
            sel |= {"chosen": INCUMBENT if best is None else best, "resolved": best == INCUMBENT}
        out["selection"][str(N)] = sel
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N, rows in out["rows"].items():
        for d, r in rows.items():
            print(f"N={N} d={d} " + " ".join(f"{k}={fmt(r[k])}" for k in METRICS[:5]))
    print("selection:", json.dumps(out["selection"], indent=1))
    print("aneks sha256:", out["aneks_sha256"], "weights sha256:", out["weights_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
