"""Amendment-31 (Phase B steps 2-3): conformal sigma-CBF vs the fixed-margin frontier of the same learned v3 arm, amend21
crossover geometry, precision=highest, seeds 0-9.

  F  frontier      learned v3, fixed d_safe {0.70, 0.85, 1.00, 1.20} x N {25..200}
  S  sigma-CBF     learned v3, d_safe_ij = max(0.70, 0.30 + k * q_N(alpha) * exp(g_ij)), k = 1, alpha {0.10, 0.05, 0.01}
                   (q_N(alpha), g from experiments.phaseB_sigma: results/phaseB/sigma_v3/calibration.json) x N {25..200};
                   sigma_bin_a0.05 = the registered §C calibration fix (Mondrian per (N, sigma-quintile) q) x N
  I  identity      N=100 seed 0, real arm, sigma margin with k = 0 (d_safe = max(0.70, 0.30) = 0.70 on every row) vs
                   the committed amend22 real_d0.70 seed 0 metrics (bit-equal, IDENTITY_KEYS)

    python -m experiments.amend31_sigma_cbf write-cells     # after phaseB_sigma (reads calibration.json)
    python -m experiments.amend31_sigma_cbf identity
    python -m experiments.amend31_sigma_cbf report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import IDENTITY_KEYS, METRICS, run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend31_cells")
RUN = "amend31_sigma_cbf"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend31_sigma_cbf.md")
ADDENDUM1 = os.path.join(ROOT, "prereg", "prereg_projectA_amend31_addendum1.md")
NI_MARGIN = 0.02   # addendum 1: completeness non-inferiority margin (absolute fraction of agents)
SIG = os.path.join(ROOT, "results", "phaseB", "sigma_v3")
NS = (25, 50, 100, 150, 200)
FIXED = (0.70, 0.85, 1.00, 1.20)
ALPHAS = (0.10, 0.05, 0.01)
SEEDS = list(range(10))
SIG_EXTRA = ("d_safe_mean", "d_safe_p95", "sigma_cov_rows", "sigma_cov_rows_near2m", "sigma_mean_rows_m",
             "sigma_bind_frac", "sigma_cov_binding")


def arms():
    return [f"fixed{d:.2f}" for d in FIXED] + [f"sigma_a{al:g}" for al in ALPHAS] + ["sigma_bin_a0.05"]


def cell_cfg(N, arm, cal, cond="learned", k=1.0):
    c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
    if arm.startswith("fixed"):
        c["cbf"]["d_safe"] = float(arm[5:])
    else:
        al = arm.split("_a")[1]
        c["cbf"].update(d_safe=0.70, margin_mode="sigma", margin_k=k, margin_floor=0.70,
                        margin_q=float(cal["q"][str(float(al))][str(N)]))
        if arm.startswith("sigma_bin"):
            c["cbf"].update(margin_q_edges=cal["q_edges"][str(N)], margin_q_vals=cal["q_bins"][str(float(al))][str(N)])
        c["sigma_weights"] = os.path.relpath(cal["sigma_weights"], ROOT) if os.path.isabs(cal["sigma_weights"]) \
            else cal["sigma_weights"]
    c["run_name"] = f"{RUN}/N{N:03d}/{arm}"
    c["chain_arm"] = cond
    return c


def cmd_write_cells(a):
    cal = json.load(open(os.path.join(SIG, "calibration.json")))
    os.makedirs(os.path.join(CELL_DIR, "identity"), exist_ok=True)
    jobs = [(N, arm, "learned", 1.0, "") for N in NS for arm in arms()] + [(100, "sigma_a0.05", "real", 0.0, "identity")]
    for N, arm, cond, k, sub in jobs:
        c = cell_cfg(N, arm, cal, cond, k)
        if sub:
            c["run_name"] = f"{RUN}/identity/N100_real_sigma_k0"
        p = os.path.join(CELL_DIR, sub, f"N{N:03d}_{arm}{'_real_k0' if sub else ''}.yaml")
        with open(p, "w") as f:
            f.write(f"# AMEND31 cell N{N:03d} {arm} ({cond}, k={k}): amend21_cells/N{N:03d}_real.yaml + cbf margin, "
                    f"sigma_weights sha {cal['sigma_weights_sha256'][:12]}, run_name, chain_arm\n")
            yaml.safe_dump(c, f, sort_keys=False)
        print(p)


def cmd_identity(a):
    ref = json.load(open(os.path.join(ROOT, "results", "amend22_N_crossover_pooled", "N100", "real_d0.70", "real",
                                      "metrics_seed0.json")))
    m = json.load(open(os.path.join(ROOT, "results", RUN, "identity", "N100_real_sigma_k0", "real", "metrics_seed0.json")))
    out = {k: {"amend22": ref[k], "amend31": m[k], "equal": ref[k] == m[k]} for k in IDENTITY_KEYS}
    out["d_safe_mean"], out["d_safe_p95"] = m["d_safe_mean"], m["d_safe_p95"]
    out["all_equal"] = all(v["equal"] for v in out.values() if isinstance(v, dict))
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "identity.json"), "w"), indent=1)
    print(json.dumps({"all_equal": out["all_equal"], "d_safe_mean": m["d_safe_mean"], "d_safe_p95": m["d_safe_p95"]}))
    if not out["all_equal"]:
        raise SystemExit(1)


def cmd_report(a):
    cal = json.load(open(os.path.join(SIG, "calibration.json")))
    out = {"aneks_sha256": sha256(ANEKS), "matmul_precision": "highest", "seeds": a.seeds,
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "gcal": cal["gcal"], "q": cal["q"], "rows": {},
           "best_fixed": {}, "gwin": {}, "vs_fixed070": {}}
    ps = lambda r, k: r[k]["per_seed"]
    for N in NS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"N{N:03d}_fixed0.70.yaml"))), N)
        rows = {arm: run_rows(os.path.join(ROOT, "results", RUN, f"N{N:03d}", arm, "learned"), sc, a.seeds, 0.30,
                              extra=SIG_EXTRA if arm.startswith("sigma") else ())
                for arm in arms()}
        out["rows"][str(N)] = rows
        best = max((f"fixed{d:.2f}" for d in FIXED), key=lambda k: rows[k]["safe_success"]["mean"])
        out["best_fixed"][str(N)] = best
        out["gwin"][str(N)] = {arm: {m: diff_ci(ps(rows[arm], m), ps(rows[best], m))
                                     for m in ("safe_success", "ever_collided", "completeness")}
                               for arm in arms() if arm.startswith("sigma")}
        out["vs_fixed070"][str(N)] = {arm: {m: diff_ci(ps(rows[arm], m), ps(rows["fixed0.70"], m))
                                            for m in ("safe_success", "ever_collided", "completeness")}
                                      for arm in arms() if arm != "fixed0.70"}
    prim = cal["gcal"]["primary_gwin_arm"]   # registered: sigma_a0.05 if G-cal PASS, sigma_bin_a0.05 if PASS_AFTER_FIX
    win = {N: out["gwin"][str(N)][prim]["safe_success"]["ci95"][0] > 0 for N in (150, 200)}
    out["gwin_verdict"] = {"primary_arm": prim, "per_N": win, "verdict": "PASS" if all(win.values()) else ("PARTIAL" if any(win.values())
                                                                                        else "NOT_MET")}
    # Addendum 1 (PI 2026-11-02, registered before any run): G-win also requires NOT worse completeness vs the same
    # best-fixed comparator = non-inferiority, completeness difference CI95 lower bound >= -NI_MARGIN, at N 150 and 200.
    ni = {N: out["gwin"][str(N)][prim]["completeness"]["ci95"][0] >= -NI_MARGIN for N in (150, 200)}
    both = {N: win[N] and ni[N] for N in (150, 200)}
    out["gwin_verdict_addendum1"] = {"primary_arm": prim, "ni_margin": NI_MARGIN, "safe_win": win,
                                     "completeness_noninferior": ni, "per_N": both,
                                     "verdict": "PASS" if all(both.values()) else ("PARTIAL" if any(both.values())
                                                                                  else "NOT_MET")}
    out["addendum1_sha256"] = sha256(ADDENDUM1)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N in NS:
        for arm, r in out["rows"][str(N)].items():
            ex = (f" d_mean={r['d_safe_mean']['mean']:.3f} cov2m={r['sigma_cov_rows_near2m']['mean']:.4f}"
                  if arm.startswith("sigma") else "")
            print(f"N{N:3d} {arm:12s} safe={fmt(r['safe_success'])} ever={fmt(r['ever_collided'])} "
                  f"compl={fmt(r['completeness'])}{ex}")
        for arm, v in out["gwin"][str(N)].items():
            s = v["safe_success"]
            print(f"N{N:3d} {arm} - best fixed ({out['best_fixed'][str(N)]}): safe {s['diff']:+.3f} "
                  f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    print("G-cal:", cal["gcal"]["verdict"], "| G-win (registered, safe-success only):", json.dumps(out["gwin_verdict"]))
    print("G-win (addendum 1, + completeness non-inferiority):", json.dumps(out["gwin_verdict_addendum1"]))
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "identity", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "identity": cmd_identity, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
