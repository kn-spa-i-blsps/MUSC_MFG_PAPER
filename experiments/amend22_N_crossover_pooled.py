"""Amendment-22: N crossover on the amend21 fixed geometry with the POOLED (N in {25..200}) predictor, precision=highest.
Arms: real fixed d_safe 0.55 (= amend21 real cells; identity vs amend21), real fixed 0.70 (margin-matched control),
learned_pooled fixed 0.70 (frozen weights results/overnight_p1/predictor_<tag>/predictor_weights.npz, sha256 in the
aneks); N in {25, 50, 100, 150, 200}; seeds 0-9. Per-N Phi/m* caches = amend21 keys. CIs: seed-level (cluster) bootstrap.

    python -m experiments.amend22_N_crossover_pooled write-cells
    python -m experiments.amend22_N_crossover_pooled report [--d_collision 0.30] [--seeds 0 .. 9]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os

import numpy as np
import yaml

from control.params import T_SLOT
from experiments.amend15_density_sweep import ci, safe_success
from experiments.amend20_closing_lockin import diff_ci
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend22_cells")
RUN = "amend22_N_crossover_pooled"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend22_N_crossover_pooled.md")
WEIGHTS = "results/overnight_p1/predictor_v3/predictor_weights.npz"     # gate-passing tag v3 (logs/overnight/progress.md)
OPENLOOP = "results/overnight_p1/openloop_v3.json"
NS = [25, 50, 100, 150, 200]
ARMS = {"real_d0.55": ("real", 0.55), "real_d0.70": ("real", 0.70), "learned_pooled_d0.70": ("learned", 0.70)}
SEEDS = list(range(10))
METRICS = ("safe_success", "ever_collided", "completeness", "p5_sep_worst_step_m", "cbf_intervention_rate",
           "slack_active_rate", "closedloop_xhat_mse_near", "closedloop_xhat_mse_all")
IDENTITY_KEYS = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
                 "cbf_intervention_rate", "slack_active_rate", "closedloop_xhat_mse_near")
SAFE_EVER_MAX, SAFE_COMPL_MIN = 0.10, 0.90


def cell_name(N, arm):
    return f"N{N:03d}_{arm}"


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N in NS:
        for arm, (cond, d) in ARMS.items():
            c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
            c["cbf"]["d_safe"] = d
            c["run_name"] = f"{RUN}/N{N:03d}/{arm}"
            p = os.path.join(CELL_DIR, f"{cell_name(N, arm)}.yaml")
            with open(p, "w") as f:
                f.write(f"# AMEND22 cell {cell_name(N, arm)} (condition {cond}): amend21_cells/N{N:03d}_real.yaml with "
                        f"cbf.d_safe={d} and run_name changed (geometry, cache key, precision unchanged)\n")
                yaml.safe_dump(c, f, sort_keys=False)
            print(p)


def load_cell(name):
    return yaml.safe_load(open(os.path.join(CELL_DIR, f"{name}.yaml")))


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_rows(rdir, sc, seeds, d_collision):
    rows, prec, w = [], set(), set()
    for seed in seeds:
        z = np.load(os.path.join(rdir, f"logs_seed{seed}.npz"))
        m = json.load(open(os.path.join(rdir, f"metrics_seed{seed}.json")))
        f = safe_success({"pos": z["pos"].astype(np.float64), "pos_final": z["pos_final"].astype(np.float64)},
                         sc, d_collision)
        f.update({k: m[k] for k in METRICS[3:]})
        assert abs(f["completeness"] - m["completeness_final"]) < 1e-9
        prec.add(str(m.get("matmul_precision"))); w.add(str(m.get("weights")))
        rows.append(f)
    return {k: ci([r[k] for r in rows]) for k in METRICS} | {"matmul_precision": sorted(prec), "weights": sorted(w)}


def safe_N(rows, arm):
    """largest tested N such that at it AND every smaller tested N: ever<0.30 CI upper <= 0.10 and completeness mean
    >= 0.90 (pre-registered definition); None if not even N=25."""
    best = None
    for N in NS:
        r = rows[str(N)][arm]
        if r["ever_collided"]["ci95"][1] <= SAFE_EVER_MAX and r["completeness"]["mean"] >= SAFE_COMPL_MIN:
            best = N
        else:
            break
    return best


def cmd_report(a):
    out = {"d_collision": a.d_collision, "seeds": a.seeds, "aneks_sha256": sha256(ANEKS), "weights": WEIGHTS,
           "weights_sha256": sha256(os.path.join(ROOT, WEIGHTS)), "matmul_precision": "highest",
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "T_frame_s": {N: N * T_SLOT for N in NS},
           "openloop_gate": json.load(open(os.path.join(ROOT, OPENLOOP)))["per_N"], "rows": {}, "diffs": {}}
    for N in NS:
        sc = build_scenario(load_cell(cell_name(N, "real_d0.55")), N)
        out["rows"][str(N)] = {arm: run_rows(os.path.join(ROOT, "results", load_cell(cell_name(N, arm))["run_name"],
                                                          cond), sc, a.seeds, a.d_collision)
                               for arm, (cond, _) in ARMS.items()}
        r = out["rows"][str(N)]
        out["diffs"][str(N)] = {f"learned_pooled_d0.70_minus_{ref}_{k}": diff_ci(r["learned_pooled_d0.70"][k]["per_seed"],
                                                                                 r[ref][k]["per_seed"])
                                for ref in ("real_d0.70", "real_d0.55") for k in ("safe_success", "ever_collided")}
    out["safe_N"] = {arm: safe_N(out["rows"], arm) for arm in ARMS}
    out["safe_N_definition"] = (f"largest tested N with ever<0.30 CI upper <= {SAFE_EVER_MAX} and completeness mean >= "
                                f"{SAFE_COMPL_MIN} at it and every smaller tested N")
    ids = {}
    for N in NS:
        for seed in a.seeds:
            m21 = json.load(open(os.path.join(ROOT, "results", "amend21_N_crossover", f"N{N:03d}", "real", "real",
                                              f"metrics_seed{seed}.json")))
            m22 = json.load(open(os.path.join(ROOT, "results", RUN, f"N{N:03d}", "real_d0.55", "real",
                                              f"metrics_seed{seed}.json")))
            ids[f"N{N}_s{seed}"] = all(m21[k] == m22[k] for k in IDENTITY_KEYS)
    out["identity_real_d0.55_vs_amend21"] = {"per_cell": ids, "all_equal": all(ids.values())}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)

    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    print(f"\nAMEND22 N crossover (pooled predictor), precision=highest, seeds {a.seeds}\n" + f"{'row':28s}"
          + "".join(f"{k:>26s}" for k in METRICS))
    for N, arms in out["rows"].items():
        for arm, r in arms.items():
            print(f"{'N=' + N + ' ' + arm:28s}" + "".join(f"{fmt(r[k]):>26s}" for k in METRICS))
    print("diffs:", json.dumps(out["diffs"], indent=1))
    print("safe_N:", out["safe_N"])
    print("identity real d0.55 vs amend21:", out["identity_real_d0.55_vs_amend21"]["all_equal"])
    print("precision per row:", {f"{N}/{arm}": r["matmul_precision"] for N, arms in out["rows"].items()
                                 for arm, r in arms.items()})
    print("aneks sha256:", out["aneks_sha256"], "weights sha256:", out["weights_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "report"])
    ap.add_argument("--d_collision", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
