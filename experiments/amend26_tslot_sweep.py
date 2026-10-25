"""Amendment-26 (overnight Phase 4): pure-staleness sweep via the TDMA slot length t_slot at fixed N=100, fixed amend21
geometry (gate 2.309) and the SAME Phi/m* cache (t_slot is perception-only and not in the cache key), precision=highest.
t_slot in {0.0025, 0.005, 0.0075, 0.010} s -> T_frame = 0.25 / 0.5 / 0.75 / 1.0 s (the T_frame range of N=50..200 in
amend22, i.e. inside the pooled predictor's training range). Arms: real d_safe 0.55, real d_safe D, learned_pooled (v3)
d_safe D; seeds 0-9. Open-loop estimator check per t_slot on that t_slot's real-arm closed-loop histories.

    python -m experiments.amend26_tslot_sweep write-cells
    python -m experiments.amend26_tslot_sweep openloop
    python -m experiments.amend26_tslot_sweep report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import METRICS, WEIGHTS, run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells", "N100_real.yaml")
CELL_DIR = os.path.join(ROOT, "experiments", "amend26_cells")
RUN = "amend26_tslot_sweep"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend26_tslot_sweep.md")
N = 100
GATES = [0.0025, 0.005, 0.0075, 0.010]           # t_slot values (name kept from amend25 for the shared code)
D = 0.70                                     # amend23 selection (set before the commit)
ARMS = {"real_d0.55": ("real", 0.55), f"real_d{D:.2f}": ("real", D), f"learned_pooled_d{D:.2f}": ("learned", D)}
SEEDS = list(range(10))
EXTRA = ("closedloop_xhat_mse_near_pairw", "near_pair_steps")


def gtag(g):
    return f"ts{g * 1000:.2f}ms"


def cell_name(g, arm):
    return f"{gtag(g)}_{arm}"


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for g in GATES:
        for arm, (cond, d) in ARMS.items():
            c = copy.deepcopy(yaml.safe_load(open(SRC)))
            c["t_slot"] = g
            c["cbf"]["d_safe"] = d
            c["run_name"] = f"{RUN}/{gtag(g)}/{arm}"
            p = os.path.join(CELL_DIR, f"{cell_name(g, arm)}.yaml")
            with open(p, "w") as f:
                f.write(f"# AMEND26 cell {cell_name(g, arm)} (condition {cond}): amend21_cells/N100_real.yaml with "
                        f"t_slot={g}, cbf.d_safe={d}, run_name changed\n")
                yaml.safe_dump(c, f, sort_keys=False)
            print(p)


def geometry(g):
    sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"{cell_name(g, 'real_d0.55')}.yaml"))), N)
    return sc, {"arena": list(sc.arena), "gate_width": sc.gate_width, "wall_x": sc.wall_x, "goal_xz": list(sc.goal),
                "r_goal": sc.r_goal, "start_box": list(sc.start_box),
                "density_agents_per_free_m2": N / sc.free_area(), "t_slot": g, "T_frame_s": N * g}


def cmd_openloop(a):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")
    from compensation.perception import perceive
    from experiments.ladder import load_predictor
    pp = load_predictor(os.path.join(ROOT, WEIGHTS))
    fns = {c: jax.jit(lambda h, k, key, c=c: perceive(c, h, k, 0.01, N, key, predictor_params=pp))
           for c in ("real", "learned")}
    out = {}
    import control.params_legacy as _pl
    import compensation.perception as _pc
    for g in GATES:
        _pl.T_SLOT = _pc.T_SLOT = g
        fns = {c: jax.jit(lambda h, k, key, c=c: perceive(c, h, k, 0.01, N, key, predictor_params=pp))
               for c in ("real", "learned")}
        acc = {c: [] for c in fns}
        for s in SEEDS:
            h = jnp.asarray(np.load(os.path.join(RES, gtag(g), "real_d0.55", "real", f"logs_seed{s}.npz"))["pos"],
                            jnp.float32)
            for k in np.random.default_rng(7 + s).integers(400, h.shape[0], 80):
                p = np.asarray(h[k]); d = np.linalg.norm(p[:, None] - p[None], axis=-1)
                near = ~np.eye(N, dtype=bool) & (d < 1.0)
                key = jax.random.fold_in(jax.random.PRNGKey(s), int(k))
                for c, f in fns.items():
                    acc[c].append(np.sum((np.asarray(f(h, int(k), key)) - p[None]) ** 2, -1)[near])
        out[gtag(g)] = {c: float(np.mean(np.concatenate(v))) for c, v in acc.items()}
        out[gtag(g)]["ratio_learned_over_real"] = out[gtag(g)]["learned"] / out[gtag(g)]["real"]
        print(gtag(g), out[gtag(g)], flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "openloop.json"), "w"), indent=1)


def cmd_report(a):
    out = {"aneks_sha256": sha256(ANEKS), "weights": WEIGHTS, "weights_sha256": sha256(os.path.join(ROOT, WEIGHTS)),
           "N": N, "matmul_precision": "highest", "seeds": a.seeds, "ci": "seed-level (cluster) bootstrap, 10k, 95%",
           "openloop": json.load(open(os.path.join(RES, "openloop.json"))), "geometry": {}, "rows": {}, "diffs": {}}
    for g in GATES:
        sc, geo = geometry(g)
        out["geometry"][gtag(g)] = geo
        rows = {arm: run_rows(os.path.join(ROOT, "results", RUN, gtag(g), arm, cond), sc, a.seeds, 0.30, EXTRA)
                for arm, (cond, _) in ARMS.items()}
        out["rows"][gtag(g)] = rows
        lk, rk = f"learned_pooled_d{D:.2f}", f"real_d{D:.2f}"
        out["diffs"][gtag(g)] = {f"learned_minus_real_d{D:.2f}_{k}": diff_ci(rows[lk][k]["per_seed"], rows[rk][k]["per_seed"])
                                 for k in ("safe_success", "ever_collided")}
    # identity: the baseline-value cell's real_d0.55 rows vs amend21 N=100 real (same config up to explicit keys)
    K = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
         "cbf_intervention_rate", "slack_active_rate", "closedloop_xhat_mse_near")
    ids = {}
    for s in a.seeds:
        m21 = json.load(open(os.path.join(ROOT, "results", "amend21_N_crossover", "N100", "real", "real",
                                          f"metrics_seed{s}.json")))
        m = json.load(open(os.path.join(RES, gtag(0.005), "real_d0.55", "real", f"metrics_seed{s}.json")))
        ids[s] = all(m21[k] == m[k] for k in K)
    out["identity_baseline_real_d0.55_vs_amend21_N100"] = {"per_seed": ids, "all_equal": all(ids.values())}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    print("identity vs amend21:", out["identity_baseline_real_d0.55_vs_amend21_N100"]["all_equal"])
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for g, rows in out["rows"].items():
        for arm, r in rows.items():
            print(f"{g} {arm:22s} " + " ".join(f"{k}={fmt(r[k])}" for k in METRICS[:5] + EXTRA[:1]))
    print("diffs:", json.dumps(out["diffs"], indent=1))
    print("openloop:", json.dumps(out["openloop"], indent=1))
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "openloop", "report"])
    ap.add_argument("--gate", type=float)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "openloop": cmd_openloop, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
