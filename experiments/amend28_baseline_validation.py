"""Amendment-28: delay-free baseline validation, layer ablations and MFG macro fidelity on the amend21 crossover geometry,
d_safe 0.70, N in {25..200}, seeds 0-9, precision=highest.

Arms (cell -> perception condition, controller):
  fresh        fresh,      MFG (c_w 1) + CBF          reference: the delay-free two-layer controller
  fresh_nocbf  fresh,      MFG, CBF off (box clip)    is the CBF needed?
  fresh_nomfg  fresh,      decoupled routing (macro c_w = w_bar = 0 -> single-agent optimal control, no mean-field
                           coupling) + c_dens = 0 (no local density term) + CBF   is the MFG layer worth anything?
  noise_only   noise_only, MFG + CBF                  delay removed, sensor noise kept
  real         real,       MFG + CBF                  TDMA delay, no compensation
  learned      learned v3, MFG + CBF                  TDMA delay, NN compensation

Macro fidelity (per rollout): every 1 s of simulated time, agents' empirical density (histogram on the solver grid,
Gaussian smoothing sigma = H_SMOOTH) vs the solved equilibrium m*(t) of the c_w=1 solve (the MFG prediction; for
fresh_nomfg too -> how far a non-MFG swarm is from it), smoothed identically, both normalized to probability:
  tv(t) = 0.5 * sum |rho_hat - m*_s| (total variation, 0..1); tv_mean over t <= T_EVAL;
  floor: tv of N i.i.d. samples drawn from m*(t) (finite-N sampling floor, 5 draws);
  goal-mass gap: mean over t of |frac of agents in goal disc - frac of m* mass in goal disc|.

    python -m experiments.amend28_baseline_validation write-cells
    python -m experiments.amend28_baseline_validation solve-nomfg --N 25
    python -m experiments.amend28_baseline_validation fidelity --cell N100_fresh
    python -m experiments.amend28_baseline_validation report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import METRICS, run_rows, safe_N, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
CELL_DIR = os.path.join(ROOT, "experiments", "amend28_cells")
RUN = "amend28_baseline_validation"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend28_baseline_validation.md")
NS = (25, 50, 100, 150, 200)
SEEDS = list(range(10))
D_SAFE = 0.70
ARMS = {"fresh": "fresh", "fresh_nocbf": "fresh", "fresh_nomfg": "fresh", "noise_only": "noise_only",
        "real": "real", "learned": "learned"}
H_SMOOTH = 1.0      # m, density smoothing (both fields)
FID_DT = 1.0        # s between fidelity snapshots
T_EVAL = None       # s, fidelity window; None = the full horizon (transit + steady state in the goal disc)
N_FLOOR = 5


def cell_name(N, arm):
    return f"N{N:03d}_{arm}"


def cell_cfg(N, arm):
    c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
    c["cbf"]["d_safe"] = D_SAFE
    c["run_name"] = f"{RUN}/N{N:03d}/{arm}"
    c["amend28_arm"] = ARMS[arm]
    if arm == "fresh_nocbf":
        c["cbf_on"] = False
    if arm == "fresh_nomfg":
        c["macro_cost"]["c_w"] = 0.0
        c["macro_cost"]["w_bar"] = 0.0
        c["control"]["c_dens"] = 0.0
    return c


def cmd_write_cells(a):
    os.makedirs(CELL_DIR, exist_ok=True)
    for N in NS:
        for arm in ARMS:
            p = os.path.join(CELL_DIR, f"{cell_name(N, arm)}.yaml")
            with open(p, "w") as f:
                f.write(f"# AMEND28 cell {cell_name(N, arm)}: amend21_cells/N{N:03d}_real.yaml with d_safe {D_SAFE}, "
                        f"run_name, amend28_arm (chain only) and the arm's controller keys changed\n")
                yaml.safe_dump(cell_cfg(N, arm), f, sort_keys=False)
            print(p)


def cmd_solve_nomfg(a):
    import time
    from experiments.ladder import setup
    t0 = time.time()
    _, grid, sol, _, path = setup(cell_cfg(a.N, "fresh_nomfg"))
    print(json.dumps({"N": a.N, "cache": os.path.relpath(path, ROOT), "wall_s": time.time() - t0,
                      "cesaro_converged": bool(sol["cesaro_converged"]), "iters": int(sol["iters"]),
                      "nan": bool(sol["nan"])}), flush=True)


def rdir(N, arm):
    return os.path.join(ROOT, "results", RUN, f"N{N:03d}", arm, ARMS[arm])


def smooth_hist(pts, nx, nz, step):
    ix = np.clip(np.round(pts[:, 0] / step).astype(int), 0, nx - 1)
    iz = np.clip(np.round(pts[:, 1] / step).astype(int), 0, nz - 1)
    h = np.zeros((nx, nz))
    np.add.at(h, (ix, iz), 1.0)
    h = gaussian_filter(h, H_SMOOTH / step, mode="constant")
    return h / h.sum()


def cmd_fidelity(a):
    from experiments.ladder import setup
    N, arm = a.N, a.arm
    ref = cell_cfg(N, "fresh")                           # the c_w=1 MFG prediction, for every arm
    sc, grid, sol, rcfg, path = setup(ref)
    m = sol["m_star"]                                    # (T+1, nx, nz), device array; one slice per snapshot
    dt, step = float(sol["dt"]), float(sol["step"])
    nx, nz = m.shape[1], m.shape[2]
    X, Z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step, indexing="ij")
    in_goal = np.asarray(sc.goal_distance(X, Z)) <= sc.r_goal
    dt_exec = ref["dt_exec"]
    ts = np.arange(0.0, (T_EVAL or grid.horizon) + 1e-9, FID_DT)
    mstar_s, floor = [], []
    rng = np.random.default_rng(0)
    for t in ts:
        mt = np.maximum(np.asarray(m[min(int(round(t / dt)), m.shape[0] - 1)], np.float64), 0.0)
        p = mt.ravel() / mt.sum()
        ms = gaussian_filter(mt, H_SMOOTH / step, mode="constant")
        ms /= ms.sum()
        mstar_s.append((ms, float((mt * in_goal).sum() / mt.sum())))
        fl = []
        for _ in range(N_FLOOR):
            idx = rng.choice(p.size, N, p=p)
            pts = np.stack([(idx // nz) * step, (idx % nz) * step], -1) + rng.uniform(-step / 2, step / 2, (N, 2))
            fl.append(0.5 * np.abs(smooth_hist(pts, nx, nz, step) - ms).sum())
        floor.append(float(np.mean(fl)))
    out = {"cell": cell_name(N, arm), "macro_cache_ref": os.path.relpath(path, ROOT), "H_SMOOTH": H_SMOOTH,
           "T_EVAL": T_EVAL or grid.horizon, "ts": ts.tolist(), "floor_tv": floor, "floor_tv_mean": float(np.mean(floor)),
           "mstar_goal_frac": [g for _, g in mstar_s], "per_seed": {}}
    for seed in a.seeds:
        pos = np.load(os.path.join(rdir(N, arm), f"logs_seed{seed}.npz"))["pos"].astype(np.float64)
        tv, gg = [], []
        for t, (ms, g) in zip(ts, mstar_s):
            k = min(int(round(t / dt_exec)), pos.shape[0] - 1)
            tv.append(0.5 * np.abs(smooth_hist(pos[k], nx, nz, step) - ms).sum())
            gg.append(abs(float((np.linalg.norm(pos[k] - np.asarray(sc.goal), axis=-1) <= sc.r_goal).mean()) - g))
        out["per_seed"][str(seed)] = {"tv": tv, "tv_mean": float(np.mean(tv)), "goal_gap_mean": float(np.mean(gg)),
                                      "excess_tv_mean": float(np.mean(tv) - np.mean(floor))}
    os.makedirs(os.path.join(RES, "fidelity"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "fidelity", f"{cell_name(N, arm)}.json"), "w"), indent=1)
    print(cell_name(N, arm), "floor", round(out["floor_tv_mean"], 3), "tv_mean",
          round(float(np.mean([v["tv_mean"] for v in out["per_seed"].values()])), 3), flush=True)


def cmd_report(a):
    from experiments.amend15_density_sweep import ci
    out = {"aneks_sha256": sha256(ANEKS), "matmul_precision": "highest", "seeds": a.seeds,
           "ci": "seed-level (cluster) bootstrap, 10k, 95%", "rows": {}, "fidelity": {}, "diffs": {}, "safe_N": {}}
    ps = lambda r, k: r[k]["per_seed"]
    for N in NS:
        sc = build_scenario(cell_cfg(N, "fresh"), N)
        out["rows"][str(N)] = {arm: run_rows(rdir(N, arm), sc, a.seeds, 0.30) for arm in ARMS}
        fid = {}
        for arm in ARMS:
            f = json.load(open(os.path.join(RES, "fidelity", f"{cell_name(N, arm)}.json")))
            fid[arm] = {k: ci([f["per_seed"][str(s)][k] for s in a.seeds]) for k in ("tv_mean", "excess_tv_mean",
                                                                                    "goal_gap_mean")}
            fid[arm]["floor_tv_mean"] = f["floor_tv_mean"]
        out["fidelity"][str(N)] = fid
        r = out["rows"][str(N)]
        d = {}
        for x, y in (("fresh", "fresh_nocbf"), ("fresh", "fresh_nomfg"), ("learned", "real"), ("noise_only", "learned")):
            d[f"{x}_minus_{y}"] = {k: diff_ci(ps(r[x], k), ps(r[y], k)) for k in ("safe_success", "ever_collided",
                                                                                  "completeness")}
            d[f"{x}_minus_{y}"]["tv_mean"] = diff_ci(ps(fid[x], "tv_mean"), ps(fid[y], "tv_mean"))
            d[f"{x}_minus_{y}"]["goal_gap_mean"] = diff_ci(ps(fid[x], "goal_gap_mean"), ps(fid[y], "goal_gap_mean"))
        out["diffs"][str(N)] = d
    for arm in ARMS:
        out["safe_N"][arm] = safe_N(out["rows"], arm)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N in out["rows"]:
        for arm in ARMS:
            r, f = out["rows"][N][arm], out["fidelity"][N][arm]
            print(f"N={N} {arm:12s} safe={fmt(r['safe_success'])} ever={fmt(r['ever_collided'])} "
                  f"compl={fmt(r['completeness'])} tv={fmt(f['tv_mean'])} (floor {f['floor_tv_mean']:.3f}) "
                  f"goalgap={fmt(f['goal_gap_mean'])}")
    for N, d in out["diffs"].items():
        for k, v in d.items():
            print(f"N={N} {k}: " + " ".join(f"{m}={v[m]['diff']:+.3f}[{v[m]['ci95'][0]:+.3f},{v[m]['ci95'][1]:+.3f}]"
                                            for m in v))
    print("safe_N:", out["safe_N"])
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "solve-nomfg", "fidelity", "report"])
    ap.add_argument("--N", type=int)
    ap.add_argument("--arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "solve-nomfg": cmd_solve_nomfg, "fidelity": cmd_fidelity,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
