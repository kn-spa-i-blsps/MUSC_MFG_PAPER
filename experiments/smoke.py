"""Smoke gate runner (prereg §9, amend1, amend2). Config-driven:

    python -m experiments.smoke --config experiments/smoke.yaml

Writes results/<run_name>/: config.yaml (copy), density_sweep.json, macro diagnostics, per-variant
metrics + logs (npz), and viz (GIF animation, snapshots, summary panel, goal profile). Reports facts
only; no pass/fail verdict is computed here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from control.law import local_density_and_grad
from control.params import horizon_for_N, horizon_for_path, RHO_GOAL
from env.base import Grid
from env.rollout import make_rollout, summarize
from env.scenarios import SingleGateA
from mfg.equilibrium import MacroCost, load_macro, save_macro, solve_macro
from viz import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG = yaml.safe_load(open(os.path.join(ROOT, "experiments", "smoke.yaml")))
DEFAULT_COST = {**_CFG["scenario"], **_CFG["macro_cost"]}


def build_scenario(cfg, N):
    s = cfg["scenario"]
    return SingleGateA(N, w_goal=s["w_goal"], w_hold=s["w_hold"], goal_softness=s["goal_softness"],
                       obs_penalty=s["obs_penalty"], wall_thickness=s["wall_thickness"],
                       edge_softness=s["edge_softness"], goal_gap_mult=s.get("goal_gap_mult"),
                       goal_shape=s.get("goal_shape", "conical"), goal_spacing=s.get("goal_spacing"),
                       goal_clearance=s.get("goal_clearance"),
                       goal_attractor_shift=s.get("goal_attractor_shift"))


def density_sweep(cfg):
    out = {}
    for N in cfg["density_sweep_N"]:
        sc = build_scenario(cfg, N)
        W, H = sc.arena
        pos = sc.initial_positions(jax.random.PRNGKey(0))
        rho0, _ = local_density_and_grad(pos, jnp.broadcast_to(pos[None], (N, N, 2)), N)
        x0, x1, z0, z1 = sc.start_box
        out[N] = {"arena_W": W, "arena_H": H, "N_over_arena_area": N / (W * H),
                  "N_over_free_area": N / sc.free_area(), "gate_width": sc.gate_width,
                  "gate_over_W": sc.gate_width / W, "r_goal": sc.r_goal,
                  "goal_usable_area_m2": sc.goal_capacity_area(),
                  "goal_capacity_at_RHO_GOAL": sc.goal_capacity_area() * RHO_GOAL,
                  "N_over_source_chamber": N / sc.source_area(),
                  "start_box_density": N / ((x1 - x0) * (z1 - z0)),
                  "kde_mean_density_t0": float(jnp.mean(rho0))}
    return out


def horizon_for(cfg, sc):
    """Default: HORIZON_MULT * W0 / V_NOM. cfg horizon: path -> amend5 (path length to the goal)."""
    if cfg.get("horizon") == "path":
        return horizon_for_path(sc.path_length_to_goal())
    return horizon_for_N(sc.N)


def get_macro(cfg, sc, grid, c_w, cache_dir):
    mc = cfg["macro_cost"]
    cost = MacroCost(c_w, mc["w_bar"], mc["m_bar"], mc["f_cap"])
    fp = cfg["fp"]
    tag = {"N": sc.N, "scenario": cfg["scenario"], "cost": cost.__dict__, "fp": fp, "step": cfg["grid_step"]}
    if "horizon" in cfg:                     # only non-default horizons enter the key (old keys stay valid)
        tag["horizon"] = [cfg["horizon"], grid.horizon]
    tag = json.dumps(tag, sort_keys=True)
    key = f"macro_N{sc.N}_cw{c_w:g}_{hashlib.sha1(tag.encode()).hexdigest()[:10]}"
    path = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(path):
        return load_macro(path), cost, path
    print(f"[macro] solving c_w={c_w} ...", flush=True)
    sol = solve_macro(sc, grid, cost, damping=fp["damping"], max_iters=fp["max_iters"],
                      tol=fp["tol"], burn_in=fp["burn_in"])
    save_macro(path, sol)
    with open(path + ".json", "w") as f:
        f.write(tag)
    return sol, cost, path


def macro_diag(sol, sc, grid):
    m = np.asarray(sol["m_star"])
    x, z = grid.meshgrid()
    r = np.asarray(sc.goal_distance(x, z))
    in_goal = (m * (r <= sc.r_goal)).sum((1, 2)) * grid.step ** 2
    gate = (np.abs(np.asarray(x) - sc.wall_x) < 1.0)
    T = m.shape[0]
    return {"cesaro_converged": sol["cesaro_converged"], "converged_last_iterate": sol["converged_last_iterate"],
            "iters": sol["iters"], "cesaro_residual_final": float(sol["cesaro_residual_history"][-1]),
            "raw_residual_final": float(sol["residual_history"][-1]),
            "raw_residual_last50_minmax": [float(sol["residual_history"][-50:].min()),
                                            float(sol["residual_history"][-50:].max())],
            "max_F_at_mstar": sol["max_F_at_mstar"], "nan": sol["nan"],
            "solve_seconds": sol["solve_seconds"],
            "mstar_peak": float(m.max()), "mstar_mass_t0": float(m[0].sum() * grid.step ** 2),
            "mstar_mass_in_goal_at_T/3,2T/3,T": [float(in_goal[T // 3]), float(in_goal[2 * T // 3]), float(in_goal[-1])],
            "mstar_peak_gate_band_(|x-W/2|<1m)": float((m * gate[None]).max()),
            "mstar_mass_gate_band_time_integral": float((m * gate[None]).sum() * grid.step ** 2 * grid.dt)}


def goal_zone_facts(series, logs, sc, cfg):
    r = series["r_goal_dist"][-1]
    edges = np.arange(0.0, max(sc.r_goal + 3.0, r.max() + 0.5), 0.5)
    W, H = sc.arena
    st = 0.02
    X, Z = np.meshgrid(np.arange(st / 2, W, st), np.arange(st / 2, H, st), indexing="ij")
    R = np.hypot(X - sc.goal[0], Z - sc.goal[1])
    side = X > sc.wall_x
    area = np.array([((R >= a) & (R < b) & side).sum() * st * st for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.histogram(r, bins=edges)[0]
    dens = np.where(area > 0, cnt / np.maximum(area, 1e-9), 0.0)
    rho_T = np.asarray(logs["rho_true"])[-1]
    ing = r <= sc.r_goal
    q = np.quantile(r, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return {"radial_profile_edges_m": edges.tolist(), "radial_profile_agents_per_m2": dens.tolist(),
            "radial_profile_counts": cnt.tolist(),
            "kde_density_in_goal_final_mean": float(rho_T[ing].mean()) if ing.any() else None,
            "kde_density_in_goal_final_peak": float(rho_T[ing].max()) if ing.any() else None,
            "kde_density_all_final_peak": float(rho_T.max()),
            "dist_to_goal_final_quantiles_[0,.1,.25,.5,.75,.9,.95,1]": q.tolist(),
            "dist_to_goal_final_mean": float(r.mean()),
            "n_outside_r_goal_final": int((~ing).sum())}, (edges, dens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "experiments", "smoke.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = render.ensure_dir(os.path.join(ROOT, "results", cfg["run_name"]))
    cache_dir = render.ensure_dir(os.path.join(ROOT, "results", "cache"))
    viz_dir = render.ensure_dir(os.path.join(out_dir, "viz"))
    shutil.copy(a.config, os.path.join(out_dir, "config.yaml"))
    report = {"config_path": a.config, "jax_version": jax.__version__}

    report["density_sweep"] = density_sweep(cfg)
    print("[density]", json.dumps(report["density_sweep"][cfg["N"]]), flush=True)

    N = cfg["N"]
    sc = build_scenario(cfg, N)
    grid = Grid.build(sc.arena, cfg["grid_step"], horizon_for(cfg, sc))
    assert grid.arena == sc.arena            # same arena feeds solve and rollout
    n_steps = int(round(grid.horizon / cfg["dt_exec"]))
    report["geometry"] = {"arena": sc.arena, "gate_width": sc.gate_width, "goal": sc.goal,
                          "r_goal": sc.r_goal, "horizon_s": grid.horizon, "n_steps_exec": n_steps,
                          "grid": [grid.nx, grid.nz, grid.n_steps], "solver_dt": grid.dt}

    macros = {}
    for c_w in sorted({v["c_w"] for v in cfg["variants"].values()}):
        sol, cost, path = get_macro(cfg, sc, grid, c_w, cache_dir)
        macros[c_w] = (sol, cost)
        report[f"macro_cw{c_w:g}"] = {**macro_diag(sol, sc, grid), "cache": os.path.relpath(path, ROOT)}
        print(f"[macro c_w={c_w}]", json.dumps(report[f"macro_cw{c_w:g}"]), flush=True)
    if 0.0 in macros and 1.0 in macros:
        d = np.abs(np.asarray(macros[1.0][0]["phi_grad"]) - np.asarray(macros[0.0][0]["phi_grad"]))
        report["macro_W_on_vs_off"] = {
            "mean_abs_diff_phi_grad": float(d.mean()), "max_abs_diff_phi_grad": float(d.max()),
            "mstar_peak_on": float(np.asarray(macros[1.0][0]["m_star"]).max()),
            "mstar_peak_off": float(np.asarray(macros[0.0][0]["m_star"]).max())}

    runs_for_panel, profiles = {}, {}
    for label, var in cfg["variants"].items():
        if a.only and label not in a.only:
            continue
        sol, cost = macros[var["c_w"]]
        rcfg = dict(dt_exec=cfg["dt_exec"], n_steps=n_steps, nu=var["nu"], cbf_on=var["cbf_on"],
                    c_w=cost.c_w, w_bar=cost.w_bar, m_bar=cost.m_bar,
                    c_dens=var.get("c_dens", cfg["control"]["c_dens"]),
                    cbf={**copy.deepcopy(cfg["cbf"]), **var.get("cbf", {})})
        run = make_rollout(cfg["condition"], sc, rcfg)
        fields = {k: sol[k] for k in ("phi_grad", "m_star", "m_star_grad")}
        fields.update(dt=sol["dt"], step=sol["step"])
        per_seed = []
        for seed in var.get("seeds", cfg["seeds"]):
            t0 = time.time()
            logs = jax.device_get(run(jax.random.PRNGKey(seed), fields))
            wall = time.time() - t0
            metrics, series = summarize(logs, sc, cfg["dt_exec"], c_w=1.0)
            gz, prof = goal_zone_facts(series, logs, sc, cfg)
            metrics.update(gz)
            metrics["seed"] = seed
            metrics["wall_seconds"] = wall
            per_seed.append(metrics)
            print(f"[{label} seed={seed}] {wall:.0f}s compl={metrics['completeness_final']:.3f} "
                  f"coll={metrics['collision_pair_steps']} minsep={metrics['min_sep_overall_m']:.3f} "
                  f"interv={metrics['cbf_intervention_rate']:.3f} slack={metrics['slack_active_rate']:.4f}",
                  flush=True)
            np.savez_compressed(os.path.join(out_dir, f"logs_{label}_seed{seed}.npz"),
                                **{k: np.asarray(v) for k, v in logs.items()},
                                t=series["t"], arrived_frac=series["arrived_frac"])
            tag = f"{label}_seed{seed}"
            if seed == var.get("seeds", cfg["seeds"])[0]:
                runs_for_panel[label] = (logs, series)
                profiles[label] = prof
                title = (f"single-gate N={N} arm={cfg['condition']} {label} (cbf={var['cbf_on']}, "
                         f"nu={var['nu']}, c_w={var['c_w']}) seed={seed}")
                nfr = render.animate(os.path.join(viz_dir, f"anim_{tag}.gif"), sc, logs, series, sol,
                                     dt_exec=cfg["dt_exec"], title=title)
                render.snapshots(os.path.join(viz_dir, f"snapshots_{tag}.png"), sc, logs, series,
                                 cfg.get("snapshot_times", [0, 2, 4, 6, 8, 12, 20]) + [grid.horizon],
                                 dt_exec=cfg["dt_exec"])
                metrics["anim_frames"] = nfr
        report[f"variant_{label}"] = per_seed
        json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=float)

    render.summary_panel(os.path.join(viz_dir, "summary_panel.png"), sc, runs_for_panel, dt_exec=cfg["dt_exec"])
    render.goal_profile(os.path.join(viz_dir, "goal_profile.png"), profiles, sc.r_goal)
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=float)
    print("[done]", out_dir, flush=True)


if __name__ == "__main__":
    main()
