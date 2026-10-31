"""Compensation ladder (prereg amend6). Config-driven, one process per (arm, seeds) so runs can be
spread over cores:

    python -m experiments.ladder --config experiments/ladder_fargoal.yaml --mode datagen --seeds 100 101
    python -m experiments.ladder --config experiments/ladder_fargoal.yaml --mode eval --arm smith --seeds 0 1

datagen: `real`-arm closed-loop rollouts, saves the TRUE position history (full horizon) per seed.
eval:    closed-loop rollout of one arm, saves logs + metrics per seed. Phi/m* must be in the cache.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from env.base import Grid
from env.rollout import make_rollout, summarize
from experiments.smoke import ROOT, build_scenario, get_macro, goal_zone_facts, horizon_for
from viz import render


def setup(cfg):
    N = cfg["N"]
    sc = build_scenario(cfg, N)
    grid = Grid.build(sc.arena, cfg["grid_step"], horizon_for(cfg, sc))
    cache_dir = os.path.join(ROOT, "results", "cache")
    sol, cost, path = get_macro(cfg, sc, grid, cfg["macro_cost"]["c_w"], cache_dir)
    n_steps = int(round(grid.horizon / cfg["dt_exec"]))
    rcfg = dict(dt_exec=cfg["dt_exec"], n_steps=n_steps, nu=float(cfg.get("nu", 0.0)), cbf_on=cfg.get("cbf_on", True), c_w=cost.c_w,
                w_bar=cost.w_bar, m_bar=cost.m_bar, c_dens=cfg["control"]["c_dens"],
                cbf=copy.deepcopy(cfg["cbf"]), hold_on_arrival=cfg["control"].get("hold_on_arrival", False),
                log_xhat=cfg.get("log_xhat", False), sigma_weights=cfg.get("sigma_weights"))
    return sc, grid, sol, rcfg, path


def field_params_for(sol):
    """smith: routing field -grad Phi (time-indexed in Project A) + its dt / grid step."""
    return dict(phi_grad_field=jnp.asarray(sol["phi_grad"]), grid_step=float(sol["step"]),
                field_dt=float(sol["dt"]))


def load_predictor(path):
    return {k: jnp.asarray(v) for k, v in np.load(path).items()}


def load_sigma(path):
    """amend31: frozen error model (experiments.phaseB_sigma export): normalisers + MLP layers W0,b0,..."""
    return {k: jnp.asarray(v) for k, v in np.load(path).items()}


def make_arm_rollout(arm, sc, rcfg, sol, weights=None):
    kw = {}
    if rcfg["cbf"].get("margin_mode") == "sigma":   # amend31: error model input needs v3 whatever the arm
        kw["sigma_params"] = load_sigma(rcfg["sigma_weights"])
        kw["predictor_params"] = load_predictor(weights)
    parts = arm.split(":", 1)[1].split("/") if arm.startswith("split:") else [arm]   # amend27 layer split
    if "smith" in parts:
        kw["field_params"] = field_params_for(sol)
    if "learned" in parts:
        kw["predictor_params"] = load_predictor(weights)
    return make_rollout(arm, sc, rcfg, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["datagen", "eval"], required=True)
    ap.add_argument("--arm", default="real")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--datagen_arm", default="real")   # overnight v4: closed-loop histories under another arm (default real)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if cfg.get("matmul_precision"):    # amend20: global matmul precision (e.g. "highest"), before anything is traced
        jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])
    precision = jax.config.jax_default_matmul_precision
    print(f"[ladder] jax_default_matmul_precision = {precision!r}", flush=True)
    if cfg.get("t_slot") is not None:  # overnight P4: TDMA slot length override (perception only; absent = T_SLOT 0.005)
        import control.params_legacy as _pl
        import compensation.perception as _pc
        _pl.T_SLOT = _pc.T_SLOT = float(cfg["t_slot"])
    from control.params_legacy import T_SLOT as t_slot
    print(f"[ladder] t_slot = {t_slot!r}", flush=True)
    import compensation.perception as _pc
    if cfg.get("sigma_scale") is not None:   # amend29: UWB range / bearing noise scale (perception only; absent = 1)
        _pc.SIGMA_R, _pc.SIGMA_THETA = _pc.SIGMA_R * float(cfg["sigma_scale"]), _pc.SIGMA_THETA * float(cfg["sigma_scale"])
    p_loss = float(cfg.get("p_loss", 0.0))   # amend29: TDMA packet loss (absent = 0 = unchanged)
    print(f"[ladder] sigma_r = {_pc.SIGMA_R!r} sigma_theta = {_pc.SIGMA_THETA!r} p_loss = {p_loss!r} nu = {cfg.get('nu', 0.0)!r}", flush=True)
    sc, grid, sol, rcfg, path = setup(cfg)
    fields = {k: sol[k] for k in ("phi_grad", "m_star", "m_star_grad")}
    fields.update(dt=sol["dt"], step=sol["step"])
    arm = a.datagen_arm if a.mode == "datagen" else a.arm
    out_dir = render.ensure_dir(os.path.join(ROOT, "results", cfg["run_name"],
                                             "datagen" if a.mode == "datagen" else arm.replace(":", "_").replace("/", "-")))
    run = make_arm_rollout(arm, sc, rcfg, sol, a.weights)
    for seed in a.seeds:
        if p_loss > 0.0:   # loss pattern per seed: set the globals, re-trace (they are read at trace time)
            _pc.P_LOSS, _pc.LOSS_SALT = p_loss, 1000 + seed
            run = make_arm_rollout(arm, sc, rcfg, sol, a.weights)
        t0 = time.time()
        logs = jax.device_get(run(jax.random.PRNGKey(seed), fields))
        wall = time.time() - t0
        if a.mode == "datagen":
            np.savez_compressed(os.path.join(out_dir, f"pos_seed{seed}.npz"),
                                pos=np.asarray(logs["pos"]), pos_final=np.asarray(logs["pos_final"]))
            print(f"[datagen seed={seed}] {wall:.0f}s", flush=True)
            continue
        metrics, series = summarize(logs, sc, cfg["dt_exec"], c_w=1.0)
        gz, _ = goal_zone_facts(series, logs, sc, cfg)
        metrics.update(gz, seed=seed, arm=arm, wall_seconds=wall, macro_cache=os.path.relpath(path, ROOT),
                       weights=a.weights, matmul_precision=precision, t_slot=t_slot,
                       sigma_r=_pc.SIGMA_R, sigma_theta=_pc.SIGMA_THETA, p_loss=p_loss, nu=rcfg["nu"])
        np.savez_compressed(os.path.join(out_dir, f"logs_seed{seed}.npz"),
                            **{k: np.asarray(v) for k, v in logs.items()},
                            t=series["t"], arrived_frac=series["arrived_frac"])
        json.dump(metrics, open(os.path.join(out_dir, f"metrics_seed{seed}.json"), "w"), indent=1,
                  default=float)
        print(f"[{arm} seed={seed}] {wall:.0f}s compl={metrics['completeness_final']:.3f} "
              f"coll={metrics['collision_pair_steps']} minsep={metrics['min_sep_overall_m']:.3f} "
              f"p5={metrics['p5_sep_worst_step_m']:.3f} interv={metrics['cbf_intervention_rate']:.3f} "
              f"slack={metrics['slack_active_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
