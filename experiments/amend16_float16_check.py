"""Amendment-16 post-hoc (PI-ordered, not preregistered): is the Part 3 replay mismatch (cbf_engaged False for real /
learned) a float16 artifact of the x_hat log? Re-runs the deterministic p3 rollouts real / learned seed 0 with x_hat
logged as float32 (env/rollout.py), checks the trajectory equals the float16-logged Part 3 run, then replays the CBF
filter at the same 200 steps with (a) x_hat float32 and (b) the same x_hat rounded to float16, against the live u.

    python -m experiments.amend16_float16_check
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import yaml

from experiments.amend16_gate_diagnostic import RES, WEIGHTS, load_cell
from experiments.smoke import ROOT, build_scenario

RUNS = {"real": "real", "learned": "learned"}          # p3 cell name -> arm


def replay(cfg, sc, pos, xr, u_nom, ks):
    import jax
    import jax.numpy as jnp
    from control.cbf import CBFParams, cbf_filter, init_warm_start
    prm = CBFParams(**cfg["cbf"])
    segs = jnp.asarray(sc.wall_segments())
    init = init_warm_start(pos.shape[1], prm, segs.shape[0])
    f = jax.jit(lambda p, xh, un: cbf_filter(p, xh, un, jnp.asarray(sc.goal), tuple(sc.arena), segs, init, prm)[0])
    return np.stack([np.asarray(f(jnp.asarray(pos[k]), jnp.asarray(pos[k][:, None] + xr[k]), jnp.asarray(u_nom[k])))
                     for k in ks])


def main():
    out = {}
    for name, arm in RUNS.items():
        cfg = load_cell(f"p3_{name}")
        old_dir = os.path.join(ROOT, "results", cfg["run_name"], arm)
        cfg["run_name"] = cfg["run_name"].replace("/p3/", "/p3_f32/")
        new_dir = os.path.join(ROOT, "results", cfg["run_name"], arm)
        os.makedirs(new_dir, exist_ok=True)
        cpath = os.path.join(ROOT, "results", cfg["run_name"], "config.yaml")
        yaml.safe_dump(cfg, open(cpath, "w"), sort_keys=False)
        if not os.path.exists(os.path.join(new_dir, "logs_seed0.npz")):
            subprocess.run([sys.executable, "-m", "experiments.ladder", "--config", cpath, "--mode", "eval", "--arm", arm,
                            "--weights", WEIGHTS, "--seeds", "0"], check=True, cwd=ROOT)
        z = np.load(os.path.join(new_dir, "logs_seed0.npz"))
        z16 = np.load(os.path.join(old_dir, "logs_seed0.npz"))
        assert z["xhat_rel"].dtype == np.float32, z["xhat_rel"].dtype
        sc = build_scenario(cfg, cfg["N"])
        pos, u_live, u_nom, interv = z["pos"], z["u"], z["u_nom"], z["intervention"]
        T = pos.shape[0]
        ks = np.linspace(0, T - 1, 200).astype(int)
        x32 = z["xhat_rel"][ks].astype(np.float64)
        x16 = z["xhat_rel"][ks].astype(np.float16).astype(np.float64)
        sub = lambda a: a[ks]
        u32 = replay(cfg, sc, sub(pos), x32, sub(u_nom), range(len(ks)))
        u16 = replay(cfg, sc, sub(pos), x16, sub(u_nom), range(len(ks)))
        act = sub(interv) > 0.01
        e = lambda a, b, m: float(np.linalg.norm(a - b, axis=-1)[m].mean())
        out[name] = {
            "same_trajectory_as_part3_float16_run": bool(np.array_equal(pos, z16["pos"])),
            "same_u_as_part3_run": bool(np.array_equal(u_live, z16["u"])),
            "float16_log_equals_float32_rounded": bool(np.array_equal(z16["xhat_rel"][ks], z["xhat_rel"][ks].astype(np.float16))),
            "max_abs_quantization_m": float(np.abs(x16 - x32).max()),
            "intervened_agent_steps": int(act.sum()),
            "mean_err_live_vs_replay_float32": e(u32, u_live[ks], act),
            "mean_err_live_vs_replay_float16": e(u16, u_live[ks], act),
            "mean_err_replay16_vs_replay32": e(u16, u32, act),
            "max_err_live_vs_replay_float32_all_agents": float(np.linalg.norm(u32 - u_live[ks], axis=-1).max()),
            "frac_agent_steps_err32_gt_0.01": float((np.linalg.norm(u32 - u_live[ks], axis=-1) > 0.01).mean()),
            "frac_agent_steps_err16_gt_0.01": float((np.linalg.norm(u16 - u_live[ks], axis=-1) > 0.01).mean()),
        }
        print(name, json.dumps(out[name], indent=1), flush=True)
    json.dump(out, open(os.path.join(RES, "float16_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
