"""Project A closed-loop rollout: perception (ported, any ladder arm) -> macro nominal control ->
CBF-QP filter -> single-integrator step. The SAME scenario object (hence the same arena) that fed
the offline solve feeds the rollout. Control uses estimated neighbours x_hat; every metric uses TRUE
positions only. FULL-horizon logging (no settled-only window).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from compensation.perception import perceive
from control.cbf import CBFParams, cbf_filter, init_warm_start
from control.law import local_density_and_grad
from control.macro import nominal_control
from control.params import D_COLLISION, V_MAX


def make_rollout(condition, scenario, cfg, *, predictor_params=None, field_params=None):
    """Returns run(key, macro) -> logs; macro = {phi_grad, m_star, m_star_grad} (passed to jit as
    arguments) + python floats dt, step. cfg keys: dt_exec, n_steps, nu, cbf_on, c_w, w_bar, m_bar, c_dens, cbf (dict of CBFParams)."""
    N = scenario.N
    W, H = scenario.arena
    arena_arr = jnp.array([W, H])
    goal = jnp.asarray(scenario.goal)
    segs = jnp.asarray(scenario.wall_segments())
    prm = CBFParams(**cfg["cbf"])
    dt_exec, n_steps, nu = cfg["dt_exec"], cfg["n_steps"], cfg["nu"]
    cbf_on = cfg["cbf_on"]
    iu = jnp.triu_indices(N, 1)

    def step_fn(carry, k, macro):
        positions, history, key, warm = carry
        history = history.at[k].set(positions)
        key, pk = jax.random.split(key)
        x_hat = perceive(condition, history, k, dt_exec, N, pk,
                         predictor_params=predictor_params, field_params=field_params)
        t_sim = k * dt_exec
        u_nom, mdiag = nominal_control(positions, x_hat, t_sim, N, macro, c_w=cfg["c_w"],
                                       w_bar=cfg["w_bar"], m_bar=cfg["m_bar"], c_dens=cfg["c_dens"])
        nrm = jnp.linalg.norm(u_nom, axis=-1, keepdims=True)
        u_nom = u_nom * jnp.minimum(1.0, V_MAX / jnp.maximum(nrm, 1e-9))
        if cbf_on:
            u, info, warm = cbf_filter(positions, x_hat, u_nom, goal, (W, H), segs, warm, prm)
        else:
            u = jnp.clip(u_nom, -prm.u_box, prm.u_box)
            z = jnp.zeros(N)
            info = {"slack_max": z, "slack_hard": z, "intervention": z, "qp_iters": jnp.zeros(N, jnp.int32),
                    "row_violation": z}
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, positions.shape) * jnp.sqrt(2.0 * nu * dt_exec)
        positions_new = jnp.clip(positions + dt_exec * u + noise, jnp.zeros(2), arena_arr)

        dfull = jnp.linalg.norm(positions[:, None] - positions[None], axis=-1)   # true
        d = dfull[iu] if N > 1 else jnp.full((1,), jnp.inf)                     # unique pairs
        nn = jnp.min(jnp.where(jnp.eye(N, dtype=bool), jnp.inf, dfull), axis=1)  # per-agent NN dist
        rho_true, _ = local_density_and_grad(positions, jnp.broadcast_to(positions[None], (N, N, 2)), N)
        # estimation error of x_hat (TRUE positions), safety-relevant pairs (true d < 1 m) and all pairs
        e2 = jnp.sum((x_hat - positions[None]) ** 2, -1)
        offd = ~jnp.eye(N, dtype=bool)
        near = offd & (dfull < 1.0)
        xhat_err_near = jnp.sum(jnp.where(near, e2, 0.0)) / jnp.maximum(jnp.sum(near), 1)
        xhat_err_all = jnp.sum(jnp.where(offd, e2, 0.0)) / (N * (N - 1))
        y = {"pos": positions, "u": u, "u_nom": u_nom,
             "n_coll_pairs": jnp.sum(d < D_COLLISION), "min_sep": jnp.min(d),
             "p5_sep_pairs": jnp.percentile(nn, 5.0),
             "slack_max": info["slack_max"], "slack_hard": info["slack_hard"], "intervention": info["intervention"],
             "qp_iters": info["qp_iters"], "row_violation": info["row_violation"],
             "m_hat": mdiag["m_hat"], "m_star_at": mdiag["m_star"], "rho_true": rho_true,
             "xhat_err_near": xhat_err_near, "xhat_err_all": xhat_err_all}
        return (positions_new, history, key, warm), y

    def run_arrays(key, arrays, dt, step):
        macro = dict(arrays, dt=dt, step=step)
        key, k0 = jax.random.split(key)
        pos0 = scenario.initial_positions(k0)
        n_walls = segs.shape[0]
        carry0 = (pos0, jnp.zeros((n_steps, N, 2)), key, init_warm_start(N, prm, n_walls))
        (pos_T, _, _, _), logs = jax.lax.scan(lambda c, k: step_fn(c, k, macro), carry0,
                                              jnp.arange(n_steps))
        logs = dict(logs)
        logs["pos_final"] = pos_T
        return logs

    jitted = jax.jit(run_arrays, static_argnames=("dt", "step"))

    def run(key, macro):
        arrays = {k: macro[k] for k in ("phi_grad", "m_star", "m_star_grad")}
        return jitted(key, arrays, dt=float(macro["dt"]), step=float(macro["step"]))

    return run


def summarize(logs, scenario, dt_exec, c_w=1.0):
    """Per-run metrics (prereg §7) from TRUE positions. All numpy."""
    pos = np.asarray(logs["pos"])                                     # (T,N,2), state at step start
    pos = np.concatenate([pos, np.asarray(logs["pos_final"])[None]], 0)
    g = np.asarray(scenario.goal)
    r = np.linalg.norm(pos - g[None, None], axis=-1)                   # (T+1,N)
    arrived = r <= scenario.r_goal
    frac = arrived.mean(1)
    t = np.arange(pos.shape[0]) * dt_exec
    ever = arrived.any(0)
    first = np.where(ever, arrived.argmax(0), -1)
    crossed = pos[..., 0] > scenario.wall_x
    coll = np.asarray(logs["n_coll_pairs"])
    interv = np.asarray(logs["intervention"])
    slack = np.asarray(logs["slack_max"])
    p5 = np.asarray(logs["p5_sep_pairs"])
    out = {
        "nan_position_entries": int(np.isnan(pos).sum()),
        "qp_nan_fallbacks": int((np.asarray(logs["qp_iters"]) < 0).sum()),
        "completeness_final": float(frac[-1]),
        "arrived_frac_max": float(frac.max()),
        "t_first_90pct": float(t[np.argmax(frac >= 0.9)]) if (frac >= 0.9).any() else None,
        "makespan_s": float(t[first].max()) if ever.all() else None,
        "mean_time_to_goal_s": float(t[first[ever]].mean()) if ever.any() else None,
        "frac_ever_arrived": float(ever.mean()),
        "frac_left_goal_after_arrival": float(np.mean([(~arrived[f:, i]).any() for i, f in enumerate(first) if f >= 0])) if ever.any() else None,
        "crossed_wall_final": float(crossed[-1].mean()),
        "collision_pair_steps": int(coll.sum()),
        "collision_steps": int((coll > 0).sum()),
        "min_sep_overall_m": float(np.asarray(logs["min_sep"]).min()),
        "p5_sep_worst_step_m": float(p5.min()),
        "p5_sep_median_over_time_m": float(np.median(p5)),
        "cbf_intervention_rate": float((interv > 0.01).mean()),
        "cbf_intervention_mean_mps": float(interv.mean()),
        "slack_active_rate": float((slack > 1e-4).mean()),
        "slack_max": float(slack.max()),
        "slack_hard_max": float(np.asarray(logs["slack_hard"]).max()) if "slack_hard" in logs else None,
        "qp_aux_mean": float(np.asarray(logs["qp_iters"]).mean()),
        "qp_aux_max": int(np.asarray(logs["qp_iters"]).max()),
        "row_violation_max": float(np.asarray(logs["row_violation"]).max()),
        "mean_speed_mps": float(np.linalg.norm(np.asarray(logs["u"]), axis=-1).mean()),
        "closedloop_xhat_mse_near": float(np.mean(logs["xhat_err_near"])) if "xhat_err_near" in logs else None,
        "closedloop_xhat_mse_all": float(np.mean(logs["xhat_err_all"])) if "xhat_err_all" in logs else None,
        "integral_W_true": float(np.sum(np.mean(c_w * np.asarray(logs["rho_true"]) ** 2, 1)) * dt_exec),
    }
    return out, {"t": t, "arrived_frac": frac, "r_goal_dist": r, "first_arrival_idx": first}
