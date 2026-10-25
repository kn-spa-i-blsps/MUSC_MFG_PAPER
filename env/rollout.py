"""Project A closed-loop rollout: perception (ported, any ladder arm) -> macro nominal control ->
CBF-QP filter -> single-integrator step. The SAME scenario object (hence the same arena) that fed
the offline solve feeds the rollout. Control uses estimated neighbours x_hat; every metric uses TRUE
positions only. FULL-horizon logging (no settled-only window).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from compensation.perception import neighbour_abs_velocity, perceive, staleness_and_rel_velocity
from control.cbf import CBFParams, cbf_filter, init_warm_start
from control.law import local_density_and_grad
from control.macro import nominal_control
from control.params import D_COLLISION, V_MAX


def make_rollout(condition, scenario, cfg, *, predictor_params=None, field_params=None):
    """Returns run(key, macro) -> logs; macro = {phi_grad, m_star, m_star_grad} (passed to jit as
    arguments) + python floats dt, step. cfg keys: dt_exec, n_steps, nu, cbf_on, c_w, w_bar, m_bar, c_dens, cbf (dict of CBFParams);
    optional (amend7): hold_on_arrival (bool; d_hold = cbf d_safe), log_xhat (bool: log x_hat - own
    position per ordered pair as float32 (float16 before the amend16 post-hoc check), the CBF neighbour indices and the hold flag)."""
    N = scenario.N
    W, H = scenario.arena
    arena_arr = jnp.array([W, H])
    goal = jnp.asarray(scenario.goal)
    segs = jnp.asarray(scenario.wall_segments())
    prm = CBFParams(**cfg["cbf"])
    dt_exec, n_steps, nu = cfg["dt_exec"], cfg["n_steps"], cfg["nu"]
    cbf_on = cfg["cbf_on"]
    hold = (goal, scenario.r_goal, prm.d_safe) if cfg.get("hold_on_arrival", False) else None
    log_xhat = cfg.get("log_xhat", False)
    adaptive = prm.margin_k is not None                                # amend17 staleness-aware margin
    closing = adaptive and prm.margin_mode == "closing"                # amend19 closure-gated margin
    iu = jnp.triu_indices(N, 1)

    def step_fn(carry, k, macro):
        positions, history, key, warm = carry
        history = history.at[k].set(positions)
        key, pk = jax.random.split(key)
        x_hat = perceive(condition, history, k, dt_exec, N, pk,
                         predictor_params=predictor_params, field_params=field_params)
        t_sim = k * dt_exec
        u_nom, mdiag = nominal_control(positions, x_hat, t_sim, N, macro, c_w=cfg["c_w"],
                                       w_bar=cfg["w_bar"], m_bar=cfg["m_bar"], c_dens=cfg["c_dens"],
                                       hold=hold)
        nrm = jnp.linalg.norm(u_nom, axis=-1, keepdims=True)
        u_nom = u_nom * jnp.minimum(1.0, V_MAX / jnp.maximum(nrm, 1e-9))
        d_safe_pair = v_closing = None
        if adaptive and closing:   # amend19: d_safe_ij = max(floor, D_COLLISION + k * max(0, v_closing_ij) * h)
            hb, pkb = jax.lax.optimization_barrier((history, pk))
            v_hat_j = neighbour_abs_velocity(hb, k, dt_exec, N, pkb)            # (N_i,N_j,2) heard abs velocity of j
            v_i = jnp.where(k > 0, history[k] - history[jnp.maximum(k - 1, 0)], 0.0) / dt_exec   # own velocity (N,2)
            e = x_hat - positions[:, None]                                       # direction i -> j (the CBF's x_hat)
            e = e / jnp.maximum(jnp.linalg.norm(e, axis=-1, keepdims=True), 1e-9)
            v_closing = jnp.maximum(jnp.sum((v_i[:, None] - v_hat_j) * e, -1), 0.0)
            d_safe_pair = jnp.maximum(prm.margin_floor, D_COLLISION + prm.margin_k * v_closing * prm.margin_h)
        elif adaptive:   # d_safe_ij = min(D_COLLISION + k * |v_hat_ij| * age_j, d_max); same key -> same draws as perceive
            hb, pkb = jax.lax.optimization_barrier((history, pk))   # identity; keeps XLA from fusing into perceive()
            age, v_hat = staleness_and_rel_velocity(hb, k, dt_exec, N, pkb)
            sigma = jnp.linalg.norm(v_hat, axis=-1) * age[None, :]
            d_safe_pair = jnp.minimum(D_COLLISION + prm.margin_k * sigma, prm.margin_d_max)
        if cbf_on:
            u, info, warm = cbf_filter(positions, x_hat, u_nom, goal, (W, H), segs, warm, prm, d_safe_pair)
        else:
            u = jnp.clip(u_nom, -prm.u_box, prm.u_box)
            z = jnp.zeros(N)
            info = {"slack_max": z, "slack_hard": z, "intervention": z, "qp_iters": jnp.zeros(N, jnp.int32),
                    "row_violation": z, "nb_idx": jnp.zeros((N, prm.k_nb), jnp.int32)}
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
             "xhat_err_near": xhat_err_near, "xhat_err_all": xhat_err_all,
             "n_near": jnp.sum(near).astype(jnp.int32)}   # overnight: near-pair count (logging only) -> pair-weighted MSE
        if adaptive and cbf_on:   # realized margin on the valid neighbour rows: exact sum/count + 0.01 m histogram
            ds, ok = info["nb_d_safe"], info["nb_valid"]
            b = jnp.clip(jnp.round((ds - D_COLLISION) / DSAFE_BIN).astype(jnp.int32), 0, DSAFE_NBINS - 1)
            y["d_safe_hist"] = jnp.zeros(DSAFE_NBINS, jnp.int32).at[b.reshape(-1)].add(ok.reshape(-1).astype(jnp.int32))
            y["d_safe_sum"] = jnp.sum(jnp.where(ok, ds, 0.0))
            y["d_safe_n"] = jnp.sum(ok)
            if closing:   # amend19: v_closing on the same valid neighbour rows, 0.05 m/s histogram + exact sum
                vc = jnp.take_along_axis(v_closing, info["nb_idx"], axis=1)
                bv = jnp.clip(jnp.round(vc / VCLOSE_BIN).astype(jnp.int32), 0, VCLOSE_NBINS - 1)
                y["v_closing_hist"] = jnp.zeros(VCLOSE_NBINS, jnp.int32).at[bv.reshape(-1)].add(
                    ok.reshape(-1).astype(jnp.int32))
                y["v_closing_sum"] = jnp.sum(jnp.where(ok, vc, 0.0))
        if log_xhat:
            y["xhat_rel"] = (x_hat - positions[:, None]).astype(jnp.float32)   # (N,N,2) estimate - own pos (float32 since amend16 post-hoc; float16 before)
            y["cbf_nb_idx"] = info["nb_idx"].astype(jnp.int8)
            y["held"] = mdiag["held"]
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


DSAFE_BIN, DSAFE_NBINS = 0.01, 1001   # realized-d_safe histogram: bins D_COLLISION + 0.01 b, b = 0..1000
                                      # (241 bins in amend17; widened in amend19, closing margins reach ~10 m)
VCLOSE_BIN, VCLOSE_NBINS = 0.05, 341  # amend19 v_closing histogram: bins 0.05 b m/s, b = 0..340 (top bin = clip)


def dsafe_stats(logs):
    """amend17: mean / p50 / p95 of the realized adaptive d_safe over all valid neighbour rows x agents x steps
    (mean exact; percentiles at the 0.01 m bin resolution). None for constant-d_safe runs."""
    if "d_safe_hist" not in logs:
        return {"d_safe_mean": None, "d_safe_p50": None, "d_safe_p95": None}
    h = np.asarray(logs["d_safe_hist"]).sum(0).astype(np.float64)
    c = np.cumsum(h) / h.sum()
    centres = D_COLLISION + DSAFE_BIN * np.arange(DSAFE_NBINS)
    return {"d_safe_mean": float(np.sum(np.asarray(logs["d_safe_sum"], np.float64)) / np.sum(logs["d_safe_n"])),
            "d_safe_p50": float(centres[np.searchsorted(c, 0.50)]), "d_safe_p95": float(centres[np.searchsorted(c, 0.95)]),
            **vclosing_stats(logs)}


def vclosing_stats(logs):
    """amend19: mean / p50 / p95 of v_closing over the valid neighbour rows (mean exact; percentiles at 0.05 m/s) + the
    summed histogram. Empty for runs without the closing margin."""
    if "v_closing_hist" not in logs:
        return {}
    h = np.asarray(logs["v_closing_hist"]).sum(0).astype(np.int64)
    c = np.cumsum(h) / h.sum()
    centres = VCLOSE_BIN * np.arange(VCLOSE_NBINS)
    return {"v_closing_mean": float(np.sum(np.asarray(logs["v_closing_sum"], np.float64)) / np.sum(logs["d_safe_n"])),
            "v_closing_p50": float(centres[np.searchsorted(c, 0.50)]),
            "v_closing_p95": float(centres[np.searchsorted(c, 0.95)]),
            "v_closing_frac_zero_bin": float(h[0] / h.sum()), "v_closing_hist": h.tolist(),
            "v_closing_hist_bin_mps": VCLOSE_BIN}


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
        # pair-weighted (every near pair-step counts once; steps without near pairs do not enter as 0)
        "closedloop_xhat_mse_near_pairw": (float(np.sum(np.asarray(logs["xhat_err_near"], np.float64) * logs["n_near"])
                                                 / max(int(np.sum(logs["n_near"])), 1)) if "n_near" in logs else None),
        "near_pair_steps": int(np.sum(logs["n_near"])) if "n_near" in logs else None,
        "integral_W_true": float(np.sum(np.mean(c_w * np.asarray(logs["rho_true"]) ** 2, 1)) * dt_exec),
        **dsafe_stats(logs),
    }
    return out, {"t": t, "arrived_frac": frac, "r_goal_dist": r, "first_arrival_idx": first}
