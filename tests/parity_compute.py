"""Computes a fixed battery of outputs with EITHER the old code (MUSC_MFG @ physical-refactor-final,
exported by tests/test_parity.py) or the ported code in this repo, and writes them to one npz.
Run as a standalone process per side so the two module trees (both define `mfg`, `env`, and
`control`) never share an interpreter.

    python tests/parity_compute.py --side old --old-root <export dir> --out old.npz
    python tests/parity_compute.py --side new --out new.npz
"""
import argparse
import os
import sys

import numpy as np


def load_modules(side, old_root):
    if side == "old":
        sys.path[:0] = [old_root, os.path.join(old_root, "scratchpad", "physical_refactor")]
        from env.base import Grid
        from env.geometry import SingleGate
        import mfg.hjb as hjb
        import mfg.transport as transport
        import mfg.fictitious_play as fp
        import cesaro_fp as cesaro
        import params
        import control as law
        import perception
        import phase2d_closedloop as roll
        import phys_rollout
        roll_consts = (phys_rollout.DT_EXEC, phys_rollout.NU)
        sample_initial = None
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from env.base import Grid
        from env.geometry import SingleGate
        import mfg.hjb as hjb
        import mfg.transport as transport
        import mfg.fictitious_play as fp
        import mfg.cesaro as cesaro
        import control.params_legacy as params
        import control.law as law
        import compensation.perception as perception
        import env.legacy_rollout as roll
        roll_consts = (roll.DT_EXEC, roll.NU)
        sample_initial = roll.sample_initial
    return dict(Grid=Grid, SingleGate=SingleGate, hjb=hjb, transport=transport, fp=fp,
                cesaro=cesaro, params=params, law=law, perception=perception, roll=roll,
                roll_consts=roll_consts, sample_initial=sample_initial)


def compute(M, weights_path):
    import jax
    import jax.numpy as jnp

    out = {}
    P = M["params"]
    for N in (25, 50, 100, 150, 200):
        out[f"params/arena_{N}"] = np.array(P.arena_for_N(N))
        out[f"params/horizon_{N}"] = np.array(P.horizon_for_N(N))
        out[f"params/spread_{N}"] = np.array(P.source_spread_for_N(N))
        out[f"params/tframe_{N}"] = np.array(P.t_frame(N))
    for name in ("R_D", "D_COLLISION", "R_BARRIER", "D_STAR", "RHO_STAR", "RHO_NOM", "V_NOM",
                 "HORIZON_MULT", "SIGMA_R", "SIGMA_THETA", "T_SLOT", "H_BANDWIDTH", "R_SENSE",
                 "V_MAX", "F_BARRIER_MAX", "C_DENS"):
        out[f"params/{name}"] = np.array(getattr(P, name))

    # --- scenario + solver on a small grid ---
    Grid, SingleGate = M["Grid"], M["SingleGate"]
    grid = Grid.build(arena=(4.0, 3.0), step=0.2, horizon=1.0)
    scen = SingleGate(gate_width=0.8, arena=(4.0, 3.0), density_scale=12.0, source_spread=0.5)
    v_goal, v_obs = scen.potentials(grid)
    m0 = scen.initial_density(grid)
    out["scen/v_goal"], out["scen/v_obs"], out["scen/m0"] = map(np.asarray, (v_goal, v_obs, m0))

    hjb = M["hjb"]
    w_fn = lambda m: 5.0 * jnp.maximum(m, 0.0) ** 2
    cp = hjb.CostParams(V_goal=v_goal, V_obs=v_obs, W_fn=w_fn)
    m_b = jnp.broadcast_to(m0, (grid.n_steps + 1, grid.nx, grid.nz))
    value, policy = hjb.solve_hjb(m_b, cp, grid)
    out["hjb/value"], out["hjb/policy"] = np.asarray(value), np.asarray(policy)

    out["transport/one_step"] = np.asarray(M["transport"].push_forward(m0, policy[0], grid, grid.dt))
    fp = M["fp"]
    out["fp/rollout_density"] = np.asarray(fp._rollout_density(m0, policy, grid))
    m_eq, pol_eq, diag = fp.solve_equilibrium(scen, cp, grid, damping=0.01, max_iters=5, tol=1e-12)
    out["fp/m"], out["fp/policy"] = np.asarray(m_eq), np.asarray(pol_eq)
    out["fp/residuals"] = np.array(diag.residual_history)
    m_c, pol_c, cdiag = M["cesaro"].solve_equilibrium_cesaro(
        scen, cp, grid, damping=0.01, max_iters=6, tol=1e-12, burn_in=2)
    out["cesaro/m"], out["cesaro/policy"] = np.asarray(m_c), np.asarray(pol_c)
    out["cesaro/residuals"] = np.array(cdiag.residual_history)
    out["cesaro/cesaro_residuals"] = np.array(cdiag.cesaro_residual_history)

    # static fields in the format the controller consumes
    from env.base import grid_gradient
    zero_cp = hjb.CostParams(V_goal=v_goal, V_obs=v_obs, W_fn=lambda m: 0.0 * m)
    val_phi, pol_phi = hjb.solve_hjb(jnp.zeros_like(m_b), zero_cp, grid)
    m_star_grad = jax.vmap(lambda f: grid_gradient(f, grid.step))(m_c)
    sdata = {"phi_grad_field": pol_phi[0], "m_star": m_c, "m_star_grad_field": m_star_grad,
             "m_star_dt": grid.dt, "grid": grid, "arena": (4.0, 3.0), "scenario": scen}
    field_params = dict(phi_grad_field=sdata["phi_grad_field"], m_star=m_c,
                        m_star_grad_field=m_star_grad, m_star_dt=grid.dt, grid_step=grid.step,
                        c_dens=1.0)

    # --- perception (all arms) + control law on a synthetic history ---
    N, T = 12, 60
    key = jax.random.PRNGKey(1234)
    k1, k2 = jax.random.split(key)
    base = jax.random.uniform(k1, (N, 2)) * jnp.array([4.0, 3.0])
    vel = jax.random.normal(k2, (N, 2)) * 0.5
    history = base[None] + vel[None] * (jnp.arange(T) * 0.01)[:, None, None]
    d = np.load(weights_path)
    pp = {k: jnp.asarray(d[k]) for k in d.files}
    perc = M["perception"]
    out["perc/tdma_age"] = np.asarray(perc.tdma_age(0.37, N))
    out["perc/learned_features"] = np.asarray(
        perc.learned_features(history, 45, 0.01, N, jax.random.PRNGKey(7)))
    for cond in ("fresh", "real", "cv", "pred", "learned", "smith"):
        x_hat = perc.perceive(cond, history, 45, 0.01, N, jax.random.PRNGKey(99),
                              predictor_params=pp, field_params=field_params)
        out[f"perc/{cond}"] = np.asarray(x_hat)
    law = M["law"]
    x_hat = jnp.asarray(out["perc/real"])
    u, cdiag2 = law.control_law(history[45], x_hat, 0.45, N, sdata["phi_grad_field"], m_c,
                                m_star_grad, grid.dt, grid.step)
    out["law/u"] = np.asarray(u)
    for k_, v_ in cdiag2.items():
        out[f"law/diag_{k_}"] = np.asarray(v_)
    out["law/barrier"] = np.asarray(law.bounded_barrier_force_all(history[45], x_hat, N))

    # --- closed-loop legacy step, all four arms, short horizon ---
    dt_exec, nu = M["roll_consts"]
    out["roll/consts"] = np.array([dt_exec, nu])
    roll = M["roll"]
    n_steps = 40
    for cond in ("real", "cv", "learned", "smith"):
        step_fn = roll.make_step_fn(cond, N, sdata, pp, field_params)
        key = jax.random.PRNGKey(0 * 1_000_003 + N * 1_009 + 3 * 7 + roll.SEED_TAGS[cond])
        if M["sample_initial"] is not None:
            pos0, key = M["sample_initial"](key, m0, grid, N)
        else:  # verbatim copy of phase2d_closedloop.run's inner sample_initial
            key, sub = jax.random.split(key)
            flat_probs = (m0 / jnp.sum(m0)).reshape(-1)
            idx = jax.random.choice(sub, flat_probs.shape[0], shape=(N,), p=flat_probs)
            ix, iz = jnp.unravel_index(idx, m0.shape)
            pos0 = jnp.stack([grid.xs[ix], grid.zs[iz]], axis=-1)
        carry0 = (pos0, jnp.zeros((n_steps, N, 2)), key)
        (pos_f, hist_f, _), (md, cg) = jax.lax.scan(step_fn, carry0, jnp.arange(n_steps))
        out[f"roll/{cond}/pos0"] = np.asarray(pos0)
        out[f"roll/{cond}/history"] = np.asarray(hist_f)
        out[f"roll/{cond}/min_dist"] = np.asarray(md)
        out[f"roll/{cond}/congestion"] = np.asarray(cg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("old", "new"), required=True)
    ap.add_argument("--old-root")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = load_modules(a.side, a.old_root)
    np.savez(a.out, **compute(M, a.weights))
