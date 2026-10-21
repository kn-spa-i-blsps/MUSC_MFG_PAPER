"""Ported, unmodified closed-loop step of the physical_refactor line (MUSC_MFG @
physical-refactor-final): scratchpad/physical_refactor/phase2d_closedloop.py::make_step_fn and
::run's sample_initial / per-seed key, with DT_EXEC/NU from phys_rollout.py. Kept as the parity
reference for the ladder arms (real/cv/learned/smith) under the old density-tracking control law;
the Project A rollout lives in env/rollout.py.
"""
import jax
import jax.numpy as jnp

from control.params_legacy import C_DENS, V_MAX
from compensation.perception import perceive
from control.law import control_law, local_density_and_grad, sample_m_star_and_grad

NU = 0.091
DT_EXEC = 0.01    # s

SEED_TAGS = {"real": 2, "cv": 6, "learned": 10, "smith": 14}


def make_step_fn(condition, N, sdata, predictor_params, field_params):
    phi_grad_field = sdata["phi_grad_field"]
    m_star = sdata["m_star"]
    m_star_grad_field = sdata["m_star_grad_field"]
    m_star_dt = sdata["m_star_dt"]
    grid_step = sdata["grid"].step
    W, H = sdata["arena"]
    arena_arr = jnp.array([W, H])
    lo = jnp.zeros(2)
    self_mask = 1.0 - jnp.eye(N)

    def step_fn(carry, k):
        positions, history, key = carry
        history = history.at[k].set(positions)

        key, pk = jax.random.split(key)
        x_hat = perceive(condition, history, k, DT_EXEC, N, pk,
                          predictor_params=predictor_params, field_params=field_params)
        t_sim = k * DT_EXEC
        u, diag = control_law(positions, x_hat, t_sim, N, phi_grad_field, m_star,
                               m_star_grad_field, m_star_dt, grid_step)
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        u_real = u * jnp.minimum(1.0, V_MAX / jnp.maximum(u_norm, 1e-9))

        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, positions.shape) * jnp.sqrt(2.0 * NU * DT_EXEC)
        positions_new = jnp.clip(positions + DT_EXEC * u_real + noise, lo, arena_arr)

        d_true = jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        d_masked = jnp.where(self_mask > 0, d_true, jnp.inf)
        min_dist = jnp.min(d_masked, axis=1)

        true_oracle_xhat = jnp.broadcast_to(positions[None, :, :], (N, N, 2))
        rho_true, _ = local_density_and_grad(positions, true_oracle_xhat, N)
        m_star_val, _ = sample_m_star_and_grad(positions, t_sim, m_star, m_star_grad_field, m_star_dt, grid_step)
        congestion_cost = C_DENS * (rho_true - m_star_val) ** 2

        new_carry = (positions_new, history, key)
        y = (min_dist, congestion_cost)
        return new_carry, y

    return step_fn


def sample_initial(key, M0, grid, N):
    key, sub = jax.random.split(key)
    flat_probs = (M0 / jnp.sum(M0)).reshape(-1)
    idx = jax.random.choice(sub, flat_probs.shape[0], shape=(N,), p=flat_probs)
    ix, iz = jnp.unravel_index(idx, M0.shape)
    return jnp.stack([grid.xs[ix], grid.zs[iz]], axis=-1), key


def seed_key(N, s, condition):
    return jax.random.PRNGKey(0 * 1_000_003 + N * 1_009 + s * 7 + SEED_TAGS[condition])
