"""Unit tests required before the smoke run (task KROK 2): (a) density-constant scaling,
(b) CBF head-on no-collision, (c) single-agent goal-hold, (d) CBF-QP slack fallback."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from control.cbf import CBFParams, cbf_filter, init_warm_start
from control.params import (RHO_NOM, D_COLLISION, arena_for_N, gate_width_for_N, r_goal_for_N,
                            horizon_for_N)
from env.scenarios import SingleGateA

PRM = CBFParams(d_safe=0.40, r_wall=0.20, r_body=0.15, gamma=20.0, rho=1.0, slack_lin=100.0, k_nb=4,
                r_sense=5.0, prio_gain=2.0, prio_min=0.25, u_box=8.0 / math.sqrt(2), tol=1e-6, maxiter=4000)
FAR_WALL = np.array([[[-50.0, -50.0], [-50.0, -49.0]]])
_filter = jax.jit(cbf_filter, static_argnames=("arena", "prm"))


# ---------- (a) density-constant ----------
PREREG_ARENA = {25: (5.77, 4.33), 50: (8.16, 6.12), 100: (11.55, 8.66), 150: (14.14, 10.61),
                200: (16.33, 12.25)}
PREREG_RGOAL = {25: 1.58, 50: 2.23, 100: 3.15, 150: 3.86, 200: 4.45}


@pytest.mark.parametrize("N", sorted(PREREG_ARENA))
def test_density_constant(N):
    W, H = arena_for_N(N)
    assert N / (W * H) == pytest.approx(RHO_NOM, rel=1e-12)
    assert (round(W, 2), round(H, 2)) == PREREG_ARENA[N]
    assert gate_width_for_N(N) / W == pytest.approx(0.20)
    assert r_goal_for_N(N) == pytest.approx(PREREG_RGOAL[N], abs=0.01)  # amend1 lists 4.4551 as 4.45
    sc = SingleGateA(N, w_goal=1.0, w_hold=1.0)
    assert sc.arena == (W, H)                                  # the scenario owns THE arena
    assert N / sc.free_area() == pytest.approx(RHO_NOM, rel=0.02)
    pos = np.asarray(sc.initial_positions(jax.random.PRNGKey(0)))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(N) * 1e9
    assert d.min() > PRM.d_safe                                # safe set holds at t=0
    assert (pos[:, 0] < sc.wall_x).all()


# ---------- (b) CBF head-on ----------
def _sim_pair(p0, goal, steps=600, dt=0.01, v=1.5):
    pos = jnp.asarray(p0)
    warm = init_warm_start(2, PRM, 1)
    dmin = np.inf
    for _ in range(steps):
        rel = pos[::-1] - pos
        u_nom = v * rel / jnp.linalg.norm(rel, axis=-1, keepdims=True)   # straight at each other
        x_hat = jnp.broadcast_to(pos[None], (2, 2, 2))
        u, info, warm = _filter(pos, x_hat, u_nom, goal, (6.0, 3.0), FAR_WALL, warm, PRM)
        pos = pos + dt * u
        dmin = min(dmin, float(jnp.linalg.norm(pos[0] - pos[1])))
    return dmin, info


def test_cbf_head_on_no_collision():
    dmin, info = _sim_pair([[1.0, 1.5], [5.0, 1.5]], jnp.array([6.0, 1.5]))
    assert dmin >= D_COLLISION
    assert dmin >= PRM.d_safe - 1e-3
    assert float(info["slack_max"].max()) < 1e-4


def test_cbf_head_on_offset_no_collision():
    dmin, _ = _sim_pair([[1.0, 1.45], [5.0, 1.55]], jnp.array([6.0, 1.5]))
    assert dmin >= PRM.d_safe - 1e-3


# ---------- (d) QP feasibility / slack ----------
def _filter_one(p_i, nbrs, u_nom):
    pos = jnp.asarray([p_i] + nbrs)
    N = pos.shape[0]
    x_hat = jnp.broadcast_to(pos[None], (N, N, 2))
    u_n = jnp.zeros((N, 2)).at[0].set(jnp.asarray(u_nom))
    u, info, _ = _filter(pos, x_hat, u_n, jnp.array([5.0, 1.5]), (6.0, 3.0), FAR_WALL,
                         init_warm_start(N, PRM, 1), PRM)
    return np.asarray(u[0]), {k: np.asarray(v)[0] for k, v in info.items()}


def test_qp_feasible_passthrough():
    u, info = _filter_one([2.0, 1.5], [[4.0, 1.5]], [1.0, 0.3])
    np.testing.assert_allclose(u, [1.0, 0.3], atol=1e-3)
    assert info["slack_max"] < 1e-4


def test_qp_infeasible_uses_slack_not_stop():
    # squeezed left/right below d_safe: the two separation constraints demand opposite u_x,
    # so the hard QP is infeasible; the free direction (z) must stay usable (no emergency stop)
    u, info = _filter_one([2.0, 1.5], [[1.65, 1.5], [2.35, 1.5]], [0.0, 1.0])
    assert np.all(np.isfinite(u))
    assert info["slack_max"] > 1e-3
    assert u[1] > 0.9
    assert abs(u[0]) < 1e-2


def test_qp_infeasible_boxed_in_bounded():
    u, info = _filter_one([2.0, 1.5], [[1.65, 1.5], [2.35, 1.5], [2.0, 1.15], [2.0, 1.85]], [3.0, 0.0])
    assert np.all(np.isfinite(u)) and np.all(np.abs(u) <= PRM.u_box + 1e-6)
    assert info["slack_max"] > 1e-3


# ---------- (c) goal-hold (single agent, full macro + CBF pipeline) ----------
def test_single_agent_reaches_and_holds_goal():
    from env.base import Grid
    from env.rollout import make_rollout, summarize
    from mfg.equilibrium import MacroCost, solve_macro
    from experiments.smoke import DEFAULT_COST
    N_geom = 25
    sc = SingleGateA(N_geom, n_agents=1, w_goal=DEFAULT_COST["w_goal"], w_hold=DEFAULT_COST["w_hold"])
    grid = Grid.build(sc.arena, 0.2, horizon_for_N(N_geom))
    cost = MacroCost(DEFAULT_COST["c_w"], DEFAULT_COST["w_bar"], DEFAULT_COST["m_bar"], DEFAULT_COST["f_cap"])
    sol = solve_macro(sc, grid, cost, damping=0.02, max_iters=20, tol=1e-4, burn_in=5)
    assert not sol["nan"]
    cfg = dict(dt_exec=0.01, n_steps=int(round(grid.horizon / 0.01)), nu=0.0, cbf_on=True,
               c_w=cost.c_w, w_bar=cost.w_bar, m_bar=cost.m_bar, c_dens=1.0,
               cbf=PRM._replace(k_nb=1)._asdict())
    logs = make_rollout("real", sc, cfg)(jax.random.PRNGKey(0), sol)
    m, series = summarize(logs, sc, 0.01)
    first = int(series["first_arrival_idx"][0])
    assert first >= 0, "never reached the goal region"
    assert first * 0.01 < 0.5 * grid.horizon
    assert np.all(series["r_goal_dist"][first:, 0] <= sc.r_goal), "left the goal region after arrival"
    assert m["completeness_final"] == 1.0
