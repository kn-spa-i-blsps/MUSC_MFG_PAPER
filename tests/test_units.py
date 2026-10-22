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


def test_exact_qp_matches_bruteforce():
    """Exact 2-D solver vs dense grid search on random QPs (feasible and infeasible)."""
    from control.cbf import solve_exact, LOOSE
    rng = np.random.default_rng(0)
    g = np.linspace(-PRM.u_box, PRM.u_box, 801)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    f = jax.jit(lambda a, l, u: solve_exact(a, l, u, PRM))
    for trial in range(30):
        C = 14
        A = rng.normal(size=(C, 2))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        lo = rng.uniform(-4.0, 1.5 if trial % 3 else 4.0, size=C)
        A[10:] = 0.0
        lo[10:] = -LOOSE
        u_nom = rng.uniform(-6, 6, size=2)
        u, t, _ = f(jnp.asarray(A), jnp.asarray(lo), jnp.asarray(u_nom))
        u, t = np.asarray(u), float(t)
        v = np.max(np.maximum(lo[None, :10] - G @ A[:10].T, 0.0), axis=1)
        assert t <= v.min() + 1e-3                              # minimal uniform violation
        assert np.all(A[:10] @ u >= lo[:10] - t - 1e-4)          # feasible for relaxed rows
        assert np.all(np.abs(u) <= PRM.u_box + 1e-6)
        if t == 0.0:                                             # hard QP feasible: exact projection
            ok = v == 0.0
            best = G[ok][np.argmin(np.sum((G[ok] - u_nom) ** 2, 1))]
            assert np.linalg.norm(u - u_nom) <= np.linalg.norm(best - u_nom) + 1e-4
        n_feasible_cases = locals().get("n_feasible_cases", 0) + (t == 0.0)
    assert n_feasible_cases >= 5


def test_exact_qp_near_parallel_rows_finite():
    """Regression (smoke attempt 2 NaN): near-duplicate neighbour rows must not produce NaN."""
    from control.cbf import solve_exact
    f = jax.jit(lambda a, l, u: solve_exact(a, l, u, PRM))
    rng = np.random.default_rng(1)
    for _ in range(200):
        base = rng.normal(size=2)
        A = base[None] + rng.normal(size=(14, 2)) * 10.0 ** rng.uniform(-8, -3)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        A[8:] = rng.normal(size=(6, 2))
        A[8:] /= np.linalg.norm(A[8:], axis=1, keepdims=True)
        lo = rng.uniform(-3, 1, size=14)
        u, t, _ = f(jnp.asarray(A, jnp.float32), jnp.asarray(lo, jnp.float32),
                    jnp.asarray(rng.uniform(-6, 6, 2), jnp.float32))
        assert np.all(np.isfinite(np.asarray(u))) and np.isfinite(float(t))


# ---------- P1 regression: walls get their own slack ----------
GATE_WALL = np.array([[[3.0, 0.0], [3.0, 1.0]], [[3.0, 2.0], [3.0, 3.0]]])   # gate z in (1, 2)
ADVERSARIAL = np.array([[0.0, 0.0], [-0.05, 0.0], [0.0, 0.30], [0.0, -0.30], [-0.30, 0.05]])


def _seg_dist(p):
    a, b = GATE_WALL[:, 0], GATE_WALL[:, 1]
    t = np.clip(np.sum((p - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0, 1)
    return np.min(np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=-1))


def _push_into_wall(solver, steps=300, dt=0.01):
    """Agent 0 beside the lower wall segment, u_nom straight into it, and neighbour ESTIMATES glued
    to it (stale/noisy-estimate worst case) so its neighbour rows are infeasible every step."""
    prm = PRM._replace(qp_solver=solver)
    f = jax.jit(cbf_filter, static_argnames=("arena", "prm"))
    p0 = np.array([2.75, 0.5])
    others = np.array([[0.5, 2.5], [1.0, 2.5], [1.5, 2.5], [2.0, 2.5]])
    u_nom = jnp.zeros((5, 2)).at[0].set(jnp.array([5.0, 0.0]))
    traj, nb_slack, hard_slack = [p0.copy()], [], []
    for _ in range(steps):
        pos = np.vstack([p0, others])
        x_hat = np.broadcast_to(pos[None], (5, 5, 2)).copy()
        x_hat[0] = p0 + ADVERSARIAL
        u, info, _ = f(jnp.asarray(pos), jnp.asarray(x_hat), u_nom, jnp.array([5.0, 1.5]), (6.0, 3.0),
                       GATE_WALL, init_warm_start(5, prm, 2), prm)
        nb_slack.append(float(info["slack_max"][0]))
        hard_slack.append(float(info["slack_hard"][0]))
        p0 = p0 + dt * np.asarray(u[0])
        traj.append(p0.copy())
    return np.array(traj), np.array(nb_slack), np.array(hard_slack)


def test_wall_never_crossed_when_neighbour_qp_infeasible():
    traj, nb_slack, hard_slack = _push_into_wall("exact_hard")
    assert (nb_slack > 1e-3).mean() > 0.9                   # neighbour rows really infeasible
    assert hard_slack.max() == 0.0                          # walls never relaxed
    assert (traj[:, 0] < 3.0).all()                         # never through the wall
    assert min(_seg_dist(p) for p in traj) >= PRM.r_wall - 1e-4


def test_legacy_shared_slack_breaches_wall_clearance():
    """Documents the P1 bug (smoke attempt 3): with one slack for all rows the shared slack loosens the
    wall row and the same agent is driven into the wall body (in the smoke, with 8 m/s steps, through)."""
    traj, _, hard_slack = _push_into_wall("exact")
    assert hard_slack.max() > 1.0
    assert min(_seg_dist(p) for p in traj) < 0.05


def test_exact_hard_matches_lp():
    """Lexicographic solver vs scipy LP (phase 0 / phase 1 optimal slacks) and vs grid-search projection
    when everything is feasible."""
    from scipy.optimize import linprog
    from control.cbf import solve_exact_hard, LOOSE
    rng = np.random.default_rng(2)
    k, ub = 8, PRM.u_box
    f = jax.jit(lambda a, l, u: solve_exact_hard(a, l, u, PRM, k))
    g = np.linspace(-ub, ub, 801)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def lp(Amin, lomin, Ahold, lohold):        # min t s.t. Amin u + t >= lomin, Ahold u >= lohold
        A_ub = np.vstack([np.hstack([-Amin, -np.ones((len(Amin), 1))]),
                          np.hstack([-Ahold, np.zeros((len(Ahold), 1))])])
        b_ub = np.concatenate([-lomin, -lohold])
        r = linprog([0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=[(-ub, ub), (-ub, ub), (0, None)])
        return r.fun

    n_both_feasible = 0
    for trial in range(40):
        A = rng.normal(size=(14, 2))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        lo = rng.uniform(-4.0, 1.5 if trial % 3 else 4.0, size=14)
        A[6:8] = 0.0
        lo[6:8] = -LOOSE                                    # padded neighbour rows
        u_nom = rng.uniform(-6, 6, size=2)
        u, s, _ = f(jnp.asarray(A), jnp.asarray(lo), jnp.asarray(u_nom))
        u, s = np.asarray(u), np.asarray(s)
        t_s, t_h = float(s[0]), float(s[k])
        t_h_ref = lp(A[k:], lo[k:], np.zeros((0, 2)), np.zeros(0))
        assert t_h == pytest.approx(t_h_ref, abs=1e-3)
        t_s_ref = lp(A[:6], lo[:6], A[k:], lo[k:] - t_h_ref - 1e-6)
        assert t_s == pytest.approx(t_s_ref, abs=1e-3)
        assert np.all(A[k:] @ u >= lo[k:] - t_h - 1e-4)
        assert np.all(A[:6] @ u >= lo[:6] - t_s - 1e-4)
        assert np.all(np.abs(u) <= ub + 1e-6)
        if t_h == 0.0 and t_s == 0.0:
            n_both_feasible += 1
            ok = np.all(G @ A[np.r_[0:6, k:14]].T >= lo[np.r_[0:6, k:14]], axis=1)
            best = G[ok][np.argmin(np.sum((G[ok] - u_nom) ** 2, 1))]
            assert np.linalg.norm(u - u_nom) <= np.linalg.norm(best - u_nom) + 1e-4
    assert n_both_feasible >= 3


def test_exact_hard_near_parallel_rows_finite():
    """Same NaN regression as smoke attempt 2, for the exact_hard solver."""
    from control.cbf import solve_exact_hard
    f = jax.jit(lambda a, l, u: solve_exact_hard(a, l, u, PRM, 8))
    rng = np.random.default_rng(1)
    for _ in range(200):
        base = rng.normal(size=2)
        A = base[None] + rng.normal(size=(14, 2)) * 10.0 ** rng.uniform(-8, -3)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        A[8:] = rng.normal(size=(6, 2))
        A[8:] /= np.linalg.norm(A[8:], axis=1, keepdims=True)
        lo = rng.uniform(-3, 1, size=14)
        u, s, n = f(jnp.asarray(A, jnp.float32), jnp.asarray(lo, jnp.float32),
                    jnp.asarray(rng.uniform(-6, 6, 2), jnp.float32))
        assert np.all(np.isfinite(np.asarray(u))) and np.all(np.isfinite(np.asarray(s)))
        assert int(n) >= 0
