"""Micro layer (prereg §2): per-agent, decentralized CBF-QP on the realized control, from local
sensing only (own position, neighbour estimates x_hat, known static walls). Single-integrator
dynamics x' = u.

    min_{u,s}  1/2 |u - u_nom|^2 + 1/2 rho |s|^2 + slack_lin * sum(s)   (L1 part: exact penalty)
    s.t.       2 (x_i - x_hat_j) . u + s_j >= -a_ij * gamma * h_ij,   h_ij = |x_i - x_hat_j|^2 - d_safe^2
               2 (x_i - q_w) . u + s_w    >= -gamma * h_w,           h_w  = |x_i - q_w|^2 - r_wall^2
               arena half-planes (linear h), |u_x|,|u_z| <= V_MAX/sqrt(2),  s >= 0

Neighbour pairs use reciprocal responsibility (each agent of a pair guarantees its share a_ij of the
barrier rate; a_ij + a_ji = 1). Liveness: goal-biased priority — the agent nearer the goal gets the
larger share a_ij (more freedom), the one behind yields. Fallback: soft slack s (never an emergency
stop) — a QP whose hard version is infeasible returns the minimum-violation control. For h=|d|^2-r^2
the Euler step only adds dt^2|du|^2 >= 0 to h, so the linear constraint is conservative in discrete
time. Solved with jaxopt BoxOSQP, vmapped over agents, warm-started from the previous step.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

BIG = 1e6
LOOSE = 1e3   # never-binding bound (|u| <= u_box ~ 5.7); keeps the OSQP data well scaled


class CBFParams(NamedTuple):
    d_safe: float      # pair safe distance (D_COLLISION + margin)
    r_wall: float      # wall clearance (wall half-thickness + airframe radius)
    r_body: float      # arena-boundary clearance
    gamma: float       # class-K gain, 1/s
    rho: float         # slack quadratic weight
    slack_lin: float   # slack linear (exact-penalty) weight: s = 0 whenever the hard QP is feasible
    k_nb: int          # neighbours per QP (nearest by estimated distance)
    r_sense: float
    prio_gain: float   # 1/m, goal-biased priority sharpness
    prio_min: float    # share clip [prio_min, 1-prio_min]
    u_box: float       # per-axis velocity bound
    tol: float
    maxiter: int


def _closest_on_segments(p, segs):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = jnp.clip(jnp.sum((p[None] - a) * ab, -1) / jnp.maximum(jnp.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    return a + t[:, None] * ab                                         # (S,2)


def build_rows(p_i, x_hat_i, self_idx, goal, arena, segs, prm: CBFParams):
    """Constraint rows (a (C,2), lower (C,)) for agent i: k_nb neighbours, S walls, 4 boundaries."""
    if x_hat_i.shape[0] < prm.k_nb:                                    # pad with never-valid rows
        pad = prm.k_nb - x_hat_i.shape[0]
        x_hat_i = jnp.concatenate([x_hat_i, jnp.full((pad, 2), BIG)], 0)
    N = x_hat_i.shape[0]
    rel = p_i[None] - x_hat_i                                          # (N,2)
    d2 = jnp.sum(rel ** 2, -1)
    d2 = jnp.where(jnp.arange(N) == self_idx, jnp.inf, d2)
    neg, idx = jax.lax.top_k(-d2, prm.k_nb)
    d2_k = -neg
    rel_k = rel[idx]
    valid = d2_k < prm.r_sense ** 2                                    # (inf / padded -> False)
    h = d2_k - prm.d_safe ** 2
    r_i = jnp.linalg.norm(goal - p_i)
    r_j = jnp.linalg.norm(goal[None] - x_hat_i[idx], axis=-1)
    share = jax.nn.sigmoid(prm.prio_gain * (r_j - r_i))
    share = jnp.clip(share, prm.prio_min, 1.0 - prm.prio_min)
    share = jnp.where(h >= 0, share, 0.5)
    a_nb = 2.0 * rel_k
    lo_nb = jnp.where(valid, -share * prm.gamma * h, -BIG)

    q = _closest_on_segments(p_i, segs)
    rel_w = p_i[None] - q
    h_w = jnp.sum(rel_w ** 2, -1) - prm.r_wall ** 2
    a_w = 2.0 * rel_w
    lo_w = -prm.gamma * h_w

    W, H = arena
    r = prm.r_body
    a_b = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h_b = jnp.array([p_i[0] - r, W - r - p_i[0], p_i[1] - r, H - r - p_i[1]])
    lo_b = -prm.gamma * h_b

    A = jnp.concatenate([a_nb, a_w, a_b], 0)
    lo = jnp.concatenate([lo_nb, lo_w, lo_b], 0)
    active = jnp.concatenate([valid, jnp.ones(a_w.shape[0] + 4, bool)])
    # row normalization a.u >= lo  <=>  (a/|a|).u >= lo/|a|  (same feasible set, unit-scaled data)
    na = jnp.maximum(jnp.linalg.norm(A, axis=-1), 1e-9)
    A = jnp.where(active[:, None], A / na[:, None], 0.0)
    lo = jnp.where(active, jnp.clip(lo / na, -LOOSE, LOOSE), -LOOSE)
    return A, lo, idx, valid


def _qp_matrices(A_c, lo_c, u_nom, prm: CBFParams):
    C = A_c.shape[0]
    nv = 2 + C
    Q = jnp.diag(jnp.concatenate([jnp.ones(2), prm.rho * jnp.ones(C)]))
    c = jnp.concatenate([-u_nom, prm.slack_lin * jnp.ones(C)])
    A = jnp.concatenate([
        jnp.concatenate([A_c, jnp.eye(C)], 1),                         # CBF rows with slack
        jnp.concatenate([jnp.eye(2), jnp.zeros((2, C))], 1),           # velocity box
        jnp.concatenate([jnp.zeros((C, 2)), jnp.eye(C)], 1),           # s >= 0
    ], 0)
    lower = jnp.concatenate([lo_c, -prm.u_box * jnp.ones(2), jnp.zeros(C)])
    upper = jnp.concatenate([LOOSE * jnp.ones(C), prm.u_box * jnp.ones(2), LOOSE * jnp.ones(C)])
    return Q, c, A, lower, upper, nv


def solve_one(A_c, lo_c, u_nom, init, prm: CBFParams):
    Q, c, A, lower, upper, _ = _qp_matrices(A_c, lo_c, u_nom, prm)
    solver = BoxOSQP(tol=prm.tol, maxiter=prm.maxiter, check_primal_dual_infeasability=False)
    sol = solver.run(init_params=init, params_obj=(Q, c), params_eq=A, params_ineq=(lower, upper))
    return sol.params, sol.state.iter_num


def cbf_filter(positions, x_hat, u_nom, goal, arena, segs, init, prm: CBFParams):
    """positions (N,2), x_hat (N,N,2), u_nom (N,2). init: batched BoxOSQP KKTSolution or None.
    Returns (u (N,2), info dict, kkt for warm start)."""
    N = positions.shape[0]
    goal = jnp.asarray(goal)
    segs = jnp.asarray(segs)
    u_nom_box = jnp.clip(u_nom, -prm.u_box, prm.u_box)
    rows = jax.vmap(lambda p, xh, i: build_rows(p, xh, i, goal, arena, segs, prm))(
        positions, x_hat, jnp.arange(N))
    A_c, lo_c = rows[0], rows[1]
    kkt, iters = jax.vmap(lambda a, l, un, ini: solve_one(a, l, un, ini, prm))(A_c, lo_c, u_nom_box, init)
    z = kkt.primal[0]
    u = jnp.clip(z[:, :2], -prm.u_box, prm.u_box)
    s = jnp.maximum(z[:, 2:], 0.0)
    residual = jnp.einsum("ncd,nd->nc", A_c, u) + s - lo_c            # >= 0 when satisfied
    info = {"slack_max": jnp.max(s, axis=1),
            "intervention": jnp.linalg.norm(u - u_nom_box, axis=-1),
            "qp_iters": iters,
            "row_violation": jnp.maximum(-jnp.min(residual, axis=1), 0.0)}
    return u, info, kkt


def init_warm_start(N, prm: CBFParams, n_walls: int):
    """Zero-initialized batched KKTSolution matching cbf_filter's QP shape."""
    from jaxopt.base import KKTSolution
    C = prm.k_nb + n_walls + 4
    nv, nr = 2 + C, C + 2 + C
    return KKTSolution(primal=(jnp.zeros((N, nv)), jnp.zeros((N, nr))), dual_eq=jnp.zeros((N, nr)),
                       dual_ineq=(jnp.zeros((N, nr)), jnp.zeros((N, nr))))
