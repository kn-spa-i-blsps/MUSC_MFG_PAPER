"""Project A locked parameters (prereg/prereg_projectA.md §3 + amendments 1, 2). Constants shared
with the ported line are re-exported from params_legacy unchanged (parity-locked); only the
density-constant scaling (RHO_NOM=1.0) and the goal region r_goal(N) are new.

SIGMA_THETA stays math.radians(2.0) = 0.034907 rad, which the prereg lists rounded as 0.035 rad.
"""
import math

from control.params_legacy import (  # noqa: F401  (re-exported)
    R_D, D_COLLISION, R_BARRIER, D_STAR, V_NOM, HORIZON_MULT, SIGMA_R, SIGMA_THETA, T_SLOT,
    H_BANDWIDTH, R_SENSE, V_MAX, C_DENS, t_frame,
)

RHO_NOM = 1.0          # agents/m^2, nominal free-space density (prereg §3)
GATE_RATIO = 0.20      # gate width / arena width, fixed across N
ASPECT = 4.0 / 3.0     # arena W/H
RHO_GOAL = 2.0 / (math.sqrt(3.0) * D_STAR ** 2)  # amend1: hex packing at D_STAR, ~3.21 agents/m^2


def arena_for_N(N: int, rho_nom: float = RHO_NOM) -> tuple[float, float]:
    """(W, H) with W*H = N / rho_nom, aspect 4:3."""
    A = N / rho_nom
    return math.sqrt(A * ASPECT), math.sqrt(A / ASPECT)


def gate_width_for_N(N: int) -> float:
    return GATE_RATIO * arena_for_N(N)[0]


def r_goal_for_N(N: int) -> float:
    """amend1: r_goal(N) = sqrt(N / (pi * RHO_GOAL))."""
    return math.sqrt(N / (math.pi * RHO_GOAL))


def horizon_for_N(N: int) -> float:
    return HORIZON_MULT * arena_for_N(N)[0] / V_NOM
