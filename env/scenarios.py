"""Project A scenarios. A scenario owns the ONE arena used by both the offline MFG solve and the
rollout (prereg §9 (2)): soft macro potentials (V_goal, V_obs), the initial density m0 and matching
initial agent positions, the goal (point + r_goal), and the hard wall geometry used by the CBF.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from control.params import arena_for_N, gate_width_for_N, r_goal_for_N, D_STAR
from env.base import Grid

WALL_HALF_THICKNESS = 0.05  # m, physical (hard) wall half-thickness seen by the CBF


class SingleGateA:
    """Vertical wall at x=W0/2 with one centred gate of width GATE_RATIO*W0, (W0, H) = arena_for_N(N).
    Start region left of the wall. Goal placement:
      goal_gap_mult=None (amend1 note): goal disc centred in the post-gate chamber at (0.75W0, H/2).
      goal_gap_mult=k (amend4): downstream chamber -- goal waypoint at (wall_x + k*r_goal, H/2), arena
        extended to W = x_goal + r_goal + (H - 2 r_goal)/2; source chamber and gate unchanged.
    goal_spacing=s (amend9): r_goal from the hex-packing density 2/(sqrt3 s^2) instead of RHO_GOAL (D_STAR).
    goal_clearance=c (amend9, with goal_gap_mult): clearance c from the goal disc to the top, bottom and
      right walls; the arena height becomes max(H0, 2 r_goal + 2c), the source chamber (W0/2 x H0) and
      the gate are kept, centred, and closed off from the extra height by two horizontal hard walls.
    goal_attractor_shift=k (amend10): the cone (progress) term of V_goal pulls to an attractor point
      k*r_goal downstream (+x) of the disc centre (k=1: far edge of the disc); the tanh hold step at
      r_goal, the success region, the local goal observation and the CBF priority keep the disc centre.
      None (default) = attractor at the disc centre (amend1-9).
    goal_shape (amend7): "conical" (default) -- V_goal = cone to the waypoint + tanh hold step at r_goal;
      "flat" -- flat-bottomed: both terms act on relu(r - r_goal) only, so grad V_goal = 0 inside the
      disc and (up to the 0.05 m cone rounding) equals the conical gradient outside it.
    The goal is a settable waypoint (static in A), read by V_goal, the local goal observation and the
    CBF priority."""

    def __init__(self, N: int, *, w_goal: float, w_hold: float, goal_softness: float = 0.25,
                 obs_penalty: float = 200.0, wall_thickness: float = 0.3,
                 edge_softness: float = 0.25, start_margin: float = 0.4,
                 start_wall_gap: float = 0.6, start_softness: float = 0.2,
                 n_agents: int | None = None, goal_gap_mult: float | None = None,
                 goal_shape: str = "conical", goal_spacing: float | None = None,
                 goal_clearance: float | None = None, goal_attractor_shift: float | None = None):
        # geometry is sized by N; n_agents (default N) only sets the mass / agent count -- used by
        # the single-agent goal-hold unit test
        self.N_geom = N
        self.N = N if n_agents is None else n_agents
        W, H = arena_for_N(N)
        self.gate_width = gate_width_for_N(N)
        self.wall_x = W / 2.0
        self.H0 = H
        self.r_goal = r_goal_for_N(N) if goal_spacing is None else \
            math.sqrt(N / (math.pi * 2.0 / (math.sqrt(3.0) * goal_spacing ** 2)))
        self.z_off = 0.0                                   # source chamber spans z in [z_off, z_off + H0]
        if goal_gap_mult is None:
            self.arena = (W, H)
            self.waypoint = (0.75 * W, H / 2.0)
        else:
            x_goal = self.wall_x + goal_gap_mult * self.r_goal
            m = (H - 2.0 * self.r_goal) / 2.0 if goal_clearance is None else goal_clearance
            Ha = max(H, 2.0 * self.r_goal + 2.0 * m)
            self.z_off = (Ha - H) / 2.0
            self.arena = (x_goal + self.r_goal + m, Ha)
            self.waypoint = (x_goal, Ha / 2.0)
        self.gate_z = self.arena[1] / 2.0
        k = 0.0 if goal_attractor_shift is None else goal_attractor_shift
        self.attractor = (self.waypoint[0] + k * self.r_goal, self.waypoint[1])
        self.w_goal, self.w_hold, self.goal_softness = w_goal, w_hold, goal_softness
        if goal_shape not in ("conical", "flat"):
            raise ValueError(goal_shape)
        self.goal_shape = goal_shape
        self.obs_penalty, self.wall_thickness, self.edge_softness = obs_penalty, wall_thickness, edge_softness
        self.start_box = (start_margin, self.wall_x - start_wall_gap,
                          self.z_off + start_margin, self.z_off + H - start_margin)
        self.start_softness = start_softness

    @property
    def goal(self):
        return self._waypoint

    @property
    def waypoint(self):
        return self._waypoint

    @waypoint.setter
    def waypoint(self, xz):
        self._waypoint = (float(xz[0]), float(xz[1]))

    def path_length_to_goal(self) -> float:
        """amend5: farthest start-box corner -> gate centre -> farthest point of the goal disc."""
        x0, x1, z0, z1 = self.start_box
        gate = (self.wall_x, self.gate_z)
        far = max(((x0, z0), (x0, z1)), key=lambda c: math.dist(c, gate))
        return math.dist(far, gate) + math.dist(gate, self.goal) + self.r_goal

    def source_area(self) -> float:
        """Source chamber (left of the wall): density-constant reference on the source side."""
        return self.wall_x * self.H0

    def domain(self) -> tuple[float, float]:
        return self.arena

    # --- hard geometry (CBF) ---
    def wall_segments(self) -> np.ndarray:
        """(n_seg, 2, 2) wall centre-line segments [[x0,z0],[x1,z1]]."""
        H = self.arena[1]
        wx, gz, half = self.wall_x, self.gate_z, self.gate_width / 2.0
        segs = [[[wx, 0.0], [wx, gz - half]], [[wx, gz + half], [wx, H]]]
        if self.z_off > 0:                                 # amend9: source chamber top / bottom walls
            zl, zh = self.z_off, self.z_off + self.H0
            segs += [[[0.0, zl], [wx, zl]], [[0.0, zh], [wx, zh]]]
        return np.array(segs)

    def free_area(self) -> float:
        """Arena area minus the hard wall footprint."""
        W, H = self.arena
        return W * H - 2.0 * self.wall_x * self.z_off - 2.0 * WALL_HALF_THICKNESS * (H - self.gate_width)

    # --- soft macro potentials ---
    def goal_distance(self, x, z):
        return jnp.sqrt((x - self.goal[0]) ** 2 + (z - self.goal[1]) ** 2)

    def potentials(self, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, z = grid.meshgrid()
        wall_profile = self.obs_penalty * jnp.exp(-((x - self.wall_x) / self.wall_thickness) ** 2)
        blocked = 0.5 * (jnp.tanh((jnp.abs(z - self.gate_z) - self.gate_width / 2.0)
                                  / self.edge_softness) + 1.0)
        v_obs = wall_profile * blocked
        if self.z_off > 0:                                 # amend9: source chamber top / bottom walls
            src = 0.5 * (jnp.tanh((self.wall_x - x) / self.edge_softness) + 1.0)
            for zw in (self.z_off, self.z_off + self.H0):
                v_obs = v_obs + self.obs_penalty * jnp.exp(-((z - zw) / self.wall_thickness) ** 2) * src
        r = self.goal_distance(x, z)
        if self.goal_shape == "flat":                           # amend7: flat bottom inside r_goal
            e = jnp.maximum(r - self.r_goal, 0.0)
            progress = self.w_goal * jnp.sqrt(e ** 2 + 0.05 ** 2)
            hold = self.w_hold * 0.5 * jnp.tanh(e / self.goal_softness)
            return progress + hold, v_obs
        ra = jnp.sqrt((x - self.attractor[0]) ** 2 + (z - self.attractor[1]) ** 2)  # amend10: = r by default
        progress = self.w_goal * jnp.sqrt(ra ** 2 + 0.05 ** 2)  # conical: non-vanishing pull
        hold = self.w_hold * 0.5 * (jnp.tanh((r - self.r_goal) / self.goal_softness) + 1.0)
        return progress + hold, v_obs

    def initial_density(self, grid: Grid) -> jnp.ndarray:
        """Smoothed indicator of the start box, total mass N (agents, so m is in agents/m^2)."""
        x, z = grid.meshgrid()
        x0, x1, z0, z1 = self.start_box
        s = self.start_softness
        ind = (0.25 * (1 + jnp.tanh((x - x0) / s)) * (1 + jnp.tanh((x1 - x) / s))
               * 0.25 * (1 + jnp.tanh((z - z0) / s)) * (1 + jnp.tanh((z1 - z) / s)))
        return ind * (self.N / (jnp.sum(ind) * grid.step ** 2))

    def initial_positions(self, key: jax.Array) -> jnp.ndarray:
        """N agents on a jittered hexagonal lattice filling the start box (min spacing well above
        D_COLLISION, so the CBF safe set holds at t=0). Explicit PRNG key."""
        x0, x1, z0, z1 = self.start_box
        area = (x1 - x0) * (z1 - z0)
        a = math.sqrt(2.0 * area / (math.sqrt(3.0) * self.N))  # hex spacing filling the box
        if self.N == 1:
            return jnp.array([[(x0 + x1) / 2, (z0 + z1) / 2]])
        while True:
            pts = []
            row = 0
            z = z0 + a / 2
            while z <= z1 - a / 4:
                off = (a / 2) * (row % 2)
                x = x0 + a / 2 + off
                while x <= x1 - a / 4:
                    pts.append((x, z))
                    x += a
                z += a * math.sqrt(3.0) / 2
                row += 1
            if len(pts) >= self.N:
                break
            a *= 0.98
        pts = jnp.asarray(np.array(pts))
        k1, k2 = jax.random.split(key)
        idx = jax.random.permutation(k1, pts.shape[0])[: self.N]
        jitter = jax.random.uniform(k2, (self.N, 2), minval=-0.1 * a, maxval=0.1 * a)
        return pts[idx] + jitter

    def goal_capacity_area(self, step: float = 0.02) -> float:
        """Area of the goal disc that is inside the arena and on the goal side of the wall
        (amend1 implementation note)."""
        W, H = self.arena
        xs = np.arange(step / 2, W, step)
        zs = np.arange(step / 2, H, step)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        inside = ((X - self.goal[0]) ** 2 + (Z - self.goal[1]) ** 2 <= self.r_goal ** 2) & \
                 (X > self.wall_x + WALL_HALF_THICKNESS)
        return float(inside.sum() * step ** 2)
