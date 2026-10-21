"""Geometrie scenariuszy. RDZEŃ: SingleGate. Track zostaje stubem do dnia po v0-gate (§11)."""
from __future__ import annotations

import jax.numpy as jnp

from env.base import Grid


class SingleGate:
    """Jedna bramka w pionowej ścianie na środku areny. `gate_width` to POKRĘTŁO
    intensywności crowd-aversion — miękki barrier (V_obs), nie twardy constraint (§8).
    """

    def __init__(
        self,
        gate_width: float,
        arena: tuple[float, float] = (10.0, 6.0),
        *,
        wall_thickness: float = 0.3,
        obs_penalty: float = 20.0,
        edge_softness: float = 0.25,
        goal_weight: float = 1.0,
        density_scale: float = 1.0,
        source_spread: float = 0.6,
    ) -> None:
        self.gate_width = gate_width
        self.arena = arena
        self.wall_thickness = wall_thickness
        self.obs_penalty = obs_penalty
        self.edge_softness = edge_softness
        self.goal_weight = goal_weight
        self.density_scale = density_scale
        self.source_spread = source_spread

    def domain(self) -> tuple[float, float]:
        return self.arena

    def potentials(self, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
        w, h = self.arena
        x, z = grid.meshgrid()
        wall_x = w / 2.0
        gate_z = h / 2.0

        wall_profile = self.obs_penalty * jnp.exp(-((x - wall_x) / self.wall_thickness) ** 2)
        opening_half = self.gate_width / 2.0
        blocked = 0.5 * (
            jnp.tanh((jnp.abs(z - gate_z) - opening_half) / self.edge_softness) + 1.0
        )
        v_obs = wall_profile * blocked

        goal = jnp.array([0.9 * w, gate_z])
        v_goal = self.goal_weight * ((x - goal[0]) ** 2 + (z - goal[1]) ** 2)
        return v_goal, v_obs

    def initial_density(self, grid: Grid) -> jnp.ndarray:
        w, h = self.arena
        x, z = grid.meshgrid()
        source = jnp.array([0.15 * w, h / 2.0])
        m0 = jnp.exp(
            -((x - source[0]) ** 2 + (z - source[1]) ** 2) / (2.0 * self.source_spread**2)
        )
        mass = jnp.sum(m0) * grid.step**2
        return m0 * (self.density_scale / mass)


class PersistentGate(SingleGate):
    """Geometria SingleGate + NIEUSTAJĄCY popyt: cel wspólny dla całego roju naprzemiennie
    (prawy A=(0.9W,H/2) / lewy B=(0.1W,H/2)), przełączany co `period` sekund. Zamiast jednego
    tranzytu (po którym pole średnie "osiada" na celu, a nieaktualne m_hat przestaje szkodzić),
    rój PRZECHODZI przez bramkę w kółko, więc tłok przy bramce nie wygasa.

    V_goal jest zależne od czasu: kształt (n_steps+1, nx, nz). solve_hjb i rollout akceptują to
    (ndim==3), więc m_star jest liczone dokładnie dla tego samego, okresowego celu, a nie dla
    jednorazowego tranzytu — równowaga jest samo-spójna z tym, co realizują agenci.

    PRZYBLIŻENIA (świadome, patrz też docstring w experiments/tau_sweep.py):
    - Przełączanie jest SYNCHRONICZNE dla całego roju, nie per-agent po dotarciu do celu.
      Re-tasking per-agent wymagałby źródeł/upustów masy w równaniu ciągłości (mfg/transport.py)
      i wielopopulacyjnego MFG; bez tego m_star nie opisywałoby faktycznej populacji. Koszt:
      agenci w połowie tranzytu zawracają w chwili przełączenia (kołysanie, nie równomierny
      przeciwprąd).
    - Skończony horyzont z V(T)=0: w ostatnich ~7% horyzontu polityka wygasa (agenci zwalniają),
      bo brak kosztu po T. Okno kosztu sięga T, więc ten końcowy fragment jest lekko "osiadły".
    - Cel jest widziany w przód (HJB antycypuje przełączenia): przy zbyt krótkim `period`
      optymalny agent zawisa przy bramce zamiast przechodzić — period musi być rzędu
      kilku czasów tranzytu.
    """

    def __init__(
        self,
        gate_width: float,
        arena: tuple[float, float] = (10.0, 6.0),
        *,
        period: float,
        **kwargs,
    ) -> None:
        super().__init__(gate_width, arena, **kwargs)
        self.period = period

    def goal_phase(self, t: jnp.ndarray) -> jnp.ndarray:
        """0 -> cel prawy (A), 1 -> cel lewy (B)."""
        return jnp.floor(t / self.period).astype(int) % 2

    def potentials(self, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, v_obs = super().potentials(grid)
        w, h = self.arena
        x, z = grid.meshgrid()
        t = jnp.arange(grid.n_steps + 1) * grid.dt
        phase = self.goal_phase(t)
        gx = jnp.where(phase == 0, 0.9 * w, 0.1 * w)[:, None, None]
        gz = h / 2.0
        v_goal_t = self.goal_weight * ((x[None] - gx) ** 2 + (z[None] - gz) ** 2)
        return v_goal_t, v_obs


class Track:
    """[stub, po v0-gate] Tor złożony: przewężenie -> szerszy obszar -> dwie szczeliny (§11)."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Track jest post-gate (patrz CLAUDE.md §11) — nie ruszać teraz.")
