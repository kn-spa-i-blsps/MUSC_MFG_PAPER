# Amendment 1 to prereg_projectA.md — goal-region radius r_goal(N)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a), §3 (locked parameter "goal region r_goal")
and §9 items (3) and (5). The original file is NOT edited; this amendment takes precedence for the
listed items only.
**Status:** approved by the PI before any Project-A run (no outcome observed).

## Reason
With r_goal = 1.0 m and D_COLLISION = 0.30 m, at most ~40 agents fit in the goal disc (hexagonal
packing at 0.30 m: ~12.8 agents/m^2 x pi m^2; ~23 at the 0.40 m CBF margin). §9 (3) and (5) —
>=90% of N=100 within r_goal with zero collisions — were therefore geometrically infeasible.

## Change
- r_goal(N) = sqrt(N / (pi * RHO_GOAL)), RHO_GOAL = 2 / (sqrt(3) * D_STAR^2) ~= 3.21 agents/m^2
  (hexagonal packing at the desired spacing D_STAR = 0.60 m). r_goal ~= 0.27 * W: a fixed fraction
  of the arena, self-similar like the gate.
- Values: r_goal(25)=1.58, r_goal(50)=2.23, r_goal(100)=3.15, r_goal(150)=3.86, r_goal(200)=4.45 m.
- Completeness threshold unchanged: >= 90% (§9 (3), (5)).
- Smoke report additionally states, as facts: the emergent density in the goal zone (profile / peak)
  and the distribution of agent distances to the goal.

## Implementation note (consequence, not a new free parameter)
The prereg does not fix the goal position. The disc diameter 2*r_goal ~= 0.545*W exceeds the
width of the post-gate chamber (W/2), so with the legacy goal position 0.9*W ~27% of the disc lies
outside the arena (usable area ~22.7 m^2 at N=100 -> ~73 agents at RHO_GOAL < 90). The goal
centre is therefore placed at the centre of the post-gate chamber, (0.75*W, H/2), which minimizes
the disc area lost to the arena boundary and gate wall (~30 m^2 usable at N=100 -> ~97 agents at
RHO_GOAL).
