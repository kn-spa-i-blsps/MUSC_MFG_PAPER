# Amendment 4 to prereg_projectA.md — goal far downstream of the gate (downstream chamber)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a), §5/§9 single-gate
geometry, and the goal placement of amendment 1 (2c080dd4…, implementation note "goal centre at
(0.75 W, H/2)"). The original and amendments 1 (2c080dd4…), 2 (4c5015a3…) and 3 (54edaa36…) are NOT
edited. No official smoke (§9) or factorial run is made under this amendment.
**Status:** written before any far-goal run (no outcome of this geometry observed).

## Reason
Clogging at the gate exit: with the goal disc centred at (0.75 W, H/2) its near edge lies ~0.3 m
behind the gate plane (2.89 m chamber half-width minus r_goal = 3.15 m -> the disc reaches the wall),
so agents that have just passed the gate stop in the throat and block it. Evidence: snapshots of
results/diag_cdens0_N100_fresh (and diag_singlegate_N100_fresh) show a dense column jammed through and
directly behind the gate while the far part of the post-gate chamber stays empty.

## Change (single geometric change)
- Single-gate gets a downstream chamber. Source chamber and gate are unchanged: wall at
  x = W0/2, gate width 0.2 W0, height H, start box as before, with (W0, H) = arena_for_N(N), i.e. the
  source side stays density-constant (N / (W0 H) = RHO_NOM; source chamber = W0/2 x H).
- Goal WAYPOINT centre at x = wall_x + 3 r_goal(N), z = H/2 (the disc's near edge is 2 r_goal behind
  the gate plane); r_goal(N) unchanged (amend1). The arena is extended to the right to
  W = x_goal + r_goal + m with m = (H - 2 r_goal)/2, so the disc lies in open space (equal clearance to
  the top, bottom and right walls). N=100: x_goal = 15.22 m, W = 19.55 m, H = 8.66 m.
- The goal is a settable waypoint (scenario attribute, static in Project A, dynamic in B), consumed by
  the existing inputs: V_goal (macro cost), the agent's local goal observation and the CBF goal-biased
  priority. Arrival is still |x - goal| <= r_goal.
- Unchanged: horizon (horizon_for_N, from W0), cost weights, Phi/m* solver settings (re-solved and
  cached for the new arena), controller (C_DENS = 1), CBF (exact_hard), perception arms.

## Test (diagnostic)
Single-gate N = 100, fresh perception, CBF on, C_DENS = 1, NU = 0, seed 0. Reported as facts:
completeness (final and over time), whether the gate clog dissolves (agents near the gate over time),
agents stuck before / after the gate at T, min separation, worst-case p5, CBF intervention rate, QP
infeasibility; comparison to diag_cdens0_N100_fresh/cdens1 (near goal, same controller and solver).
