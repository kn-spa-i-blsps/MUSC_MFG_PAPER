# Amendment 5 to prereg_projectA.md — horizon from the actual path length to the goal waypoint

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a), §3 horizon
(HORIZON_MULT * W / V_NOM), for the far-goal geometry of amendment 4 only. The original and amendments
1 (2c080dd4…), 2 (4c5015a3…), 3 (54edaa36…) and 4 (0869f717…) are NOT edited. No official smoke (§9) or
factorial run is made under this amendment.
**Status:** written before any run with the new horizon (no outcome observed).

## Reason
With the goal waypoint 3 r_goal behind the gate (amend4) the path is ~2x longer, while the horizon was
still HORIZON_MULT * W0 / V_NOM = 34.6 s (N=100). In results/diag_fargoal_N100_fresh the arrived fraction
was still rising at T (0.65 at 25 s, 0.76 at 30 s, 0.83 at T): the reported completeness is truncated
by time, not a ceiling.

## Change (single change: the horizon)
- T = HORIZON_MULT * L / V_NOM, HORIZON_MULT = 3.0 and V_NOM = 1.0 m/s unchanged, where L is the
  actual path length to the goal: farthest start-box corner -> gate centre -> farthest point of the
  goal disc (waypoint + r_goal along the axis), i.e. every agent's longest route to any point of the
  disc. N=100: L = 6.657 + 9.451 + 3.150 = 19.26 m -> T = 57.8 s (was 34.6 s).
- The MFG solve (Phi, m*) uses the same T (solver dt and grid step unchanged; re-solved, cached).
- Everything else unchanged: geometry (amend4), cost weights, C_DENS = 1, CBF exact_hard, perception.

## Saturation criterion (pre-committed, to read a ceiling rather than a truncated value)
The arrived fraction is "saturated" if it changes by less than 0.02 between T - 10 s and T. If this is
not met the run is reported as NOT saturated (the value is a lower bound); no ad-hoc extension inside
this amendment.

## Test (diagnostic)
Single-gate far goal, N = 100, fresh perception, CBF on, C_DENS = 1, NU = 0, seed 0. Facts: completeness
(final + curve; saturation criterion), whether the goal disc fills to the centre or stays a crescent /
ring (radial profile inside r_goal over time), agents downstream outside r_goal at T, churn
(enter-then-leave), min separation, worst-case p5, CBF intervention rate, QP infeasibility.
