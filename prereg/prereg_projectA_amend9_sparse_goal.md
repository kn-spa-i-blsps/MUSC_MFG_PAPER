# Amendment 9 to prereg_projectA.md — sparse goal matched to the safety margin (learned arm)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the goal
density / r_goal(N) of amendment 1, the downstream-chamber geometry of amendment 4, and the CBF pair
margin (§3 d_safe). It also withdraws the goal-shape changes of amendment 7 for this run. The original
and amendments 1 (2c080dd4…), 2 (4c5015a3…), 3 (54edaa36…), 4 (0869f717…), 5 (4a9c86b9…),
6 (8e88e0dc…), 7 (a2fa24dc…), 8 (a46ee261…) are NOT edited. No factorial and no attribution ablation
are run under this amendment.
**Status:** written after the amendment-8 run and before any run with these changes (no amendment-9
outcome observed). Code, config and report script are committed together with this file.

## Motivation (facts, learned arm, mean over 5 seeds)
| run | d_safe | V_goal / hold | completeness | collision pair-steps (in disc) |
|---|---|---|---|---|
| amendment 6 | 0.40 | conical / no | 0.872 | 76 080 (66 900) |
| amendment 7 | 0.55 | flat / yes | 0.084 | 885 (0) |
| amendment 8 | 0.40 | flat / yes | 0.278 | 38 150 (4 783) |
The closed-loop estimate error on near pairs is about 0.19 m RMS, which is larger than the amendment-6
buffer d_safe − D_COLLISION = 0.10 m. With the goal packed at D_STAR = 0.60 m, a margin that covers this
error (d_safe 0.55) leaves almost no slack to the target spacing.

## Change (one coherent change: a safe goal density)
1. **Goal mechanics of amendment 6 restored:** conical V_goal (scenario.goal_shape = conical, the
   default) and no hold-on-arrival (control.hold_on_arrival = false, the default).
2. **Sparse goal:** hex-packing spacing in the goal disc s_goal = 0.80 m (scenario.goal_spacing),
   so RHO_GOAL = 2 / (sqrt(3) s_goal²) = 1.80 agents/m² (was 3.21 at D_STAR = 0.60), and
   r_goal(N) = sqrt(N / (pi RHO_GOAL)) = 4.20 m at N = 100 (was 3.15). D_STAR, R_BARRIER and
   macro_cost.m_bar (6.25) are NOT changed.
3. **Downstream chamber widened to hold the disc with a margin (scenario.goal_clearance = 1.18 m,
   the amendment-4/6 disc-to-wall clearance (8.66 − 2·3.15)/2 at N = 100):**
   - Amendment-4 placement rule unchanged: goal waypoint at x = wall_x + 3 r_goal, arena right edge at
     x_goal + r_goal + clearance. With the new r_goal: x_goal = 18.37 m, W = 23.76 m.
   - The arena height becomes max(H0, 2 r_goal + 2·clearance) = 10.76 m (H0 = 8.66 m). The source
     chamber (W0/2 × H0 = 50 m², the same density-constant reference) and the gate (width 2.31 m) are
     kept unchanged and centred at z = 5.38 m. Two horizontal hard walls at z = 1.05 m and z = 9.71 m,
     from x = 0 to the wall, close the source chamber off from the extra height. They are CBF wall
     segments (same r_wall) and soft V_obs ridges (same obs_penalty, wall_thickness, edge_softness)
     as the existing wall. The start box moves with the source chamber.
   - The disc then has 1.18 m clearance to the top, bottom and right walls.
   - Defaults (goal_spacing, goal_clearance unset) reproduce amendments 1–8 exactly (unit and parity
     tests pass).
4. **cbf.d_safe = 0.55 m** (margin 0.25 m > 0.19 m estimate error; 0.55 < 0.80 goal spacing).
5. Horizon: the amendment-5 rule is unchanged (T = HORIZON_MULT · path length / V_NOM). The farther
   and larger goal gives T = 70.4 s (amendments 6–8: 57.8 s). Phi / m* are re-solved for the new
   arena and potential (new macro cache key; same grid step, cost, FP settings).
6. The learned predictor is NOT retrained: the amendment-6 checkpoint (predictor_weights.npz,
   SHA256 in experiments/ladder_fargoal_predictor_record.json) is used as is.

## Run
experiments/ladder_fargoal_amend9.yaml, arm learned, N = 100, delayed perception as in amendment 6,
seeds 0–4 (the same seeds, paired), per-agent x_hat logged (amendment-7 logging).
Output: results/ladder_fargoal_amend9_N100/.

## Reported facts (no verdict; the PI evaluates)
Per seed and mean ± sd, compared with the amendment-6 learned arm and the amendment-8 run (paired by
seed). Each arm is evaluated against its own goal (r_goal 4.20 m for this run, 3.15 m for the others).
- completeness (final, and at t = 57.8 s, the amendment-6/8 horizon), t_first_90pct;
- agents outside r_goal at the end; mean agents inside the disc for t >= 28 s (the amendment-6–8
  window) and over the final 10 s of each run;
- collision pair-steps total and in the goal disc; depth distribution (bands 0–0.10–0.20–0.25–0.30 m);
- worst-case p5 nearest-neighbour separation, min separation;
- CBF intervention rate; QP infeasibility (slack-active rate) overall, inside and outside the disc;
- the per-pair collision attribution of amendment 7 (all, in disc, step before onset).
Viz: snapshots and animation (seed 0), and a summary panel against amendments 6 and 8.
