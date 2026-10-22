# Amendment 7 to prereg_projectA.md — robust safety margin and flat-bottomed goal (learned arm)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the CBF pair
margin (§3 d_safe) and the goal potential / nominal control inside the goal disc (§2, §4), on the
ladder setup of amendment 6. The original and amendments 1 (2c080dd4…), 2 (4c5015a3…), 3 (54edaa36…),
4 (0869f717…), 5 (4a9c86b9…), 6 (8e88e0dc…) are NOT edited. No factorial and no attribution ablation
are run under this amendment.
**Status:** written before any run with these changes (no outcome observed). Code and config are
committed together with this file.

## Motivation (facts from the amendment-6 run, results/ladder_fargoal_N100, learned arm)
- 76 082 collision pair-steps (mean over 5 seeds), 88 % with both agents inside the goal disc,
  73 % of those after the disc has filled (t >= 28 s), at a constant per-pair rate — persistent,
  not a filling transient.
- 85 % of in-disc collision pair-steps occur with a FEASIBLE QP at both agents (slack inactive).
- Closed-loop estimation error of x_hat on near pairs ~0.19 m RMS (mean sq. 0.036 m²), larger than
  the barrier buffer d_safe − D_COLLISION = 0.10 m.
- The nominal control has a positive component towards the collision partner in 72 % of colliding
  agent-steps (conical V_goal keeps pulling every agent to the disc centre).
Per-pair attribution (perceived vs true distance) could not be computed: x_hat was not logged.

## Changes (all config-gated; defaults reproduce amendments 1–6 exactly)
1. **Fix A — robust margin:** cbf.d_safe = 0.55 m (was 0.40). D_COLLISION = 0.30, RHO_GOAL = 3.21
   (D_STAR = 0.60) unchanged; 0.55 < 0.60 so the target packing remains admissible for the barrier.
   macro_cost.m_bar (= 1/R_BARRIER² = 6.25) is NOT changed.
2. **Fix B — goal shape:**
   - V_goal flat-bottomed (scenario.goal_shape = flat): with e = relu(r − r_goal),
     V_goal = w_goal·sqrt(e² + 0.05²) + w_hold·0.5·tanh(e / goal_softness).
     grad V_goal = 0 inside r_goal; outside it equals the conical gradient up to the 0.05 m cone
     rounding (the two potentials differ by a constant there). Weights w_goal = 2, w_hold = 8,
     goal_softness = 0.25 unchanged. Phi / m* are re-solved offline with this potential (new macro
     cache key; same grid, horizon, cost, FP settings).
   - Hold-on-arrival (control.hold_on_arrival): u_nom_i = 0 when |x_i − goal| <= r_goal (own true
     position) AND min_j |x_i − x_hat_ij| >= d_safe (nearest ESTIMATED neighbour at a safe distance);
     otherwise u_nom_i is unchanged (routing + density term). The CBF filter is unchanged.
3. **Logging (log_xhat):** per step, x_hat_ij − x_i for every ordered pair (float16), the CBF's
   k_nb neighbour indices per agent, and the hold flag. Does not change the dynamics.

The learned predictor is NOT retrained: the amendment-6 checkpoint (predictor_weights.npz, SHA256 in
experiments/ladder_fargoal_predictor_record.json) is used as is, although the closed-loop
trajectories now come from a different controller.

## Run
experiments/ladder_fargoal_amend7.yaml, arm learned, N = 100, far goal (amend4), horizon 57.8 s
(amend5), delayed perception as in amendment 6, seeds 0–4 (same as the amendment-6 ladder, paired).
Output: results/ladder_fargoal_amend7_N100/.

## Reported facts (no verdict; the PI evaluates)
Per seed and mean ± sd, compared with the amendment-6 learned arm (paired by seed):
- collision pair-steps total and in the goal disc (both agents within r_goal); depth distribution
  (true d of colliding pairs), min separation;
- completeness (final), arrived-fraction curve, t_first_90pct;
- QP infeasibility (slack-active rate) overall and inside the disc;
- worst-case p5 nearest-neighbour separation;
- direct per-pair attribution of every collision pair-step (true d < D_COLLISION), using the
  perceived distances d̂_ij = |x_hat_ij − x_i| and d̂_ji of the two agents at that step:
  (i) ESTIMATE ERROR: min(d̂_ij, d̂_ji) >= d_safe (neither agent perceived a barrier violation);
  (ii) perceived inside the barrier with the QP feasible at both agents;
  (iii) QP infeasible (slack active) at at least one agent.
  Also: share of collisions where the partner was not among the agent's k_nb CBF neighbours; the
  same attribution for the step before collision onset; distribution of d̂ − d.
- Hold flag rate inside the disc. Viz: snapshots and animation (seed 0).
