# Amendment 8 to prereg_projectA.md — d_safe revert ablation of amendment 7 (learned arm)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the CBF pair
margin (§3 d_safe) only, on the setup of amendment 7. The original and amendments 1 (2c080dd4…),
2 (4c5015a3…), 3 (54edaa36…), 4 (0869f717…), 5 (4a9c86b9…), 6 (8e88e0dc…), 7 (a2fa24dc…) are NOT
edited. This is a single ablation. No factorial and no attribution ablation are run under this
amendment.
**Status:** written after the amendment-7 run and before any run with this change (no amendment-8
outcome observed). The config and report script are committed together with this file.

## Motivation (facts from the amendment-7 run, results/ladder_fargoal_amend7_N100, learned arm)
Amendment 7 bundled three changes: d_safe 0.40 → 0.55, flat-bottomed V_goal, and hold-on-arrival.
Mean over 5 seeds, against the amendment-6 learned arm:
- completeness_final 0.084 (amendment 6: 0.872); 91.6 / 100 agents outside r_goal at the end (12.8);
- collision pair-steps 885 (76 080); 0 in the goal disc (66 900).
This ablation separates the margin change from the goal-shape changes.

## Change
1. **cbf.d_safe = 0.40 m** (the value of amendments 1–6). Everything else is as in amendment 7:
   scenario.goal_shape = flat, control.hold_on_arrival = true (whose neighbour test now uses
   d_safe = 0.40), log_xhat = true, and the same weights, cost, grid, horizon and FP settings.
   No code change.
2. Phi / m* do not depend on d_safe (the macro cache key covers the scenario, cost, FP settings and
   grid only), so the amendment-7 cache macro_N100_cw1_4182f7b068.npz is reused as is.
3. The learned predictor is NOT retrained: the amendment-6 checkpoint (predictor_weights.npz,
   SHA256 in experiments/ladder_fargoal_predictor_record.json) is used.

## Run
experiments/ladder_fargoal_amend8.yaml, arm learned, N = 100, far goal (amendment 4), horizon
57.8 s (amendment 5), delayed perception as in amendment 6, seeds 0–4 (the same seeds, paired), with
per-agent x_hat logged. Output: results/ladder_fargoal_amend8_N100/.

## Reported facts (no verdict; the PI evaluates)
Per seed and mean ± sd, compared with the amendment-6 learned arm and the amendment-7 run (paired by
seed):
- completeness (final), t_first_90pct;
- agents outside r_goal at the end; mean number of agents inside the disc in the steady state
  (t >= 28 s, the same window as in the amendment-6/7 analysis);
- collision pair-steps total and in the goal disc (both agents within r_goal); depth distribution
  (true d of colliding pairs in bands 0–0.10–0.20–0.25–0.30 m) and the d quantiles;
- worst-case p5 nearest-neighbour separation, min separation;
- CBF intervention rate;
- QP infeasibility (slack-active rate) overall, inside and outside the disc;
- for runs with x_hat logged: the per-pair collision attribution defined in amendment 7, and the
  hold flag rate in the disc.
Viz: snapshots and animation (seed 0), and a summary panel against amendments 6 and 7.
