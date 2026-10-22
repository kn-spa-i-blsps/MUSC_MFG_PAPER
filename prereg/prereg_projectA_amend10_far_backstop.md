# Amendment 10 to prereg_projectA.md — far backstop: V_goal attractor at the far edge of the goal disc (learned arm)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the
V_goal progress term (§3), on top of the amendment-9 setup. The original and amendments 1 (2c080dd4…), 2 (4c5015a3…),
3 (54edaa36…), 4 (0869f717…), 5 (4a9c86b9…), 6 (8e88e0dc…), 7 (a2fa24dc…), 8 (a46ee261…), 9 (d8eab51f…) are NOT
edited. No factorial and no attribution ablation are run under this amendment.
**Status:** written after the amendment-9 run and before any run with this change (no amendment-10 outcome
observed). Code, config and report script are committed together with this file.

## Motivation (facts, amendment 9, learned arm, mean over 5 seeds)
Completeness 0.678 (amendment 6: 0.872); collision pair-steps 1 152, 43 of them in the disc (amendment 6:
76 082 / 66 904); worst-step p5 separation 0.234 m. The agents formed a crescent on the inlet side of the disc and
left the centre and the far half empty. Seed-0 fill occupancy at the end (agents / uniform share of the region):
near third 1.23, middle third 0.62, far third 0.21, inner disc (r ≤ r_goal/2) 0.20; 31 of 32 agents outside the disc
are on the upstream side. The V_goal cone pulls every agent to the disc centre, so the first arrivals stop on the
inlet side, and later agents queue behind them outside the disc.

## Change (one change: where V_goal pulls)
1. The cone (progress) term of V_goal, w_goal·sqrt(r_a² + 0.05²), now measures r_a from an **attractor point at the
   far (downstream, +x) edge of the goal disc**: (x_goal + r_goal, z_goal) = (22.57, 5.38) m at N = 100
   (scenario.goal_attractor_shift = 1.0, in units of r_goal). This is 1.18 m (the amendment-9 clearance) from the
   right arena wall. The attractor is placed on the disc edge, not against the far wall: an attractor at the wall
   would hold agents 1.18 m outside the disc, where the tanh hold step (width 0.25 m) has no gradient.
2. **Unchanged** (as amendment 9): the success region (disc of r_goal = 4.20 m around (18.37, 5.38)), the tanh hold
   step w_hold at r_goal around the disc centre, the local goal observation and the CBF goal-biased priority (both
   still use the disc centre), RHO_GOAL 1.80 (spacing 0.80 m), the arena and walls, d_safe 0.55, conical V_goal
   shape, no hold-on-arrival, the cost, FP settings and grid step.
3. Phi / m* are re-solved for the new potential (new macro cache key; scenario config enters the key).
4. Horizon: the amendment-5 rule gives the same T = 70.4 s (path length = gate → disc centre + r_goal, which is the
   gate → attractor distance since the gate, the disc centre and the attractor are on one horizontal line).
5. The learned predictor is NOT retrained (amendment-6 checkpoint, SHA256 in
   experiments/ladder_fargoal_predictor_record.json). Its inputs are neighbour histories only (no goal input).
6. Defaults (goal_attractor_shift unset) reproduce amendments 1–9 exactly; unit and parity tests pass (27/27).

## Run
experiments/ladder_fargoal_amend10.yaml, arm learned, N = 100, delayed perception as in amendment 6, seeds 0–4
(paired with amendments 6 and 9), per-agent x_hat logged. Output: results/ladder_fargoal_amend10_N100/.

## Reported facts (no verdict; the PI evaluates)
experiments/amend10_report.py, per seed and mean ± sd, paired differences against amendment 9 and amendment 6.
Each arm is evaluated against its own disc (amendment 6: r_goal 3.15 m).
- completeness final, at t = 57.8 s, t_first_90pct, and the completeness curve (1 s sampling);
- agents outside r_goal at the end (upstream / downstream of the disc centre); agents in the disc (t ≥ 28 s mean,
  final-10 s mean);
- fill order: occupancy (agents / RHO_GOAL·area) of the near, middle and far third of the disc (strips of width
  2 r_goal/3 along x) and of the inner disc r ≤ r_goal/2, final and final-10 s mean, and curves; x-offset of the
  in-disc centroid from the disc centre;
- collision pair-steps total and in the disc, depth bands 0–0.10–0.20–0.25–0.30 m;
- worst-step p5 nearest-neighbour separation, min separation;
- CBF intervention rate; QP infeasibility (slack-active rate) overall, in the disc and outside it;
- the per-pair collision attribution of amendment 7.
Viz: fill-order figure (completeness curves, region occupancy curves, seed-0 snapshots amendment 9 vs 10),
snapshots and animation (seed 0), summary panel against amendments 9 and 6.
