# Amendment 12 to prereg_projectA.md — safe-success re-analysis of the amendment-11 ladder logs

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the
per-agent task metric reported for the amendment-11 ladder. The original and amendments 1–11 are NOT edited
(amendment 11: 39188e89…). **No new runs**: no rollout, no datagen, no macro solve, no sweep. Only the existing logs
results/ladder_fargoal_amend11_N100/{real,cv,learned,smith}/logs_seed{0..4}.npz are read.
**Status:** written after the amendment-11 run and before this per-agent analysis was computed. Script committed
together with this file.

## Motivation
Completeness (fraction of agents within r_goal at the end) counts an agent that reached the disc while colliding.
A drone that touched another drone at a true distance below D_COLLISION has crashed; its arrival is not a success.

## Definitions (per agent i, per arm, per seed; TRUE positions)
- Rollout states: logs `pos` (T states, state at step start) plus `pos_final` (T+1 states in total), as in
  env.rollout.summarize.
- **arrived_i**: |pos_final_i − goal| ≤ r_goal (identical to completeness_final).
- **collided_i**: there exists a state t and an agent j ≠ i with |pos_t,i − pos_t,j| < D_COLLISION = 0.30 m.
- **safe success** = fraction of agents with arrived ∧ ¬collided.
- Breakdown (fractions of N, sum to 1): arrived ∧ safe / arrived ∧ collided / not arrived ∧ safe /
  not arrived ∧ collided.
- **ever-collided** = fraction of distinct agents with collided_i (not pair-steps).
- **old completeness** = fraction with arrived_i, regardless of collisions.
- Consistency check: the pair-step count d < 0.30 over the T `pos` states must equal `collision_pair_steps`
  in metrics_seed*.json.

## Reported facts (experiments/amend12_report.py; no verdict)
Per arm: mean ± sd over seeds 0–4 (ddof = 1) and per seed of every quantity above; the table
arm × (safe success, % agents ever collided, old completeness). Output:
results/ladder_fargoal_amend11_N100/amend12_safe_success.json.
