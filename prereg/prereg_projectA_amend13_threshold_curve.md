# Amendment 13 to prereg_projectA.md — safe success vs crash threshold (re-analysis of the amendment-11 logs)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the
crash threshold in the amendment-12 per-agent safe-success metric. The original and amendments 1–12 are NOT edited
(amendment 12: 426d4c3a…). **No new runs**: no rollout, no datagen, no macro solve, no sweep. Only the existing logs
results/ladder_fargoal_amend11_N100/{real,cv,learned,smith}/logs_seed{0..4}.npz are read. D_COLLISION (0.30 m) in
the code and in all other metrics is unchanged.
**Status:** written after the amendment-12 analysis and before this threshold analysis was computed. Script
committed together with this file.

## Definitions (per agent i, per arm, per seed; TRUE positions; rollout = `pos` + `pos_final`, as amendment 12)
- **arrived_i**: |pos_final_i − goal| ≤ r_goal (= completeness_final).
- **minsep_i** = min over all states t and all agents j ≠ i of |pos_t,i − pos_t,j|.
- For a crash threshold c: **collided_i(c)** = minsep_i < c; **safe success(c)** = fraction of agents with
  arrived ∧ ¬collided(c); **ever-collided(c)** = fraction of agents with collided(c).
- Thresholds c ∈ {0.30, 0.25, 0.20, 0.15, 0.10} m. At c = 0.30 the values must equal amendment 12 (check).

## Reported facts (experiments/amend13_report.py; no verdict)
- Per arm: mean ± sd over seeds 0–4 (ddof = 1) and per seed of safe success(c) and ever-collided(c) for every c;
  tables arm × threshold.
- For each arm, the largest grid threshold at which mean safe success ≥ 0.5 and ≥ 0.9 (none if never).
- Supplementary (continuous, same data): the largest c (from the per-agent minsep values) at which mean safe success
  over seeds ≥ 0.5 and ≥ 0.9, and the ceiling, i.e. old completeness (safe success as c → 0).
Output: results/ladder_fargoal_amend11_N100/amend13_threshold_curve.json (includes per-agent minsep and arrived).
