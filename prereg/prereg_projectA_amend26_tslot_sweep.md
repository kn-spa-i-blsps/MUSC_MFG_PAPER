# Amendment 26 — pure-staleness sweep via TDMA slot length at fixed N and geometry (overnight Phase 4), precision=highest

**Base:** amend22/23 (pooled predictor v3, sha256 9583cb7c…; d_safe 0.70). The N crossover (amend21/22) moves
staleness and density together; this sweep moves ONLY staleness. Frozen, pre-run, SHA before run, detached, self-push.

## Design
- N = 100, amend21 geometry with gate 2.309 (identical to the amend22 N=100 cells), the SAME Φ/m\* cache (key
  14d351fc8e; t_slot is perception-only and does not enter the macro solve).
- t_slot ∈ {0.0025, 0.005 (= amend21–23), 0.0075, 0.010} s → T_frame = N·t_slot = 0.25 / 0.5 / 0.75 / 1.0 s (the same
  T_frame values N = 50 / 100 / 150 / 200 had in amend22, now at fixed density). Implemented by the optional cell key
  `t_slot` (experiments/ladder.py, commit 326464e: sets control.params_legacy.T_SLOT and compensation.perception.T_SLOT
  before tracing; logged per run as metrics.t_slot; absent = 0.005).
- Arms: real d_safe 0.55, real d_safe 0.70, learned_pooled v3 d_safe 0.70. Seeds 0–9. 120 rollouts.
- Open-loop estimator check per t_slot on that t_slot's real_d0.55 histories (as amend25); learned/real > 1 → OOD flag,
  learned rows quarantined. v3 was trained with t_slot 0.005 only, but over T_frame 0.125–1.0 s, so the age features
  of every t_slot here lie inside its training range; the slot-phase lattice (j·t_slot) differs — the check decides.
- Identity: t_slot 0.005 real_d0.55 rows vs amend21 N=100 real (7 metrics, 10 seeds; the explicit key t_slot = 0.005
  must change nothing).

## Pre-registered questions
- **Q1 (staleness alone kills real):** real safe-success / ever<0.30 vs T_frame at fixed density.
- **Q2 (compensation vs staleness):** learned 0.70 − real 0.70 per t_slot with bootstrap CI.
- **Q3 (decomposition):** compare each t_slot row with the amend22 row of equal T_frame (N = 50/150/200): if the fixed-
  density learned row is safe where the amend22 row (higher density) is not, the amend22 degradation at N ≥ 150 is a
  density effect, not staleness (cross-check with amend24). Reported descriptively (different densities).

## Review
logs/overnight/review_amend26.md.
