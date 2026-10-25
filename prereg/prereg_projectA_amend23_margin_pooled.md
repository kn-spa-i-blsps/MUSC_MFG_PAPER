# Amendment 23 — fixed-margin re-tune with the pooled predictor v3 (overnight Phase 2), precision=highest

**Base:** amend22 (pooled predictor v3, sha256 9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5; amend21
fixed geometry; per-N Φ/m\* caches). d_safe 0.70 was chosen in amend20 on the old OOD-contaminated predictor in a
tighter arena; it is re-tuned here. Frozen, pre-run, SHA before run, detached, self-push.

## Design
- learned_pooled (v3) with fixed d_safe ∈ {0.55, 0.70, 0.85, 1.00} at N ∈ {100, 150} (N=100 = representative, learned
  safe there in amend22; N=150 = where learned degrades), seeds 0–9, amend21 geometry (43.379 × 19.081 m, gate 2.309;
  min initial sep 0.788 at N=100, 0.626 at N=150 — note d_safe 0.70/0.85/1.00 start below the initial spacing at N=150
  and 0.85/1.00 at N=100). 80 rollouts, no solve.
- The d_safe 0.70 cells repeat amend22's learned cells (same config): their agreement within CI is a reproducibility
  check of the chaotic learned closed loop (not bit-identity — learned is not run-to-run bit-stable on GPU).
- Logging-only code change before this run (env/rollout.py): per-step near-pair count `n_near` → new metrics
  `closedloop_xhat_mse_near_pairw` (pair-weighted; fixes the bias found in review_amend22 Step A) and
  `near_pair_steps`. Identity gate run on the pod BEFORE the amendment runs: real seed 0 at N=25 and N=100 must
  equal amend21 on 8 metrics, else the chain stops (logs/overnight/identity_nnear/identity.json).

## Pre-registered selection rule (Phase 2 question)
Among margins with completeness mean ≥ 0.90, the one with the highest safe-success mean; it replaces 0.70 only if its
safe-success difference vs 0.70 (unpaired seed bootstrap) has a CI excluding 0; otherwise 0.70 is kept ("not resolved").
Applied per N; reported for both. Also reported: ever<0.30, p5 separation, intervention, slack, x̂ MSE (pair-weighted).
Expectation stated (not a hypothesis test): review_amend22 found that a larger margin HURTS real (stale estimates →
more evasive motion → more delay error); whether the same holds with compensation is open.

## Review
logs/overnight/review_amend23.md (brief §3–§4).
