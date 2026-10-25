# Amendment 25 — operating-density sweep via gate width at N=100 (overnight Phase 3), precision=highest

**Base:** amend22 (pooled predictor v3, sha256 9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5),
amend23 (operating margin d_safe 0.70 selected at N=100 by the pre-registered rule; review_amend23.md). Frozen, pre-run,
SHA before run, detached, self-push.

## Design
- N = 100, amend21 arena (geometry_N 230: 43.379 × 19.081 m, wall at x = 8.756, goal disc r 8.361, start box
  [0.4, 8.156] × [3.374, 15.708], min initial sep 0.788); ONE geometric axis varied: gate width
  {1.5, 2.309 (= amend21–23), 3.5, 5.0, 7.0} m (gate flux N/gate = 66.7 / 43.3 / 28.6 / 20.0 / 14.3 agents per m;
  free-area density 0.129 agents/m² for all). One Φ/m\* solve per gate (the gate enters the cache key; 2.309 = the
  existing amend21 N=100 key), on the same pod.
- Arms: real d_safe 0.55 (baseline), real d_safe 0.70 (margin-matched control), learned_pooled v3 d_safe 0.70.
  Seeds 0–9. 150 rollouts. Metrics as amend22 + pair-weighted x̂ MSE.
- Open-loop estimator check per gate (OOD, brief §3.3): perceive() for real and learned v3 on that gate's real_d0.55
  closed-loop histories (seeds 0–9, 80 sampled steps each, k ≥ 400; used for estimation only), x̂ MSE on pairs < 1 m.
  Rule: any gate with learned/real ratio > 1 is flagged OOD and its learned rows quarantined.
- Identity: gate 2.309 real_d0.55 rows vs amend21 N=100 real (same config) on 7 metrics, all 10 seeds.

## Pre-registered questions
- **Q1 (density dependence):** safe-success and ever<0.30 vs gate width per arm.
- **Q2 (advantage holds across densities):** learned 0.70 − real 0.70 safe-success per gate, bootstrap CI; "holds" at
  a gate iff the CI excludes 0 with learned better.
- **Q3 (operating density):** the narrowest gate at which learned 0.70 is safe (ever<0.30 CI upper ≤ 0.10 and
  completeness ≥ 0.90) — reported as the honest operating density for the other sweeps.

## Review
logs/overnight/review_amend25.md.
