# Amendment 27 — compensation-method ladder and MFG/CBF layer attribution on the crossover geometry

**Base:** amend22 (real d0.55/0.70, learned_pooled v3 d0.70), amend24 (noise_only, fresh N150/200). PI goal
(2026-10-31): show (1) the delay-free controller works, (2) the damage of the TDMA delay, (3) which compensation method
removes the most of it (cv, Smith, NN), and (4) how much of the recovery acts through each of the two layers the
estimate x̂ feeds: the MFG nominal law (local density m̂ in the congestion term) and the CBF-QP. Frozen, pre-run, SHA
before run, detached, self-push. Phase B (σ-CBF) is NOT touched by this amendment.

## Design (amend21 geometry, per-N caches with unchanged keys, precision=highest, seeds 0–9, weights v3 sha 9583cb7c…)
- **A — method ladder:** `cv` and `smith` (existing perception conditions, unchanged) × d_safe {0.55, 0.70} × N ∈
  {25, 50, 100, 150, 200}; `fresh` d0.70 × N ∈ {25, 50, 100} (completes amend24's N150/200 to all N).
- **B — layer split (new arm, CODE CHANGE):** condition `split:<mfg>/<cbf>` computes x̂ twice from the same PRNG key
  (same m=0 observation): the first feeds nominal_control (density term, hold), the second feeds the CBF (and the x̂
  logs). Any other condition string runs the unchanged single-x̂ path. Cells at d0.70 × N ∈ {100, 150, 200}:
  `learned/real` (compensation in the MFG layer only), `real/learned` (CBF only), `noise_only/real`, `real/noise_only`
  (oracle per layer). "both" = learned_pooled d0.70 (amend22) / noise_only d0.70 (amend24); "none" = real d0.70.
- **I — identity gate (run first on the pod, aborts the chain on failure):** N=100 seed 0 d0.70, `real` and
  `split:real/real` vs the committed amend22 real_d0.70 seed 0 metrics, every IDENTITY_KEY exactly equal.
- 352 rollouts; read-only rows from amend22/amend24 report.json (per-seed values). CIs: seed-level bootstrap, 10k, 95 %.

## Questions and pre-registered readings
- **Q1 (delay-free baseline):** fresh and noise_only safe (ever<0.30 CI upper ≤ 0.10, completeness ≥ 0.90) at every N.
  A completeness shortfall with zero collisions is reported as a liveness limitation, not a safety failure
  (amend24 already showed fresh N=200 completeness 0.763).
- **Q2 (method ranking):** per N, learned d0.70 minus each non-learned compensator at ITS better margin (chosen by mean
  safe-success, so the comparator is given the advantage). "NN best at N" = safe-success difference CI excludes 0 vs
  every comparator. Also safe_N per arm (amend22 definition). No other claim of superiority.
- **Q3 (layer attribution):** per N ∈ {100,150,200} and per compensator (learned, oracle): safe-success and ever<0.30
  of MFG-only and CBF-only vs none and vs both (CIs), and the recovered fraction (split − none)/(both − none).
  Reading: a layer "carries" the compensation if its split differs from none with CI excluding 0; "both layers needed"
  if each single-layer split is below both with CI excluding 0. No prediction of which layer is pre-registered.

## Review
logs/overnight/review_amend27.md (REVIEW MODE, PLUMBING vs THESIS classification for any non-positive result).
