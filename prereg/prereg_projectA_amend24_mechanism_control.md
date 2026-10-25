# Amendment 24 — mechanism control for the N crossover: perfect delay compensation (noise_only) and oracle (fresh)

**Base:** amend22 + review_amend22 (fix list item 1). In amend22 learned_pooled d0.70 is safe at N=50 and nearly at
N=100 but degrades at N=150/200 (ever<0.30 0.473 / 0.827) while its open-loop estimation error stays ≤ 0.31× real's;
candidate causes: (i) density/packing (min initial sep 0.626 / 0.553 < d_safe 0.70; CBF slack 0.22–0.24), (ii) predictor
tail error. Frozen, pre-run, SHA before run, detached, self-push.

## Design
- `noise_only` (existing perception condition: TDMA age set to 0, UWB range / bearing noise kept, same PRNG use as
  real) = perfect delay compensation with realistic sensing, d_safe 0.70, N ∈ {25, 50, 100, 150, 200}, seeds 0–9.
- `fresh` (no delay, no noise — oracle) d_safe 0.70 at N ∈ {150, 200}, seeds 0–9.
- amend21 geometry + per-N caches, precision=highest; 70 rollouts; same pod; runs after amend23 in the same night chain
  (identity gate of the logging-only n_near change applies; logs/overnight/identity_nnear/identity.json).
- Compared (read-only) with amend22 rows learned_pooled_d0.70 and real_d0.70: per N differences on safe-success and
  ever<0.30, unpaired seed bootstrap CIs.

## Pre-registered reading (decision rule)
- If noise_only at N=150/200 is NOT safe (ever<0.30 CI upper > 0.10) → the residual learned failure there is not a
  staleness/predictor effect; it is density/controller (packing at/below d_safe) — the N≥150 learned rows are then
  labelled "density-limited", and the crossover headline is read on N ≤ 100.
- If noise_only is safe at N=150/200 while learned is not (difference CI excludes 0) → the residual is predictor error
  (tail), and better compensation would extend the safe N.
- fresh vs noise_only at N=150/200 separates sensor noise from delay.

## Review
logs/overnight/review_amend24.md.
