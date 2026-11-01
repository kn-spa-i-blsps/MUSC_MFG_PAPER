# Amendment 32 — window-matched classical baseline `cvls`; added to amend29/30

**Base:** review_amend27.md. Both classical comparators failed for identifiable reasons: cv amplifies noise
(finite difference of 2 broadcasts × age), and smith is congestion-blind. "NN beats classical compensation" therefore
needs a noise-robust classical estimator on the SAME input the NN receives. Frozen, pre-run, SHA before run, detached,
self-push. Code: condition `cvls` in compensation/perception.py (other conditions unchanged),
experiments/amend32_cvls.py, logs/overnight/amend32_chain_pod.sh.

## Estimator (no learning, no privileged information)
The 4-broadcast window of the learned arm comes from `learned_features_dense` with the same key, so the noise draws
are identical. Per block m, it gives j's position relative to i's current position at j's broadcast instant:
p_m = rel_m − ego_disp_m, at time τ_m = −age_m. An ordinary least-squares line p(τ) = a + bτ is fitted through the 4
points. x̂_ij = x_i + a, the intercept at τ = 0. This is the minimum-variance constant-velocity estimate for white
observation noise over that window, i.e. a fixed-window Kalman smoother without process noise.

**Observed before this commit** (implementation check, open-loop, real-arm history seed 100, not an eval). Near-pair
(true d < 1 m) MSE of x̂:

| condition | real | cv | cvls | learned v3 |
|---|---|---|---|---|
| N=100, t = 10 s | 1.03 | 1.55 | 0.90 | 0.48 |
| N=100, t = 60 s | 0.45 | 0.59 | 0.29 | 0.24 |
| N=200, t = 10 s | 3.32 | 5.90 | 4.25 | 1.11 |

cvls improves on cv everywhere and on real at N=100. At N=200 it is worse than real: the window spans about 3 s
there.

## Design
- cvls × d_safe {0.55, 0.70} × N ∈ {25, 50, 100, 150, 200}; amend21 geometry, per-N caches unchanged, seeds 0–9,
  precision=highest. 100 rollouts.
- learned v3, real and noise_only rows are read from the committed amend27 report (per-seed values).
- The cvls margin per N is the d with the higher mean safe-success (tie → 0.70). This is the amend27 rule, and it
  favours the comparator.

## Readings (fixed now)
- **Q1:** learned v3 d0.70 − cvls (its better margin) safe-success, per N. "NN beats the window-matched classical
  baseline at N" = CI excludes 0 with a positive sign. Combined with amend27, "NN best" at N requires this AND the
  amend27 Q2 condition.
- **Q2:** cvls − real (better margins) safe-success per N, and cvls safe_N (amend22 definition at the better margin).
- No other claim.

## Change to amend29 and amend30 (before either has run)
- Arm `cvls` is added to every amend29 setting and amend30 layout, at its amend32 margin for that N. Their reports add
  `learned_minus_cvls` and the cvls recovered fraction.
- **Order on the pod:** amend28 → **amend32** → amend29 → amend30 → amend31. amend29/30 write their cells after
  amend32's report exists.
- Every other item of the amend29/30 aneksy is unchanged.
- Expected rollouts: amend29 800 → 960; amend30 390 → 450.

## Review
logs/overnight/review_amend32.md.
