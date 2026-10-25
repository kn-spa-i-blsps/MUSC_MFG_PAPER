# REVIEW — amend24 (mechanism control: noise_only / fresh)

Run: aneks sha256 15da76975a29f15d118469084f7877d72c28c4aaa278bfdf8386d1502932faca (prereg commit ea0a5c5, before the
run); result commit eacfb15 (DONE, 70/70); amend21 geometry; d_safe 0.70; precision=highest. Comparison rows: amend22
(learned_pooled v3 d0.70, real d0.70), read-only.

## Data (mean [95% CI], 10 seeds)

| N | arm | safe-success | ever<0.30 | completeness | p5 sep | interv. | slack | x̂ MSE near (pair-w.) |
|---|---|---|---|---|---|---|---|---|
| 25 | noise_only | 1.000 | 0.000 | 1.000 | 0.657 | 0.564 | 0.047 | 0.011 |
| 50 | noise_only | 1.000 | 0.000 | 1.000 | 0.655 | 0.659 | 0.095 | 0.011 |
| 100 | noise_only | 1.000 | 0.000 | 1.000 | 0.641 | 0.841 | 0.209 | 0.011 |
| 150 | noise_only | 1.000 | 0.000 | 1.000 | 0.645 | 0.912 | 0.264 | 0.011 |
| 150 | fresh | 0.998 [0.995,1.000] | 0.000 | 0.998 | 0.680 | 0.957 | 0.000 | 0.000 |
| 200 | noise_only | 0.968 [0.963,0.973] | 0.000 | 0.968 | 0.590 | 0.947 | 0.321 | 0.011 |
| 200 | fresh | 0.763 [0.658,0.822] | 0.000 | 0.763 [0.658,0.822] | 0.590 | 0.971 | 0.002 | 0.000 |

For reference, amend22 learned v3 d0.70 safe-success / ever<0.30:

| N | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| safe-success | 1.000 | 0.992 | 0.880 | 0.479 | 0.137 |
| ever<0.30 | 0.000 | 0.008 | 0.076 | 0.473 | 0.827 |

noise_only − learned safe-success:

| N | difference [95% CI] |
|---|---|
| 50 | +0.008 [0.000, 0.020] |
| 100 | +0.120 [0.092, 0.153] |
| 150 | +0.521 [0.488, 0.555] |
| 200 | +0.831 [0.812, 0.849] |

## Step A — mechanism
- **What noise_only isolates:** it uses the same controller, margin, geometry, packing and UWB/vision noise as the
  learned arm, with only the TDMA delay removed. It is collision-free at every N, including N=150/200, where the swarm
  starts below d_safe (min initial sep 0.626 / 0.553). The CBF works hard there (intervention 0.91–0.95, slack
  0.26–0.32) but keeps every pair ≥ 0.30 m.
- **Density hypothesis (i) refuted:** packing and controller limits do not cause collisions at these densities.
- **Prediction hypothesis (ii) confirmed:** the residual learned failure at N ≥ 100 is the delay-compensation error
  that v3 leaves. This is the pre-registered branch: "noise_only safe at N=150/200 while learned is not
  (difference CI excludes 0) → the residual is predictor error; better compensation would extend the safe N".
- **How a small mean error causes collisions:**
  - Learned's pair-weighted x̂ MSE (amend23: 0.023 at N=100, 0.031 at N=150) is only 2–3× noise_only's 0.011, yet its
    collision rate is far higher.
  - Interpretation, a hypothesis consistent with the data: zero-mean independent sensor noise is absorbed by the 0.70
    margin. The residual delay error is correlated in time and along the direction of motion, and has heavier tails,
    so a few large misses per pair produce contacts that the mean does not show.
  - Testable next: tail (p99) of the near-pair open-loop error, learned vs noise_only.
- **Liveness side-observation:** fresh (no noise) at N=200 has lower completeness than noise_only (0.763 [0.658, 0.822]
  vs 0.968), with zero collisions in both. Plausible mechanism, unverified: symmetric CBF deadlocks among densely
  packed agents that sensor noise breaks. Safety is unaffected; the claims below concern safety, so this is a note,
  not a finding.

## Step B — artifact scan (§3)
1. **Identity:** PASS. noise_only and fresh are existing perception conditions (no code change since the identity
   gate). noise_only differs from real only by age = 0 (compensation/perception.perceive).
2. **Precision:** PASS. All 70 rows are "highest" (chain log).
3. **OOD:** n/a. There is no predictor in these arms.
4. **Wrong knob:** PASS. The compared variable (delay present vs removed) changes ever<0.30 from 0.47–0.83 to 0.
5. **Margin starvation:** n/a. The margin is constant 0.70.
6. **Geometry/density:** PASS. Identical geometry and density to the amend22 rows they are compared with, so this is
   a within-condition contrast.
7. **Statistics:** PASS. Differences vs learned have CIs excluding 0 at N = 100, 150, 200. At N=50 the difference is
   not resolved (+0.008 [0, 0.020]).
8. **Completeness-as-sham:** PASS for noise_only (completeness ≥ 0.968, so it is safe and arrives). Fresh at N=200 has
   completeness 0.763, so its safe-success is liveness-limited; it is noted, not used for a claim.

## Step C — consistency
- **Corrects review_amend22 Step A:** the hypothesis that learned's N ≥ 150 degradation was density/packing is
  falsified.
- **Corrects §0 "no hard collision-free guarantee under large staleness" in scope:** with perfect delay compensation,
  collisions vanish in this geometry up to N=200 at d_safe 0.70. The obstacle is purely the estimation error.
- **Consistent with amend23:** a larger margin buys safety for learned only through freezing. That is what one expects
  if the residual is an estimation error that the margin must cover.

## Step D — statistics
All claims rest on differences with separated CIs. Ten seeds suffice: the effect sizes are 0.12–0.83 against CI
widths of ≈ 0.04–0.07.

## Step E — verdict + fix list
- **CONFIRMED (headline mechanism):** collisions in the N crossover are caused by TDMA-delay estimation error.
  - Removing the delay (noise_only) gives ever<0.30 = 0 at every N ∈ {25..200}.
  - Real collides from N=50.
  - Learned v3 closes the gap fully at N ≤ 50, by 88% at N=100 (safe-success 0.880 vs 1.000), and only partly at
    N ≥ 150.
- **The ceiling (perfect compensation) is far above the current predictor.** Headroom for better compensation at
  N=100–200 is +0.12 / +0.52 / +0.83 safe-success.
- **Note (not a claim):** fresh < noise_only completeness at N=200 (possible noise-broken deadlocks).
- **FIX LIST:**
  1. **(top) Improve the predictor where it matters:**
     - Train on closed-loop LEARNED trajectories (DAgger-style; the eval distribution) plus real.
     - Weight near pairs, and add capacity.
     - Gate on open-loop near-pair MSE AND its tail (p99) at every N.
     - Then a pre-registered crossover with the improved predictor.
  2. **Mechanism panel:** open-loop near-pair error tail (p99, max) for real / learned / noise_only per N.
