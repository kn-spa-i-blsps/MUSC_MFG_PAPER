# Amendment 6 to prereg_projectA.md — compensation ladder on delayed perception (far-goal geometry)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the compensation
ladder run (§6/§7) and the NN training data (§10: full-horizon), on the geometry of amendments 4–5.
The original and amendments 1 (2c080dd4…), 2 (4c5015a3…), 3 (54edaa36…), 4 (0869f717…), 5 (4a9c86b9…)
are NOT edited. This is a reduced-seed ladder run (5 seeds); no factorial is run under this amendment.
**Status:** written before the data generation, the training and any ladder run (no outcome observed).
**Decisions in force (PI):** the safety guarantee is ROBUST and CONDITIONAL on compensation; direction
B = amortized MFG.

## Central claim
On delayed perception (TDMA, nominal T_SLOT = 5 ms -> T_frame = 0.5 s at N = 100, plus SIGMA_R /
SIGMA_THETA), compensation (learned and/or smith) moves the safety–liveness operating point towards the
clean ceiling. Reference points:
- clean ceiling (fresh perception, same geometry/controller, diag_fargoal_longT_N100_fresh):
  0 collisions, completeness 0.97;
- stale floor: the `real` arm of THIS run (same geometry, seeds, controller). For orientation only:
  ~36k collision pair-steps on the old near-goal geometry.

## Go / no-go (evaluated by the PI against the facts; the report states no verdict)
Statistics: paired by seed (same seed = same initial positions and noise stream key), 5 seeds,
bootstrap over seeds (10 000 resamples) of the mean paired difference arm − real; "significant" =
95 % CI excludes 0.
- **GO** (claim stands, green for the full factorial): learned and/or smith has significantly fewer
  collision pair-steps than real at equal-or-better completeness (mean paired completeness difference
  >= −0.03), OR significantly higher completeness at equal-or-better safety (collision pair-steps not
  significantly higher than real). I.e. a real shift of the frontier, not noise.
- **NO-GO** (STOP, no factorial, report): compensation barely beats real — best compensating arm
  reduces mean collision pair-steps by < 50 % vs real — OR the margin stays catastrophic: worst-case
  p5 nearest-neighbour separation (min over time, mean over seeds) < 0.15 m (= D_COLLISION / 2) for
  every compensating arm.
- The 50 % and 0.15 m operationalisations of "barely" / "catastrophic" are fixed here, before the run.

## Sub-test: figure of merit (claim 2)
For every arm compute (a) the open-loop MSE of its neighbour-position estimate and (b) its closed-loop
CBF safety. Claim (2) stands if the arm with the lowest open-loop MSE is NOT the safest arm (safest =
fewest mean collision pair-steps; ties broken by higher worst-case p5).
- Open-loop MSE definition: on the held-out `real`-arm trajectories of the datagen TEST seeds, at 400
  uniformly sampled control steps (full horizon, warm buffer k >= 4 T_frame), each arm's estimate
  x_hat[i, j] is computed with the SAME perceive() code used in closed loop, and
  MSE = mean |x_hat[i,j] − x_j|^2 over ordered pairs with true distance < 1.0 m (safety-relevant
  pairs); the all-pairs MSE is reported alongside. fresh = 0 by construction.
- Also reported: closed-loop estimation error logged in each run (same near-pair MSE, on the arm's
  own trajectory).

## Arms and setup
- Arms: real (stale, noisy), cv (constant velocity from two broadcasts), learned (MLP, retrained
  below), smith (forward integration of the stale estimate under the routing field -grad Phi; since
  Phi is time-dependent here the field is sampled at each integration instant, nearest cell —
  the routing-only design of prereg_phase2d_smith amendment is unchanged).
- Geometry: single-gate far goal (amend4), horizon T = 57.8 s (amend5), N = 100, NU = 0,
  C_DENS = 1, CBF exact_hard, Phi/m* from cache (macro_N100_cw1_6ecdac0dea).
- Evaluation seeds: 0–4 (all arms, paired).

## Learned predictor: data and training (frozen recipe, ported unchanged)
- Data: `real`-arm closed-loop rollouts of this exact setup, FULL horizon (k sampled uniformly from
  k_min = 4 T_frame_steps — warm 4-broadcast buffer — to the end; no settled-only window).
  Datagen seeds disjoint from evaluation: train 100–107, val 108–109, test 110–111.
  200 000 (i, j, k) windows per seed, i != j uniform. Features/target identical to phase2b
  (20 features: 4 broadcasts x (noisy range, noisy bearing, age, ego displacement x, z); target = true
  current relative position of j from i).
- Training: 2 x 64 ReLU MLP, Adam lr 1e-3, batch 256, max 200 epochs, early stop on val MSE,
  patience 20, init seed 0, normalisation fitted on train and folded into the weights. Open-loop MSE
  (train / val / test) and the checkpoint SHA256 are recorded BEFORE any closed-loop evaluation of the
  learned arm.

## Reported facts (per arm, mean ± sd over seeds, plus per seed)
Liveness: completeness (final), arrived-fraction curve. Safety: collision pair-steps, min separation,
worst-case p5, CBF intervention rate, QP infeasibility (slack-active rate). Estimation: open-loop MSE,
closed-loop estimation error. Position of each arm relative to clean ceiling and real floor. Viz of the
best and the worst arm.

## Budget rule
If time does not allow 5 seeds for every arm, the seed count is reduced equally for all arms
(paired), stated in the report; datagen + training + ladder have priority.
