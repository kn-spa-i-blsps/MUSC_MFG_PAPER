# REVIEW — amend23 (fixed-margin re-tune, pooled predictor v3, Phase 2)

Run: aneks sha256 cb062a5e8bf22fae8d51eb089cafe329875ab3d419969b732321d307bd020021 (prereg commit ea0a5c5, before the
run); weights v3 9583cb7c…; result commit 118e4bd (DONE, 80/80); amend21 geometry (43.379 × 19.081 m, gate 2.309); min
initial sep 0.788 (N=100), 0.626 (N=150). The identity gate for the logging-only n_near change passed before the run.

## Data (learned_pooled v3; mean [95% CI], 10 seeds, precision=highest)

| N | d_safe | safe-success | ever<0.30 | completeness | p5 sep | interv. | slack | x̂ MSE near (pair-w.) | near pair-steps |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0.55 | 0.675 [0.645,0.705] | 0.325 [0.295,0.355] | 1.000 | 0.324 | 0.567 | 0.054 | 0.023 | 6.17 M |
| 100 | 0.70 | **0.892 [0.862,0.919]** | 0.072 [0.046,0.102] | 0.958 [0.949,0.967] | 0.464 | 0.903 | 0.192 | 0.023 | 5.62 M |
| 100 | 0.85 | 0.747 [0.734,0.759] | 0.020 [0.010,0.030] | 0.764 [0.755,0.773] | 0.566 | 0.944 | 0.286 | 0.022 | 4.57 M |
| 100 | 1.00 | 0.686 [0.675,0.696] | 0.010 [0.004,0.016] | 0.692 [0.682,0.702] | 0.669 | 0.945 | 0.340 | 0.022 | 1.73 M |
| 150 | 0.55 | 0.169 [0.153,0.185] | 0.831 [0.815,0.847] | 1.000 | 0.256 | 0.609 | 0.064 | 0.033 | 9.94 M |
| 150 | 0.70 | 0.479 [0.445,0.511] | 0.473 [0.433,0.511] | 0.857 [0.853,0.862] | 0.413 | 0.925 | 0.215 | 0.031 | 8.82 M |
| 150 | 0.85 | 0.486 [0.463,0.504] | 0.213 [0.187,0.239] | 0.577 [0.569,0.586] | 0.557 | 0.958 | 0.274 | 0.030 | 6.71 M |
| 150 | 1.00 | 0.411 [0.398,0.425] | 0.099 [0.082,0.115] | 0.453 [0.446,0.461] | 0.577 | 0.964 | 0.349 | 0.033 | 2.53 M |

**Selection by the pre-registered rule:**
- **N=100:** eligible margins (completeness ≥ 0.90) are {0.55, 0.70}; the best is 0.70 (the incumbent), so 0.70 is
  **kept, resolved**.
- **N=150:** only 0.55 is eligible, and its safe-success is 0.31 below 0.70 (difference CI [−0.347, −0.273]). The rule
  keeps 0.70 ("not resolved"). No margin at N=150 reaches completeness ≥ 0.90 with ever<0.30 below 0.47.
- **Operating margin for Phases 3–4:** 0.70.

## Step A — mechanism
- **The margin is a real knob with a monotone safety/liveness trade-off:**
  - A larger d_safe lowers ever<0.30 monotonically: 0.325 → 0.072 → 0.020 → 0.010 at N=100, and 0.831 → 0.473 →
    0.213 → 0.099 at N=150.
  - It raises p5 separation in the same direction.
  - It lowers completeness monotonically: 1.000 → 0.958 → 0.764 → 0.692 at N=100.
- **This is the expected CBF mechanism.** A larger keep-out radius means more pairs are constrained (intervention
  0.57 → 0.95), more QP infeasibility (slack 0.05 → 0.34) and more mutual blocking, so agents stall before the goal.
  Safe-success, which needs both safety and arrival, therefore peaks at an interior margin: 0.70 at N=100.
- **Estimation quality does not depend on the margin:** pair-weighted x̂ MSE is 0.022–0.023 at N=100 and 0.030–0.033
  at N=150 for every margin. The margin acts through the controller, not through estimation. Near pair-steps fall
  with margin, consistent with a sparser swarm.
- **Contrast with real (review_amend22):** with stale estimates a larger margin made real worse; with compensation a
  larger margin makes learned safer (but slower). This fits the mechanism: the margin only helps when the estimate
  it is applied to is accurate.

## Step B — artifact scan (§3)
1. **Identity:** PASS. The logging-only change passed its identity gate (real seed 0, N=25/100, 8 metrics equal to
   amend21).
   - **Reproducibility of learned:** the d0.70 cells repeat amend22's config.
     - At N=150 every reported metric is identical to amend22 (0.479 [0.445, 0.511] etc.); on this pod and precision
       the learned rollout repeated bit-for-bit there.
     - At N=100: 0.892 [0.862, 0.919] vs amend22 0.880 [0.848, 0.908]. Not identical, but within CI. This matches the
       known GPU nondeterminism of the learned path in some graphs (the N=100 and N=150 executables differ in shape).
2. **Precision:** PASS. All 80 rows are "highest" (chain log).
3. **OOD:** PASS. N=100 and N=150 are inside v3's gate (open-loop v3/real 0.52 / 0.40). Closed-loop pair-weighted x̂ MSE
   is 0.022–0.033, with no blow-up at any margin.
4. **Wrong knob:** PASS. The margin moves ever<0.30 by 30× at N=100.
5. **Margin starvation:** n/a. Margins are constant.
6. **Geometry/density:** PASS within the sweep (one geometry per N). Note: at N=150, d ≥ 0.70 starts below the initial
   spacing (0.626); at N=100, d ≥ 0.85 does too (0.788).
7. **Statistics:** PASS. At N=100, 0.70 beats 0.55 (0.892 vs 0.675), 0.85 (0.747) and 1.00 (0.686) with
   non-overlapping CIs. At N=150, 0.70 vs 0.85 safe-success CIs overlap (0.479 vs 0.486), so it is a tie; they differ
   in where the loss falls (collisions vs freezing).
8. **Completeness-as-sham:** FLAGGED and handled. d 0.85/1.00 reach ever<0.30 of 0.02/0.01 at N=100 only by freezing
   (completeness 0.76/0.69). That is why the selection rule requires completeness ≥ 0.90; those cells are not reported
   as "safe".

## Step C — consistency
- **Agrees:**
  - amend18/20 (old predictor, tighter arena): 0.70 was the best constant margin there as well.
  - amend22: the d0.70 rows reproduce.
- **New:** compensation reverses the sign of the margin effect seen for real in amend22.
- **Refines §0:**
  - "No hard collision-free guarantee under large staleness" holds. At N=100 the only way below ever<0.30 of 0.02 is
    freezing.
  - At N=150 no margin achieves both safety and arrival. This is consistent with the density limitation suspected in
    review_amend22, which amend24 tests.

## Step D — statistics
Claims are made only where CIs separate. The N=150 choice between 0.70 and 0.85 is unresolved (overlapping CIs);
0.70 is kept by the pre-registered rule.

## Step E — verdict
- **CONFIRMED:**
  - **Operating margin d_safe = 0.70 at N=100** (pre-registered rule, resolved).
  - **Margin trade-off:** safety vs completeness are monotone in d_safe with compensation.
  - **amend22 learned rows reproduce:** N=150 identical, N=100 within CI.
- **NOT RESOLVED (reported as such):** the best margin at N=150. There, every margin trades collisions for freezing,
  and no cell is safe and complete.
- **No quarantine.** Fix list: none for this result. amend24 (mechanism control) addresses the N=150 limitation.
