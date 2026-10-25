# REVIEW — amend22 (N crossover, pooled predictor v3)

Run: aneks sha256 ceb6aa4e077814622a068bd99b5f90c5177bdba9f48154c8984b76085fc02aa0 (prereg commit 22b0e6b, before any
run); weights v3 sha256 9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5; result commit c24e62f
(DONE, 150/150); pod biiz4t7r00oef8 (A100-SXM4-80GB). Geometry: amend21 fixed arena 43.379 × 19.081 m, gate 2.309 m,
goal disc r 8.361, start box [0.4, 8.156] × [3.374, 15.708]; min initial separation over seeds 0–9 per N =
1.669 / 1.075 / 0.788 / 0.626 / 0.553 m (N = 25/50/100/150/200); T_frame = 0.125/0.25/0.5/0.75/1.0 s.

## Data (mean [95% seed-cluster bootstrap CI], 10 seeds, precision=highest everywhere)

| N | arm | safe-success | ever d<0.30 | completeness | p5 sep | interv. | slack | cl. x̂ MSE near |
|---|---|---|---|---|---|---|---|---|
| 25 | real 0.55 | 1.000 | 0.000 | 1.000 | 0.475 | 0.413 | 0.018 | 0.012 |
| 25 | real 0.70 | 1.000 | 0.000 | 1.000 | 0.610 | 0.563 | 0.048 | 0.012 |
| 25 | learned 0.70 | 1.000 | 0.000 | 1.000 | 0.502 | 0.576 | 0.040 | 0.008 |
| 50 | real 0.55 | 0.284 [0.214,0.352] | 0.716 [0.648,0.786] | 1.000 | 0.153 | 0.454 | 0.033 | 0.023 |
| 50 | real 0.70 | 0.056 [0.034,0.078] | 0.944 [0.922,0.966] | 1.000 | 0.085 | 0.638 | 0.098 | 0.039 |
| 50 | learned 0.70 | **0.992 [0.980,1.000]** | **0.008 [0.000,0.020]** | 1.000 | 0.483 | 0.660 | 0.076 | 0.012 |
| 100 | real 0.55 | 0.000 | 1.000 | 1.000 | 0.032 | 0.586 | 0.159 | 0.461 |
| 100 | real 0.70 | 0.000 | 1.000 | 0.994 | 0.035 | 0.705 | 0.234 | 0.888 |
| 100 | learned 0.70 | **0.880 [0.848,0.908]** | **0.076 [0.048,0.106]** | 0.950 [0.944,0.957] | 0.463 | 0.903 | 0.193 | 0.023 |
| 150 | real 0.55 | 0.000 | 1.000 | 0.972 | 0.040 | 0.667 | 0.227 | 1.536 |
| 150 | real 0.70 | 0.000 | 1.000 | 0.924 | 0.044 | 0.751 | 0.295 | 2.568 |
| 150 | learned 0.70 | 0.479 [0.445,0.511] | 0.473 [0.433,0.511] | 0.857 [0.853,0.862] | 0.413 | 0.925 | 0.215 | 0.032 |
| 200 | real 0.55 | 0.000 | 1.000 | 0.883 | 0.036 | 0.720 | 0.268 | 3.115 |
| 200 | real 0.70 | 0.000 | 1.000 | 0.793 | 0.023 | 0.792 | 0.350 | 4.883 |
| 200 | learned 0.70 | 0.137 [0.121,0.154] | 0.827 [0.807,0.846] | 0.599 [0.593,0.605] | 0.328 | 0.948 | 0.243 | 0.050 |

Differences: learned 0.70 − real 0.70 safe-success = +0.936 [0.912,0.960] at N=50; +0.880 [0.848,0.908] at N=100;
+0.479 [0.445,0.511] at N=150; +0.137 [0.121,0.154] at N=200; 0 at N=25. The CIs exclude 0 at every N ≥ 50.
Pre-registered safe_N: real 0.55 = 25, real 0.70 = 25, learned 0.70 = 50. At N=100, learned misses the safe_N definition
only because the CI upper on ever<0.30 is 0.106, against the 0.10 threshold.

## Step A — mechanism
- **Hypothesis:** collisions come from position-estimate error at the CBF, and that error grows with staleness
  (T_frame = N·t_slot). Removing the delay error should restore safety until density or controller limits take over.
- **Evidence it fits:**
  - The open-loop gate on fixed real trajectories gives v3/real x̂ MSE of 0.66, 0.70, 0.52, 0.40 and 0.31 for
    N = 25 to 200. The error removed grows with N, and so does the safety gap at N=50 and N=100.
  - Real's closed-loop x̂ MSE rises monotonically: 0.012, 0.023, 0.461, 1.536, 3.115. Real collides at every
    N ≥ 50 and is collision-free at N=25.
  - Learned's closed-loop x̂ MSE stays at 0.008–0.050.
- **What does NOT fit cleanly:** learned degrades strongly at N=150 and N=200 (ever<0.30 of 0.473 and 0.827) while
  its closed-loop x̂ MSE stays small (0.032 and 0.050). The mean estimation error therefore does not explain the
  N ≥ 150 learned collisions. Three candidates:
  - **(i) Density/packing:** min initial separation is 0.626 / 0.553 at N=150/200, which is below d_safe 0.70. The
    swarm starts inside its own margin. Slack is 0.215 / 0.243 and intervention 0.925 / 0.948, so the CBF is almost
    always active and often infeasible. Completeness drops to 0.857 / 0.599, a congestion signature.
  - **(ii) Tail estimation error:** the metric is a mean, so rare large errors are invisible to it.
  - **(iii) Metric artefact (found in this review):** see the next bullet.
- **Metric artefact:** `closedloop_xhat_mse_near` is the time average of per-step means over pairs closer than
  1 m, and a step with no such pair contributes 0 (env/rollout.py, xhat_err_near). For a well-separated swarm
  (learned) many steps have no near pair, which biases the metric low. Real swarms that collapse always have near
  pairs. So the closed-loop x̂ MSE is **not** a clean like-for-like comparison between arms. The open-loop gate
  numbers (same trajectories for every estimator) are the clean mechanism evidence. The closed-loop column is kept
  as reported but labelled "not comparable across arms" for the mechanism panel.
- **Unexpected:** real at 0.70 is worse than real at 0.55: safe-success 0.056 vs 0.284 at N=50; at N=100 completeness
  is 0.994 vs 1.000. Its x̂ MSE is also higher (0.039 vs 0.023 at N=50; 0.888 vs 0.461 at N=100).
  - **Hypothesised mechanism:** a larger margin with stale estimates means more CBF intervention (0.638 vs 0.454) and
    slack (0.098 vs 0.033). That produces more evasive, high-acceleration motion, and since the stale estimate errs by
    roughly velocity change × age, more manoeuvring means larger delay error.
  - **Consequence:** a bigger margin does not substitute for compensation. This strengthens the claim that the
    learned-vs-real gap is compensation, not margin.

## Step B — artifact scan (§3)
1. **Identity:** PASS. real_d0.55 equals amend21 on 7 metrics in all 50 seed×N cells (report:
   identity_real_d0.55_vs_amend21.all_equal = True). The pod/cache identity at seed 0 was also checked in P1a.
2. **Precision:** PASS. matmul_precision is "highest" in all 150 metrics files (chain log and report).
3. **OOD:** PASS. Open-loop gate on held-out test seeds at every N: v3 ≤ real at every N (ratio 0.31–0.70), and also
   on pairs < 2 m and on all pairs. The old predictor's blow-up (5.95 at N=25) is gone. Closed loop at N=25:
   learned is not worse than real (both 1.000 / 0.000), so the amend21 N=25 collapse no longer occurs.
4. **Wrong knob:** PASS. N moves every arm's outcome monotonically.
5. **Margin starvation:** n/a. Margins are constant (realised d_safe ≡ 0.55 or 0.70).
6. **Geometry/density confound:** PARTIAL. Geometry is fixed across N, but by design density rises with N together
   with staleness (amend21 §3). At N ≥ 150 the start spacing is below d_safe 0.70, so the N ≥ 150 cells mix staleness
   with packing. The N=25–100 cells start at spacing ≥ 0.788 > 0.70.
7. **Statistics:** PASS for the claims made. Learned vs real differences have CIs excluding 0 at N = 50, 100, 150, 200.
   N=25 shows no difference (both perfect).
8. **Completeness-as-sham:** PASS at N ≤ 100 (completeness ≥ 0.95). FLAG at N=150 (0.857) and N=200 (0.599): learned
   is partly frozen there. Safe-success already requires reaching the goal, so freezing does not inflate it, but
   learned's ever<0.30 at N=200 must be read together with its 0.599 completeness.

## Step C — consistency with §0
- **Agrees with:**
  - The staleness barrier for real: bit-identical to amend21.
  - "Compensation gives large headroom at N=100": learned 0.880 now vs 0.805 with the N=100-only predictor in
    amend21. These are different predictors and different runs; there is no claim they differ.
- **Corrects:** §0 "predictor is OOD-brittle" is now resolved for N ∈ [25, 200] by pooled training. The brittleness
  stays a stated limitation of single-N training (amend21 numbers: old predictor open-loop 5.95 at N=25 and 25.2 at
  N=200).
- **Refines:** §0 "learned holds safety to much higher N". It holds to N=50 strictly by safe_N and to N=100 nearly
  (ever 0.076 [0.048, 0.106]). From N=150 it degrades with a mechanism not yet identified (Step A).

## Step D — statistics
10 seeds suffice for every learned-vs-real claim, since the effect sizes (0.14–0.94) are far outside the CI widths
(≈ 0.03–0.07). The N=100 safe_N boundary (CI upper 0.106 against the 0.10 threshold) is genuinely marginal. More
seeds would narrow it, but that is a threshold question, not an effect question, and it is not needed for the
headline.

## Step E — verdict + fix list
- **CONFIRMED:**
  - **Staleness barrier for real:** safe at N=25, unsafe from N=50 on, at both margins.
  - **Compensation gap at N = 50 and 100:** learned 0.70 vs margin-matched real 0.70, with separated CIs.
  - **Pooled-training fix of the OOD failure:** open-loop gate plus the closed-loop N=25 result.
  - **Larger margin does not help real.**
- **SUSPECT (mechanism):** learned at N=150 and N=200. The rows are measured, but what causes their collisions is
  unidentified. They go to the figure only with an explicit caveat, or after the fix below.
- **QUARANTINED for the mechanism panel:** the closed-loop x̂ MSE column as a cross-arm comparison (biased estimator,
  Step A). Use the open-loop gate numbers instead.
- **FIX LIST:**
  1. **(top) amend24 mechanism control:** run `noise_only`, which keeps sensor noise but has zero delay, i.e. perfect
     compensation, at d_safe 0.70 for all 5 N with seeds 0–9.
     - If noise_only also fails at N=150/200, the residual failure is density or controller (packing below d_safe),
       not staleness or predictor.
     - If noise_only is safe there, the residual is predictor tail error. That would point to a better predictor,
       not a different margin.
  2. **amend23 (Phase 2):** margin re-sweep d ∈ {0.55, 0.70, 0.85, 1.00} with learned v3 at N=100 and N=150. The 0.70
     cell doubles as a reproducibility replicate of amend22. Note the Step A finding that a larger margin can hurt
     under staleness.
  3. **Mechanism metric:** log the near-pair count per step (or a per-pair-weighted x̂ MSE) in future runs so that
     closed-loop x̂ MSE is comparable across arms. This is a logging-only change and needs an identity check.
