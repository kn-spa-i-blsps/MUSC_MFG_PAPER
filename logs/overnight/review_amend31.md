# REVIEW — amend31 (Phase B steps 2–3): σ head + G-cal, σ-CBF + G-win

Pod sbk37l6nvrg36q (A100-SXM4-80GB, new volume). Aneks 12678c58… + addenda 1 (7278e98b…), 2, 3 (32a06cbe…).

## Run history (infra)
- Run 1 (HEAD f90bea6):
  - stage 0 regenerated the 5 caches, with keys equal to the earlier ones (N25 0fb517496a … N200 6490094688), and
    datagenL 60/60 (4e18618);
  - stage 1 trained the σ head and ran G-cal (1349c30);
  - the identity gate FAILED (e662d89) and the chain aborted before any σ-CBF rollout.
- Diagnosis (addendum 3):
  - The pod environment reproduces amend22: fixed 0.70 real seed 0 is bit-equal.
  - The k=0 σ path is deterministic.
  - The first divergence is at step 0 in the CBF u (8 ulp). Cause: d_safe² in f32 (0.48999998) vs the folded constant
    (0.49).
  - Fixed in cbf.build_rows (σ mode, floor rows). Verified bit-equal to amend22 on all IDENTITY_KEYS.
  - **PLUMBING, fixed, not waived.**
- Run 2 (HEAD 459d624): stage 1 reused (no retraining), then identity, then rollouts.
- amend28 was started by the queue and stopped by me during its first solve (5ff7869, logs only, no rollout).

## G-cal (stage 1, TEST seeds 110–111, pairs with true distance < 2 m, uniform windows, precision=highest)
§3 check:
- split: train 100–105, val 106–107, CAL 108–109, TEST 110–111, as registered;
- n_cal: 129k / 81k / 61k / 44k / 31k (N 25…200); n_test: 132k / 82k / 61k / 44k / 31k;
- the error model on TEST reproduces G-pred: auc_top10 0.81 / 0.78 / 0.82 / 0.83 / 0.86, pooled r2_log 0.31.

**Verdict: PASS_AFTER_FIX** (registered). Primary G-win arm = sigma_bin_a0.05.

| | marginal (all N, all α) | conditional α=0.10: min decile coverage N 25/50/100/150/200 | gate |
|---|---|---|---|
| per-N conformal | within tolerance everywhere (0.897–0.908; 0.948–0.956; 0.989–0.992) | 0.874 / 0.874 / 0.868 / 0.880 / **0.842** | fails conditional at N=200 |
| (N, σ-quintile) conformal | within tolerance everywhere (0.896–0.909; 0.948–0.955; 0.990–0.992) | 0.878 / 0.876 / 0.886 / 0.895 / 0.886 | **pass** |

- **Shape of the per-N miscalibration:** low-σ deciles are under-covered (α=0.10: 0.84–0.89) and high-σ deciles
  over-covered (0.94–0.97). The normalized score e/exp(g) is not scale-free.
- **Reliability** (α=0.10, per-N, σ_c vs realised q90 per decile): within ±0.02 m in deciles 1–8 at every N. In the
  top decile σ_c over-predicts: 0.25/0.22 (N=25), 0.39/0.33 (N=100), 0.51/0.42 (N=150), 0.80/0.59 m (N=200). The
  margin is conservative exactly where it binds. The binned calibration removes most of this: σ_c mean 0.346 vs 0.364 m
  at N=200.
- **Age terciles:** flat (α=0.10: 0.88–0.93), so coverage does not depend on staleness.
- **Rows where the margin binds** (σ_c > 0.40 m, i.e. 0.30 + σ_c > 0.70):
  - binned coverage 0.92–0.97 (α=0.10), 0.955–0.98 (α=0.05), ≥0.989 (α=0.01);
  - binding fraction at α=0.05 binned: 0.002 / 0.007 / 0.032 / 0.10 / 0.41 (N 25…200).
  - The σ margin is essentially inactive at N ≤ 100 and active at N = 200.

**Where the guarantee's premise holds / breaks (offline):**
- Holds: marginally at every N and α. Conditionally on σ after the registered binned fix (worst decile 0.876 at
  α=0.10, 0.938 at α=0.05, 0.985 at α=0.01).
- Breaks:
  - per-N conformal under-covers low-σ pairs at N=200 (0.842);
  - the guarantee is per row, marginal over rows: at α=0.05, 5 % of near rows are uncovered by construction, so it does
    not bound trajectory-level collision probability;
  - TEST is datagenL (fixed 0.70 closed loop). Under the σ-CBF the distribution shifts. Closed-loop coverage is
    measured in every σ rollout (stage 3) and is the real test.
- Not covered by the statement: discrete time, QP slack, k_nb truncation, rows outside the CBF set.

**Classification:** the per-N conditional failure at N=200 is PLUMBING, resolved by the registered calibration fix.
G-cal is not a thesis-critical negative.

## G-win (registered report 29e1496, filled in 2026-11-03 from saved results only; no new run)

Primary arm sigma_bin_a0.05; comparator = best fixed per N by mean safe-success (fixed0.85 at N=150 and N=200);
10 seeds (0–9); seed-level bootstrap 95 % CI. PASS needed: lower CI of the advantage > 0 at N=150 AND N=200, with
completeness non-inferior (−0.02, addendum 1).

| N | σ-CBF safe / ever / compl | best fixed (0.85) safe / ever / compl | Δ safe [CI95] | Δ compl [CI95] |
|---|---|---|---|---|
| 150 | 0.327 / 0.617 / 0.863 | 0.486 / 0.213 / 0.577 | −0.159 [−0.197, −0.112] | +0.285 [+0.273, +0.297] |
| 200 | 0.000 / 1.000 / 0.348 | 0.188 / 0.608 / 0.408 | −0.188 [−0.200, −0.174] | −0.060 [−0.078, −0.041] |

- **Verdict: NOT_MET** (registered v1 and addendum 1). Every σ arm (α 0.10 / 0.05 / 0.01 / binned) loses to the best
  fixed margin at N = 100, 150 and 200. At N = 200 every σ arm has every drone collide (ever 1.000).
- Replication on seeds 200–209 (addendum 4, supplementary) gives the same sign and size.
- σ-CBF does **not** move off the fixed-margin frontier at N ≥ 150. At N=150 it trades ~0.16 safe-success for
  +0.29 completeness, i.e. it behaves like a smaller fixed margin (it matches fixed0.70 / conf_const). At N=200 it is
  strictly worse on both.

**Closed-loop coverage (the real test of the guarantee, addendum 4 §4):** offline TEST 0.955 at α=0.05; in closed
loop 0.938 at N=150 but 0.701 at N=200 (pairs < 2 m), 0.649 on binding rows. The conformal guarantee does not survive
the distribution shift that the σ-CBF itself creates at N=200.

## Classification

- **G-cal per-N conditional miss at N=200 → PLUMBING**, fixed by the registered (N, σ-quintile) calibration.
- **Identity-gate failure in run 1 (d_safe² in f32) → PLUMBING**, fixed, not waived.
- **G-win NOT_MET → THESIS-CRITICAL.** The σ-CBF as registered gives no safety advantage over a fixed margin at the
  densities it was built for, and its guarantee breaks in closed loop at N=200.
- **Known implementation defects that may contribute** (review_amend31_diag.md, read-only): no cap on σ_c / d_safe
  (mean d_safe in one step up to 2.8e5 m at N=200), no ḋ_safe term in the CBF row, σ_ij ≠ σ_ji without a shared
  pair margin. The logs (log_xhat false) cannot decide whether these or the method itself drive the failure (H1–H5
  UNDECIDED). Fixing them is a design change and would need a new prereg; it would not change this verdict.
