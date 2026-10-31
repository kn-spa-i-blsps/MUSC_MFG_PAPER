# REVIEW — Phase B gate G-pred (2026-11-01)

**Run:** logs/overnight/phaseB_gpred_chain_pod.sh, HEAD 13103ea, RunPod A100-SXM4-80GB, precision=highest, result
commit 01701ef. Criteria: logs/overnight/phaseB_gpred_gate.md (committed b020562 before any data). Frozen v3 predictor
sha 9583cb7c…. Error model g trained on seeds 100–107, val 108–109; all numbers below are TEST seeds 110–111, pairs
with true distance < 2 m.

## §3 verification
- 5 macro solves: converged (Cesàro), no NaN, cache keys equal to the earlier ones (N25 0fb517496a … N200 6490094688).
- datagen 60/60, datagenL 60/60 files.
- Integrity (v3 open-loop on the regenerated histories vs committed openloop_v3.json / openloop2_v3_datagenL.json):
  every key shared with the old files is bit-equal, except datagenL N=100 learned_new (mse_near1m 0.02239275910 vs
  0.02239276096, rel. 8e-8 = the known GPU nondeterminism of the learned arm). The `exact_equal=False` flags in the
  chain log come from keys added to the evaluator later (p99/p999, noise_only on datagen) that the old files lack —
  not from data differences. Data regeneration: **VERIFIED.**

## Results — PRIMARY (datagenL, default recipe D, pooled 2×128)
| N | T_frame | n | mean e [m] | auc_top10 | lift_top1 | r2_log | r2_abs | spearman | q90 cov (raw) |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 0.125 s | 28243 | 0.085 | 0.810 | 0.724 | 0.196 | 0.214 | 0.462 | 0.884 |
| 50 | 0.25 s | 24367 | 0.094 | 0.781 | 0.770 | 0.146 | 0.183 | 0.390 | 0.894 |
| 100 | 0.5 s | 20220 | 0.133 | 0.817 | 0.783 | 0.195 | 0.240 | 0.459 | 0.904 |
| 150 | 0.75 s | 18448 | 0.157 | 0.843 | 0.843 | 0.231 | 0.331 | 0.485 | 0.910 |
| 200 | 1.0 s | 17099 | 0.217 | 0.863 | 0.766 | 0.280 | 0.348 | 0.557 | 0.915 |
| pooled | | 108377 | 0.129 | 0.907 | 0.890 | 0.346 | 0.436 | 0.601 | 0.899 |

Criteria: P1 auc ≥ 0.80 at N 100/150/200 (0.817 / 0.843 / 0.863 ✓), ≥ 0.70 at N 25/50 (0.810 / 0.781 ✓); P2 lift ≥
0.40 at N 150/200 (0.843 / 0.766 ✓); P3 pooled r2_log ≥ 0.25 (0.346 ✓).

**Verdict (registered rule): PASS on the default recipe D.** The §C ladder was not needed. It ran anyway, so
robustness can be checked:

| variant | auc N100/150/200 | lift N150/200 | pooled r2_log |
|---|---|---|---|
| cap 3×256 | 0.822 / 0.842 / 0.861 | 0.838 / 0.713 | 0.357 |
| near_frac 0.8 | 0.834 / 0.853 / 0.869 | 0.896 / 0.783 | 0.367 |
| near_frac 0.0 | 0.797 / 0.813 / 0.859 | 0.591 / 0.646 | 0.290 |
| 200k windows | 0.832 / 0.851 / 0.862 | 0.856 / 0.755 | 0.368 |
| per-N models (ceiling) | 0.813 / 0.837 / 0.865 | 0.843 / 0.789 | per-N 0.13–0.27 |

All variants are within ±0.03 auc of D, except the uniform-sampling check (near_frac 0.0), which at N=100 falls just
under 0.80 (0.797). Per-N models do not beat the pooled model, so the pooled recipe leaves no headroom unused.

## Questions the PI asked
- **Reliability (calibration per σ decile):** predicted and realised medians agree within ~10 % in every decile at
  every N. Predicted medians run slightly low at N=25 (0.116 vs 0.137 m in the top decile) and slightly high at N=200
  (0.412 vs 0.383). The raw (pre-conformal) q0.9 head covers 0.88–0.92 overall. Per decile, coverage runs from 0.82
  (N=25, lowest decile) to 0.94 (N=200, top decile). This is near-calibrated before any conformal step; G-cal still has
  to be run as its own gate.
- **Heteroscedasticity:** realised median error rises monotonically across σ deciles: 0.056 → 0.255 m pooled, and
  0.102 → 0.383 m at N=200. The ratio of mean error, top vs bottom decile, is 2.6× (N=25) to 4.0× (N=200).
- **Tail:** ranking of the tail is strong. The top-10 % vs the rest has auc 0.78–0.86. 77–84 % of the top-1 % errors
  fall in the top-1 % of predicted σ at N ≥ 100 (random = 1 %). The tail magnitude is **not** regressed: within the
  top 10 %, r2_tail_log is negative at every N and spearman_tail is 0.30–0.39. g knows *which* neighbours are bad,
  not *how* bad the worst ones are. A σ-margin built on quantiles (conformal) only needs the first.
- **Large T_frame:** predictability does not degrade with staleness. It improves with N: auc 0.78 → 0.86, r2_log
  0.15 → 0.28 from N=50 to N=200 (T_frame 0.25 → 1.0 s).
- **σ beyond staleness:** the age-only baseline on PRIMARY is weak (auc 0.53–0.75, r2_log < 0 at every N). The range-only
  baseline is also weak (auc 0.56–0.62). σ uses per-neighbour information beyond age and range.

## Note against the PI's example threshold (not the registered one)
The PI's example in the step-1 instructions was R² > 0.5. The registered P3 used pooled r2_log ≥ 0.25, set before data
(phaseB_gpred_gate.md); the criteria were not moved. Against R² > 0.5: per-N r2_log is 0.15–0.28 and pooled r2_log is
0.35 on PRIMARY, so it is **not met**. On SECONDARY real-arm data, pooled r2_log is 0.625, but that is inflated by the
between-N spread of error. Stated plainly: the error *magnitude* explains 15–35 % of log-variance. The *ranking* (auc,
lift) is what is strong.

## SECONDARY (datagen, real-arm histories; not gating)
Pooled auc 0.928, lift 0.805, r2_log 0.625. Per N: auc 0.79–0.82, lift 0.37–0.78, r2_log 0.15–0.36. On real-arm data
age alone carries much of it (pooled r2_log 0.536; per-N auc 0.53–0.78), as expected for an uncompensated stale
estimate.

## Classification
No non-positive result to classify; nothing quarantined. One limitation, recorded: tail magnitude is not
regressed (r2_tail_log < 0). It is relevant for any claim that σ predicts the *size* of worst-case errors, which will
not be made.

## Consequence for Phase B
Under the registered rule the σ-CBF build may proceed: G-pred PASS, P1 holds at N ≥ 150. Next is step 2: the
conformal σ head and gate G-cal, pre-registered before the run. That step is not yet launched. The pod queue currently
runs amend27 → 28 → 29 → 30, and Phase B step 2 is to be registered and queued behind them.
