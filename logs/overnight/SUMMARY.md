# Overnight summary (2026-10-25/26) — draft, completed at the end of the night

Chronological log: logs/overnight/progress.md. Per-result reviews: logs/overnight/review_amend22.md … review_amend26.md.
Figures (CONFIRMED cells only): results/overnight_figures/ (PNG + figure_data.csv). Pod: biiz4t7r00oef8 (A100-SXM4-80GB).

## CONFIRMED (review verdict CONFIRMED; CIs separate where a comparison is claimed)

| # | Claim | Evidence |
|---|---|---|
| 1 | Pooled-N training fixes the predictor's OOD failure: v3 open-loop x̂ MSE (near pairs) ≤ real at every N, ratio 0.66 / 0.70 / 0.52 / 0.40 / 0.31 (N = 25…200); the N=100-only predictor gives 5.95 at N=25 and 25.2 at N=200 | progress.md, results/overnight_p1/openloop_v3.json, fig2 |
| 2 | Staleness barrier for real at both margins: safe_N = 25 (d 0.55 and 0.70) | amend22, review_amend22, fig1 |
| 3 | Compensation gap at matched margin: learned v3 d0.70 − real d0.70 safe-success +0.936 (N=50), +0.880 (N=100), +0.479 (N=150), +0.137 (N=200), CIs exclude 0; learned safe_N = 50 (N=100 misses only on CI upper 0.106 vs 0.10) | amend22 |
| 4 | Larger margin does not help real (0.70 worse than 0.55) but does help learned — the margin helps only on an accurate estimate | amend22, amend23 |
| 5 | Operating margin d_safe = 0.70 at N=100 (pre-registered rule, resolved); larger margins buy safety only by freezing | amend23, fig3 |
| 6 | Mechanism: with the TDMA delay removed (noise_only), ever<0.30 = 0 at every N up to 200 (same controller, margin, packing, sensor noise) → collisions in the crossover are caused by delay-estimation error, not density; v3 leaves +0.12 / +0.52 / +0.83 safe-success headroom at N = 100 / 150 / 200 | amend24, review_amend24, fig1 |

## Not resolved / negative (reported as such)
- Best margin at N=150: no margin both safe and complete; 0.70 vs 0.85 tie (amend23).
- (filled in as results arrive: amend25, amend26, predictor v4)

## Quarantined / ARTIFACT / excluded — and why
| Item | Why |
|---|---|
| amend21 learned rows at N=25 (fixed 0.70 / closing 0.75) | N=100-only predictor OOD at N=25 (open-loop 5.95 vs real 0.012); superseded by amend22 |
| Closed-loop x̂ MSE ("closedloop_xhat_mse_near") as a CROSS-ARM comparison (amend21–22) | per-step mean with steps lacking near pairs entering as 0 → biased low for separated swarms; replaced by the pair-weighted metric (from amend23) and the open-loop gate |
| Predictor v1, v2 | failed the pre-declared open-loop gate (v1: G1 at N=25 1.05, G2 at N=50 0.85; v2: G1 at N=25); never used in a closed-loop run |
| review_amend22's density hypothesis for learned at N ≥ 150 | falsified by amend24 |
| fresh (no-noise oracle) completeness at N=200 (0.763 vs noise_only 0.968) | unexplained liveness observation, not a claim |

## Infra incidents (no effect on any reported number)
- New pod (old A100 host had no free GPU) → all 5 per-N caches re-solved; identity vs amend21 real at every N passed.
- GPU OOM when rollouts ran concurrently with a solve (P1a attempt 1) or 3 rollouts in parallel (amend22 attempt 1);
  fixed by ordering / P=2; completed seeds were whole files and were kept.
- Logging-only rollout change (n_near) passed an identity gate before use.
