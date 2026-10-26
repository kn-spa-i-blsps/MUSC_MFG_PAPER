# Phase B — gate G-pred, criteria fixed BEFORE any data (exploratory gate, like v4's G4)

Code: experiments/phaseB_gpred.py. Frozen predictor: v3 (results/overnight_p1/predictor_v3, sha 9583cb7c…).
Error e = ||v3(X) − Y|| [m] per (i, j, t) window, same 20-dim feature map as perceive(). Error model g trained on seeds
100–107 (val 108–109), metrics on TEST seeds 110–111, subset true distance < 2 m. precision=highest.

## Data
- PRIMARY: `datagenL` = closed-loop learned-v3 d0.70 histories at N ∈ {25,50,100,150,200} (the distribution the σ-CBF
  operates on).
- SECONDARY (reported, not gating): `datagen` = real-arm histories.

## Default recipe (variant "D")
Pooled, N-balanced (60k windows / seed / N train, 30k val/test), near_frac 0.5, g = MLP 2×128, heads log|e| (MSE) +
q0.9 (pinball). Score s = predicted log|e|.

## PASS (all, default or a §C fix variant, on PRIMARY test seeds)
- P1 `auc_top10` ≥ 0.80 at each N ∈ {100,150,200}, and ≥ 0.70 at N ∈ {25,50}.
- P2 `lift_top1` ≥ 0.40 (4× random) at N ∈ {150,200}.
- P3 pooled `r2_log` ≥ 0.25 (and `r2_abs` reported).

## FAIL (THESIS-CRITICAL NEGATIVE) — only after the §C ladder below
- `auc_top10` < 0.65 OR `lift_top1` < 0.20 at N = 150 or N = 200, for every variant of the ladder.

Anything between PASS and FAIL = PARTIAL (reported as such; the σ-CBF build may proceed only if P1 holds at N ≥ 150).

## §C fix ladder (run only if D does not PASS; the variant is chosen by VAL metrics, reported on TEST)
1. Capacity: g = 3×256.
2. Near-pair oversampling: near_frac 0.8; and 0.0 (uniform) as a check.
3. More windows: 200k / seed / N.
4. Per-N error models (diagnostic ceiling; if per-N passes and pooled fails → PLUMBING, fix the pooled recipe).
5. If the age-only baseline matches full: σ is a function of staleness alone — reported (it still varies per
   neighbour), not a failure.

## Proxy (not the gate)
Run locally while the pod is unavailable (RunPod 402): N=100 only, amend11 geometry
(results/ladder_fargoal_N100/datagen, real-arm), the old N=100 predictor (in-distribution there). Diagnostic signal only;
cannot PASS or FAIL the gate.
