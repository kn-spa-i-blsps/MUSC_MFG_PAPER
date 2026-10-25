# Amendment 22 — N crossover with the pooled (in-distribution) predictor, precision=highest

**Base:** amend21 (fixed geometry, per-N Φ/m\*, precision=highest). amend21's learned arms were OOD off N=100 (the frozen
N=100-only predictor: open-loop MSE 5–6 m² at N=25 vs real 0.013) and were stopped. No controller change here; only the
predictor weights change. Frozen, pre-run, SHA before run, detached, self-push. Overnight brief Phase 1
(logs/overnight/AGENT_BRIEF_overnight.md).

## 1. Predictor (frozen before this commit)

- Same feature map, MLP form (2 hidden ReLU layers; width 128 here vs 64 before) and closed-loop code path (compensation/perception.perceive,
  condition `learned`) as amend6–21. The window already carries per-neighbour staleness explicitly (age_m =
  tdma_age_j + m·T_frame per block). What changed: training data = real-arm histories POOLED and balanced across
  N ∈ {25,50,100,150,200} on this geometry (datagen seeds 100–107 train, 108–109 val, 110–111 test; evaluation seeds 0–9
  never used), precision=highest. Script: experiments/overnight_p1_predictor.py (exploratory development; iterations and
  their gate numbers in logs/overnight/progress.md).
- Weights: `results/overnight_p1/predictor_v3/predictor_weights.npz (tag v3: near-pair oversampling near_frac 0.5 within 3 m, hidden 128 → MLP 20 → 128 → 128 → 2; 100k windows per seed per N; 36 epochs to early stop)`, sha256 `9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5`.
- OOD gate (fixed before any data, logs/overnight/progress.md): open-loop x̂ MSE, pairs < 1 m, test seeds 110–111:
  G1 learned ≤ real at every N; G2 learned ≤ 0.75·real at N ≥ 50; G3 at N=100 learned ≤ 1.25·(N=100-only predictor).
  Gate result (PASS) per N: x̂ MSE m² (<1 m) real / learned v3 / ratio — N=25 0.0117/0.0077/0.66, N=50 0.0224/0.0156/0.70, N=100 0.4625/0.2417/0.52 (N=100-only predictor 0.2446), N=150 1.5285/0.6114/0.40, N=200 3.1703/0.9725/0.31; v3 < real also on pairs <2 m and on all pairs at every N (results/overnight_p1/openloop_v3.json). Variants tried before freezing: v1 (recipe, hidden 64, uniform pairs) FAIL G1 at N=25 (1.05) and G2 at N=50 (0.85); v2 (near_frac 0.5, hidden 64) FAIL G1 at N=25 (0.0144 > 0.0117); v3 = first variant passing.

## 2. Design

- Geometry = amend21 (geometry_N 230, gate 2.309, arena 43.379 × 19.081 m, identical for all N); Φ/m\* per N = amend21
  cache keys, re-solved on the new pod biiz4t7r00oef8 (A100-SXM4-80GB; amend21's pod could not get its GPU back);
  identity: amend21 real seed 0 re-run on this pod at every N equals the committed amend21 metrics (all 5 N equal on 7 metrics, logs/overnight/p1/identity_vs_amend21.json).
- N ∈ {25, 50, 100, 150, 200}; T_frame = N·0.005 s. Seeds 0–9. precision=highest. D_COLLISION 0.30.
- Arms (constant margins only; closure-gating is dropped):
  - `real_d0.55` — the amend21 real arm, re-run (bit-identity vs amend21 checked in the report, all 50 cells);
  - `real_d0.70` — **margin-matched control** (amend21 compared real at 0.55 with learned at 0.70, which confounds
    compensation with margin);
  - `learned_pooled_d0.70` — pooled predictor + fixed d_safe 0.70.
- Logged: safe-success (amend12, c = 0.30), ever d<0.30, completeness, p5 worst-step separation, CBF intervention,
  slack, closed-loop x̂ MSE (near/all). Seed-cluster bootstrap 95% CIs; differences with unpaired seed bootstrap.

- Code state: experiments/ladder.py gained an optional cell key `t_slot` (commit 326464e; absent in every amend22
  cell → T_SLOT 0.005 unchanged; metrics json now also logs `t_slot`). No other code change since amend21.

## 3. Pre-registered questions and definitions

- **safe_N(arm)** = largest tested N such that at it and at every smaller tested N: ever<0.30 CI upper ≤ 0.10 and
  completeness mean ≥ 0.90.
- **Q1 (crossover gap):** safe_N(learned_pooled_d0.70) vs safe_N(real_d0.70) and safe_N(real_d0.55).
- **Q2 (compensation at matched margin):** per N, learned_pooled_d0.70 − real_d0.70 on safe-success and ever<0.30, with
  bootstrap CI; "learned better at N" only if the difference CI excludes 0.
- **Q3 (mechanism):** closed-loop x̂ MSE vs N per arm; learned < real at every N where learned is safer.
- **Q4 (OOD fixed in closed loop):** at N=25 learned_pooled is not worse than real (amend21's N=25 collapse gone).
- **Falsifiers:** learned_pooled worse than real_d0.70 at any N with separated CIs → compensation harms there; report.
  learned ≈ real_d0.70 everywhere → amend21's N=100 gap was margin, not compensation.
- Review per logs/overnight/AGENT_BRIEF_overnight.md §3–§4, written to logs/overnight/review_amend22.md.

## 4. Compute

150 rollouts (one process per cell, 3 in parallel) on the A100, ≈ 40 min; no solve.
