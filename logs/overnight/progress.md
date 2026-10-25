# Overnight progress log (brief: logs/overnight/AGENT_BRIEF_overnight.md)

Append-only. Times local (Europe/Warsaw). Facts first; reasoning marked as such.

## 2026-10-25 ~18:05 — start
- Brief saved as logs/overnight/AGENT_BRIEF_overnight.md (verbatim from the PI message).
- Pod: A100 r863dyrq106k4m (amend21, holds the 5 per-N Φ/m* caches) could NOT start ("not enough free GPUs on the
  host"). Created NEW pod biiz4t7r00oef8 (A100-SXM4-80GB, SECURE, US-WA-1, $1.79/h, 150 GB volume) — same GPU type as
  amend21. Consequence: the 5 per-N caches must be re-solved (~10 min each); identity check planned: amend21 real
  seed 0 at every N must reproduce committed amend21 metrics on this pod (VERIFICATION §3.1, cache/pod identity).

## Phase 1 plan (written BEFORE any data)
Finding from reading the code (fact): the existing 20-dim predictor input ALREADY contains per-neighbour staleness
explicitly: each of the 4 window blocks carries age_m = tdma_age_j + m·T_frame (compensation/perception.py
learned_features_dense; experiments/train_predictor.py extract_windows). T_frame is recoverable as age_1 − age_0.
So "Δt as explicit input" is satisfied by the existing feature map; the N=100 predictor's OOD failure at N=25 comes from
the TRAINING DISTRIBUTION (only N=100: ages on a T_frame = 0.5 s lattice, window span 2 s; at N=25 window span 0.5 s).
Fix under test: same feature map + same perceive() path (no closed-loop code change, only new weights), trained on
POOLED, N-BALANCED real-arm histories at N ∈ {25,50,100,150,200} on the amend21 fixed geometry, precision=highest.
If the gate fails, iterate (near-pair oversampling, wider MLP, residual target) — exploratory, not pre-registered.

Data: datagen (real arm, d_safe 0.55 = amend21 real cells, new run_name) seeds 100–111 per N
(train 100–107, val 108–109, test 110–111). Evaluation seeds 0–9 are never used for training.

### OOD GATE (fixed before seeing any numbers)
Open-loop x̂ MSE with the closed-loop perceive(), test seeds 110–111, 400 steps/seed, pairs with true d < 1 m:
- G1 (no blow-up): learned_new ≤ real (stale) at EVERY N.
- G2 (useful compensation): learned_new ≤ 0.75 × real at every N ≥ 50 (N=25 exempt: stale error there is near the
  sensor-noise floor, nothing to compensate).
- G3 (no loss vs the specialist): at N=100, learned_new ≤ 1.25 × learned_old (the N=100-trained predictor).
- Also reported: cv, learned_old at every N (learned_old expected to blow up at N≠100 — reproduces amend21 OOD).
Gate PASS = G1 ∧ G2 ∧ G3. No crossover run before PASS.

### Crossover design note (reasoning, decided before data)
amend21's real arm ran at d_safe 0.55, learned at 0.70 → real-vs-learned confounds compensation with margin. The
amend22 crossover therefore adds a margin-matched control `real fixed 0.70`; arms: real 0.55 (identity vs amend21),
real 0.70, learned_pooled 0.70.

## P1a attempt 1 — infra failure (no data)
- Solve N=25 on the new pod: cache key macro_N25_cw1_0fb517496a (= amend21 key), Cesàro converged, no NaN.
- The chain started identity+datagen for N=25 concurrently with the N=50 solve → GPU OOM in the rollout
  ("RESOURCE_EXHAUSTED ... 5.12GiB ... jit_run_arrays"). No datagen produced. Chain killed; fix: all solves first, then
  identity/datagen 2 processes at a time (commit a75a91f). Relaunched (pod log time = UTC). Attempt-1 log kept as
  logs/overnight/p1/p1a_chain_pod_attempt1.log.

## P1a relaunch — results (pod times UTC)
- Per-N solves (A100, ~11 min each): keys identical to amend21 (N25 0fb517496a, N50 b4c8c86cde, N100 14d351fc8e,
  N150 2f21768897, N200 6490094688), all Cesàro-converged, no NaN.
- **Identity (§3.1, pod/cache):** amend21 real seed 0 re-run on the new pod at every N → 7 metrics equal to the committed
  amend21 values at all 5 N (all_equal = True, logs/overnight/p1/identity_vs_amend21.json).
- Datagen: 60/60 histories (5 N × seeds 100–111).
- Predictor **v1** (recipe unchanged, pooled N-balanced, 100k windows/seed/N, hidden 64): 106 epochs, sha256
  ff12c92e06837990976d181c368d028d0d522e25557ec79d7187c9e8283c4a8e.
- The chain's open-loop step failed (old N=100 weights not present on a fresh clone; untracked file) → copied the
  local file (sha256 fc4a52a5…, matches record) and ran open-loop by hand.

### Open-loop gate, v1 (x̂ MSE m², pairs < 1 m, test seeds 110–111)
| N | real | cv | learned_old (N=100-only) | learned v1 (pooled) | v1/real |
|---|---|---|---|---|---|
| 25 | 0.0117 | 0.0299 | 5.9522 | 0.0123 | 1.05 |
| 50 | 0.0224 | 0.0551 | 4.1892 | 0.0190 | 0.85 |
| 100 | 0.4625 | 0.7348 | 0.2446 | 0.2582 | 0.56 |
| 150 | 1.5285 | 2.3945 | 5.2134 | 0.7705 | 0.50 |
| 200 | 3.1703 | 5.1619 | 25.2162 | 1.5399 | 0.49 |

- The OOD blow-up of the N=100-only predictor is reproduced (5.95 at N=25, 25.2 at N=200) and is GONE with pooled
  training (v1 within 5% of real at N=25, ≈ half of real at N ≥ 100).
- **Gate v1: FAIL** under the pre-declared thresholds: G1 fails at N=25 (1.05 > 1.00), G2 fails at N=50 (0.85 > 0.75);
  G3 passes (0.2582 ≤ 1.25 × 0.2446). No crossover on v1.
- Reasoning: at N=25/50 the stale error is close to the sensor-noise floor (range σ, bearing σ), so there is little
  delay error to remove; the training loss is dominated by FAR pairs (uniform j), whose error scale is much larger than
  near pairs. Next (exploratory): v2 = near-pair oversampling (near_frac 0.5, j within 3 m), v3 = v2 + hidden 128.
  Thresholds are NOT changed.
- Decision rule for the iterations, written BEFORE v2/v3 numbers: pick the first variant passing G1∧G2∧G3. If none
  passes G2 at N=50 but one passes G1 (≤ real at every N) and G3, that variant is still IN-DISTRIBUTION (the brief's
  gate is "no OOD blow-up"); the crossover may then run on it with the deviation stated in the aneks ("G2 failed at
  N=50: compensation ≈ no gain where stale error ≈ sensor-noise floor"). A variant failing G1 never runs.

## Predictor v2/v3 + gate decision
| N | real | v2 (near 0.5, h64) | v3 (near 0.5, h128) | v3/real |
|---|---|---|---|---|
| 25 | 0.0117 | 0.0144 | 0.0077 | 0.66 |
| 50 | 0.0224 | 0.0200 | 0.0156 | 0.70 |
| 100 | 0.4625 | 0.2371 | 0.2417 | 0.52 |
| 150 | 1.5285 | 0.6205 | 0.6114 | 0.40 |
| 200 | 3.1703 | 0.9989 | 0.9725 | 0.31 |
- v2 FAIL (G1 at N=25). **v3 PASS** G1 ∧ G2 ∧ G3 (G3: 0.2417 ≤ 1.25 × 0.2446). v3 also < real on pairs < 2 m and on all
  pairs at every N. v3 frozen: sha256 9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5 (36 epochs).
- Exploratory artefacts pushed in 5cd4443.
- **amend22 pre-registered** (aneks sha256 ceb6aa4e…, commit 22b0e6b) BEFORE any run; launched 18:45 UTC on the pod.
- amend22 attempt 1 (18:45 UTC): 3 cells in parallel → GPU OOM in 2 of 3 processes at start ("RESOURCE_EXHAUSTED ...
  5.12GiB"; each process holds the per-N Φ/m* space-time fields). Chain killed at 18:49 (its trap pushed a FAILED commit
  82dfabf holding the 11 per-seed metrics already complete: N200 learned seeds 0–8, N200 real_d0.70 seeds 0–1; completed
  seeds are whole files, kept and skipped on restart). Infra fix P=2 (f26bda8; as amend21), relaunched 18:49 UTC.
