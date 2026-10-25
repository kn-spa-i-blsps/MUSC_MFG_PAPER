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
