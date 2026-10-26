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

## amend22 DONE (c24e62f) → review_amend22.md
- Verdict (details in the review): CONFIRMED real staleness barrier (safe_N 25 at d 0.55 and 0.70), compensation gap
  at N=50/100 vs margin-matched real 0.70 (+0.936 / +0.880 safe-success, CIs exclude 0), OOD fix; larger margin does NOT
  help real (0.70 worse than 0.55). SUSPECT mechanism of learned degradation at N=150/200 (not explained by mean x̂
  error; packing below d_safe there). QUARANTINED: closed-loop x̂ MSE as a cross-arm comparison (steps without near pairs
  enter as 0 → biased low for separated swarms).
- Next (pre-registered now, before any run): identity gate for a logging-only n_near change → amend23 (Phase 2 margin
  re-tune, N=100/150) → amend24 (noise_only / fresh mechanism control), one night chain on the pod.

## Night chain (pod times UTC)
- 19:38 identity gate for the logging-only n_near change: real seed 0 at N=25 and N=100 equal amend21 on 8 metrics
  (all_equal = True; logs/overnight/identity_nnear/identity.json). New pair-weighted near MSE at N=25 real = 0.01166
  vs the old per-step-mean metric 0.01193.
- First launch ran no amendment (log dirs absent on the pod); fixed (2b32112), relaunched 19:40: amend23 started.
- Drafted while waiting (not yet pre-registered; margin D to be set from amend23's rule): amend25 gate sweep
  {1.5, 2.309, 3.5, 5.0, 7.0} at N=100 (new solves per gate) and amend26 t_slot sweep {2.5, 5, 7.5, 10 ms} at N=100
  (same cache), arms real 0.55 / real D / learned v3 D, each with an open-loop estimator check per sweep value and an
  identity check of the baseline cell vs amend21.

## amend23 DONE (118e4bd) → review_amend23.md
- CONFIRMED operating margin 0.70 at N=100 (pre-registered rule, resolved): safe-success 0.892 vs 0.675 / 0.747 / 0.686
  for 0.55 / 0.85 / 1.00; larger margins buy ever<0.30 (0.020 / 0.010) only by freezing (completeness 0.76 / 0.69).
- N=150: no margin both safe and complete; 0.70 vs 0.85 tie on safe-success; rule keeps 0.70 (not resolved).
- Reproducibility: amend22 learned d0.70 N=150 repeated bit-identically; N=100 within CI (0.892 vs 0.880).
- Pre-registered now (before any run): amend25 gate sweep (Phase 3) + amend26 t_slot sweep (Phase 4), d_safe 0.70,
  queued on the pod behind amend24 (logs/overnight/phase34_chain_pod.sh).

## amend24 DONE (eacfb15) → review_amend24.md
- CONFIRMED (headline mechanism): with the TDMA delay removed (noise_only: same controller, margin, geometry, packing,
  sensor noise), ever<0.30 = 0.000 at EVERY N ∈ {25..200}, completeness ≥ 0.968. The density hypothesis for learned's
  N ≥ 150 degradation is FALSIFIED; the residual is predictor error (pre-registered branch). Headroom over v3:
  +0.12 / +0.52 / +0.83 safe-success at N = 100 / 150 / 200.
- Note: fresh (no noise) at N=200 has completeness 0.763 vs noise_only 0.968 (safety equal); unexplained, not a claim.

## Predictor v4 (exploratory; review_amend24 fix 1) — gate fixed BEFORE data
Idea: v1–v3 were trained only on REAL-arm histories; the closed loop runs under the LEARNED arm (calmer, sparser
swarm) → distribution shift. v4 adds closed-loop learned-v3 d0.70 histories (seeds 100–111 per N; DAgger-style).
Variants: v4a (real+learned, near 0.5, h128), v4b (same, h256, 200k windows/seed/N).
Gate (in addition to G1–G3 on real test histories):
- G4: on LEARNED closed-loop test histories (seeds 110–111), near-pair MSE AND p99 below v3's at every N ≥ 100, and
  not worse than v3 by > 10 % at N ≤ 50.
- Selection among passing variants: lowest mean p99 (learned histories) over N = 150, 200.
- If none passes: v3 stays; reported as a negative.
Then (if a v4 passes) amend27 = learned v4 d0.70 crossover at all N (pre-registered), compared with amend22/24.

## INCIDENT 21:29–21:46 UTC — volume quota (150 GB) reached → amend25/26 writes failed
- 9 Φ/m* caches × 13 GB = 116 GB + results filled the 150 GB volume at ~21:29. From then: rollouts died silently
  (empty logs), one metrics json written empty (amend25 g2.309 real_d0.55 seed 2), the chain log was truncated, the
  FAILED commit did not happen. amend25 reached 35/150 metrics, amend26 started 21:44 and produced empty cells.
- Fix: deleted the N=25/50/150/200 caches (regenerable with identical keys) → 90 GB; killed both chains; moved the
  failed-attempt logs to /workspace/failed_attempts/ (copied into the repo with the result); every existing per-seed
  output is validated (json + full npz read) before reuse, corrupt ones deleted and re-run. New tail chain
  (logs/overnight/tail_chain_pod.sh): amend25 → amend26 → delete gate caches, re-solve N=25/50/150/200 → v4.
- No pre-registered design changed. No reported number came from the quota window (corrupt files are re-run).

## STOP ~22:15–22:40 UTC — RunPod account balance exhausted; pod biiz4t7r00oef8 EXITED (not stopped by me)
- Last pod state seen (22:11 UTC): amend25 150/150 rollouts done (34 validated pre-incident + 116 new), open-loop check
  (pre-report) running; disk 96 GB. Nothing of amend25 after the pre-incident partials was pushed (the result commit
  happens at the end of the chain).
- `pod-action start` → 402 "Your account balance is too low to rent a pod". No compute possible until funds are added.
- Consequences: amend25 (gate sweep) = rollouts complete ON THE POD VOLUME, report/open-loop check/review NOT done →
  no reported numbers. amend26 (t_slot sweep) = 8/120 metrics, not run. Predictor v4 = not run. All three remain
  pre-registered / gated with no results; nothing is claimed from them.
- To resume: add funds → start biiz4t7r00oef8 (volume keeps results/amend25_gate_sweep, caches N100 ×5) → re-copy
  /workspace/.ssh key → `bash logs/overnight/tail_chain_pod.sh` (validates existing outputs, skips done seeds,
  continues amend25 report → amend26 → re-solves → v4).

## 2026-10-26 resume (PI added funds)
- biiz4t7r00oef8 cannot start ("not enough free GPUs on the host"); GPU count of a stopped pod cannot be changed (no
  CPU-only start for data retrieval). Its volume (unreported amend25 rollouts) is left untouched.
- NEW pod xby8jnv4zqwvn2 (A100-SXM4-80GB, SECURE, $1.79/h, 250 GB volume to avoid the 150 GB quota). amend25 and amend26
  re-run from scratch under their unchanged pre-registrations (results of the old volume are not used).
- Real-arm datagen (seeds 100–111, needed by v4) existed only on the old volume; regenerated (real path is
  bit-deterministic) and checked: v3 open-loop on the regenerated histories must equal openloop_v3.json exactly.
- Chain: logs/overnight/resume_chain_pod.sh.
- amend25 DONE (1ce2b21) on pod xby8jnv4zqwvn2 → review_amend25.md: CONFIRMED learned advantage at every gate
  (+0.87…+0.93); gate width a weak lever (wrong-knob finding consistent with amend24); Q3 boundary non-monotone within
  noise (3.5 passes, 5.0 fails, 7.0 passes). Identity real vs amend21 PASS; learned gate 2.309 reproduces amend23
  exactly on a different pod.
- amend26 DONE (ac32b09) → review_amend26.md: CONFIRMED pure-staleness effect and the staleness/density decomposition
  (learned collisions track T_frame exactly; completeness loss at large N is density).

## Predictor v4 — gate result (pushed ee33aee): FAIL → v3 stays (negative, reported)
Open-loop near-pair x̂ MSE / p99 [m²], test seeds 110–111. "real hist." = real-arm histories; "learned hist." =
closed-loop learned-v3 histories (the evaluation distribution).

| N | real hist.: real | v3 | v4a | v4b | learned hist.: real | noise_only | v3 | v4a | v4b |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 0.0117 | 0.0077 | 0.0087 | 0.0081 | 0.0114/0.069 | 0.0108/0.066 | 0.0077/0.043 | 0.0081/0.046 | 0.0073/0.045 |
| 50 | 0.0224 | 0.0156 | 0.0157 | 0.0149 | 0.0130/0.080 | 0.0108/0.068 | 0.0104/0.064 | 0.0102/0.063 | 0.0093/0.067 |
| 100 | 0.4625 | 0.2417 | 0.2456 | 0.2313 | 0.0197/0.126 | 0.0107/0.067 | 0.0224/0.128 | 0.0196/0.114 | 0.0156/0.112 |
| 150 | 1.5285 | 0.6114 | 0.6357 | 0.6129 | 0.0317/0.247 | 0.0108/0.067 | 0.0305/0.187 | 0.0256/0.185 | 0.0222/0.170 |
| 200 | 3.1703 | 0.9725 | 1.0417 | 1.0253 | 0.0696/0.669 | 0.0108/0.067 | 0.0490/0.351 | 0.0437/0.399 | 0.0400/0.357 |

- G1–G3 (real histories): v4a and v4b both PASS (ratios ≤ 0.70, N=100 within 1.25× of the old specialist).
- G4 (learned histories, MSE AND p99 below v3 at every N ≥ 100; ≤ +10 % at N ≤ 50):
  - v4a: MSE lower at N = 100 / 150 / 200, p99 lower at N = 100 / 150, but p99 at N=200 is 0.399 > 0.351. **FAIL.**
  - v4b: MSE lower at all N ≥ 100, p99 lower at N = 100 / 150, but p99 at N=200 is 0.357 > 0.351 (+1.7 %). **FAIL.**
  - N ≤ 50: both within +10 %.
- By the pre-declared rule **v3 stays; amend27 is not run**. The threshold is not moved after seeing the data, even
  though v4b misses by 1.7 %.
- Side finding (fact, for the mechanism panel): on the learned closed-loop distribution the stale error is small
  (real 0.0197 at N=100 vs 0.4625 on real-arm histories), and v3's mean error there is not below the stale estimate
  at N = 100 / 150 (0.0224 vs 0.0197; 0.0305 vs 0.0317). The calm learned swarm keeps stale error low. v3's benefit
  appears at the regime level: it prevents the transition into the high-error, crowded regime real falls into. In the
  calm regime its per-step advantage is small, and v3 is out of its training distribution there (the motivation for v4,
  which lowers the mean but not the N=200 tail).
