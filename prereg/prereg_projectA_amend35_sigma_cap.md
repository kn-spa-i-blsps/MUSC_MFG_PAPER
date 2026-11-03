# Amendment 35 — capped σ-CBF G-win (last attempt for Phase B)

**Purpose (one sentence):** decide whether the amend31 σ-CBF failure was caused by the unbounded margin (d_safe up to
2.8e5 m), by re-running the registered G-win once with d_safe capped, so that the predictor-driven margin is either
kept as a heuristic or dropped from the thesis.

**Status:** REGISTERED, one shot. Written and committed before any rollout. PI request 2026-11-03, where it was called
"amend33"; that number is already taken by the CPU baseline, so this is amend35.

## The one change

- σ mode only: σ_c ← min(σ_c, (d_max − 0.30)/k) and d_safe_ij ← min(max(0.70, 0.30 + k·σ_c), d_max).
  - d_max = 1.5 m, k = 1, so σ_c ≤ 1.2 m.
  - Code: `CBFParams.margin_cap` (control/cbf.py) and two Python-level `if prm.margin_cap is not None` branches in
    env/rollout.py.
  - With margin_cap absent (None) the traced graph is the amend31 graph.
- **Not changed (separate questions, not in this attempt):**
  - no ḋ_safe term;
  - no symmetrisation of σ_ij / σ_ji;
  - no retraining;
  - no recalibration;
  - no change to α, the floor 0.70, k, the geometry, the seeds or the precision.

## Reuse

- **Frozen σ head and conformal calibration:** results/phaseB/sigma_v3/sigma_weights.npz (sha 26aff908862c…) and
  calibration.json, exactly as in the amend31 cells. The cells are built by amend31's `cell_cfg`, plus
  `cbf.margin_cap: 1.5` and a new run_name.
- **Predictor:** learned v3, precision=highest, seeds 0–9, amend21 crossover geometry, N = 150 and 200.
- **Fixed-margin comparator:** the registered amend31 rows fixed 0.70 / 0.85 / 1.00 / 1.20, N 150 and 200, seeds
  0–9.
  - The cap does not touch the fixed path (margin_k is None there), so re-running them would change nothing but GPU
    run-to-run noise.
  - Reusing them makes the cap the only variable.
  - Condition: the pod is A100-SXM4-80GB, as in amend31. If it is not, the fixed arms are re-run on the same GPU
    before the report, and the report says so.

## Arms (N = 150, 200; 10 seeds each; 80 new rollouts)

- σ-CBF capped: α ∈ {0.10, 0.05, 0.01} (per-N conformal q) and sigma_bin_a0.05 (Mondrian per (N, σ-quintile)).
- **PRIMARY, declared now, before any data:** sigma_bin_a0.05 vs the best fixed arm.
  - Best fixed = the fixed arm with the highest mean safe-success at that N (amend31 rule; fixed0.85 at both N in
    amend31).
- The other σ variants are SECONDARY: reported, not decisive.

## G-win (PASS, all required)

At N = 150 AND N = 200, PRIMARY − best fixed must satisfy both:
- safe-success: lower bound of the 95 % CI > 0;
- completeness: lower bound of the 95 % CI ≥ −0.02 ("not worse", the amend31 addendum 1 margin).

CI: seed-level bootstrap, 10k, the amend31 `diff_ci`.

Otherwise the verdict is NOT_MET. There is no PARTIAL verdict.

## Sanity (gating, before the capped rollouts)

**S1 (PI's check).** New code with no cap, the amend31 cell N150 sigma_bin_a0.05, seed 0, default settings as in
amend31.
- It must reproduce the amend31 seed-0 rollout bit-identically: pos arrays (if the amend31 npz is on the volume) and
  the 7 IDENTITY_KEYS plus d_safe_mean/p95, σ coverage, σ mean and binding fraction.

**S2 (fallback, used only if S1 fails).** The learned arm is known to be run-to-run nondeterministic on the GPU with
default XLA autotuning (amend18 note; amend31 addendum 4 gate g2). So S1 can fail with no code change at all.
- If S1 fails, the same cell and seed are run twice on this pod, both with `XLA_FLAGS=--xla_gpu_autotune_level=0`:
  - with the amend31 code (git worktree at 459d624, amend31 run 2 HEAD);
  - with the new code and no cap.
- They must be bit-equal (pos + keys). That shows the cap is the only code change.
- S1's differing keys and first divergence step are reported.

If S1 and S2 both fail, the chain aborts before any capped rollout and reports.

## One shot

- This is the registered test: one result, one verdict.
- No repeats until it passes, no threshold moved after the numbers, no extra seeds, no other cap value.
- Infra failures (pod stop, crash) are resumed by re-running missing seeds only.

## Reading (review_amend35.md)

- **PRIMARY still on or under the fixed frontier (NOT_MET):** Phase B is dead as a whole. THESIS-CRITICAL, confirmed
  cleanly with the blow-up bug ruled out.
- **PRIMARY wins (PASS):** the bug was hiding a real effect. σ-CBF lives as a heuristic without a guarantee. The G-cal
  guarantee stays dead regardless, because closed-loop coverage was 0.70 at N=200 in amend31.
- Also reported (descriptive): realised max d_safe (must be ≤ 1.5 m), closed-loop coverage and the secondary arms.

## Execution

- Pod A100-SXM4-80GB sbk37l6nvrg36q.
- Chain logs/amend35_pod/chain.sh, detached (tmux), pushed after the sanity, after each N and after the report.
- Then STOP: the rest of the queue stays blocked until the PI decides.
