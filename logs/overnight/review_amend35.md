# REVIEW — amend35: capped σ-CBF G-win (one shot, REGISTERED)

**Purpose:** decide whether the amend31 σ-CBF failure was caused by the unbounded margin. Same registered G-win, run
once, with d_safe_ij and σ_c capped at d_max = 1.5 m. That was the only change.

**Records**
- Aneks prereg/prereg_projectA_amend35_sigma_cap.md (sha d3b3b46a…), committed c18ba2b before any run.
- Infra notes in INDEX: 170b8ed (off-A100 path), 3065383 (NO_PUSH).
- Result commits: 9fec57d (sanity), ef7872b (N=150), e7ace94 (N=200), 14049bb (report), 66a801d (chain end).
- Numbers: results/amend35_sigma_cap/report.json, logs/amend35_pod/report.log.

## Infra (PLUMBING, no effect on the verdict)

**Pod**
- The A100 pod sbk37l6nvrg36q could not start: host GPUs taken, and no A100 or H100 in stock.
- New pod out2eyja2j0lrk: RTX PRO 6000 Blackwell Server 96 GB, US-NC-2, $2.49/h, jax 0.12.0 cuda12.
- Uptime 69 min, then stopped (EXITED).

**Fixed arms re-run.** As the aneks states for a non-A100 GPU, the four fixed arms were re-run on the same GPU, so
every comparison is within one GPU. That is 160 rollouts in total, all 10 seeds per cell, all exit 0.

**Push path.** Adding a GitHub key to the pod was refused by the local permission policy. Instead, the repo went to the
pod as a git bundle, the pod committed locally (NO_PUSH), and the workstation pulled each pod commit (ff-only) and
pushed it.

**Precomputed MFG solutions.** Recomputed on this GPU with identical cache keys (N150 2f21768897, N200 6490094688),
~470 s each.

## Sanity

- **S1 (no cap vs amend31 seed 0): FAILS, as expected across hardware** (A100 → RTX PRO 6000).
  - Example: collision pair-steps 1296 vs 1716.
  - Mean d_safe 2.42 vs 9.36 m: the uncapped blow-up is chaotic.
- **S2 (gating fallback): PASS.** On this pod, both with XLA autotune 0, the amend31 code (git worktree 459d624) and
  the new code with no cap are bit-equal on pos and all keys.
  - **The cap is the only code change.**

## Did the cap do its job?

Yes.
- Realised max d_safe is 1.50 m in every σ arm. In amend31 it ran up to 10.3 m and beyond, with mean d_safe per step
  up to 2.8e5 m.
- Mean d_safe for the PRIMARY arm, sigma_bin α=0.05: 0.725 m at N=150 (amend31: 14.7) and 0.959 m at N=200
  (amend31: 64.6).

## G-win (registered)

- PRIMARY = sigma_bin_a0.05.
- Comparator = best fixed by mean safe-success on this GPU: fixed0.85 at N=150, fixed1.00 at N=200.
- Seed-level bootstrap 95 % CI, 10 seeds.

| N | PRIMARY safe / ever / compl | best fixed safe / ever / compl | Δ safe [CI95] | Δ compl [CI95] | per N |
|---|---|---|---|---|---|
| 150 | 0.527 / 0.404 / 0.856 | 0.487 / 0.206 / 0.575 (0.85) | **+0.039 [+0.005, +0.075]** | +0.281 [+0.271, +0.290] | PASS |
| 200 | 0.001 / 1.000 / 0.548 | 0.189 / 0.459 / 0.330 (1.00) | **−0.189 [−0.200, −0.177]** | +0.218 [+0.207, +0.229] | FAIL |

**Verdict: NOT_MET.** PASS required both N, and N=200 fails.

**Secondary arms (not decisive):**
- At N=150, α 0.10 / 0.05 / 0.01 give Δ safe +0.023 / +0.041 / −0.022. All of their CIs include 0.
- At N=200, every σ arm has safe-success ≤ 0.006, with ever-collided 0.994–1.000.

**Descriptive, not registered, PRIMARY vs fixed0.70:**
- N=150: safe +0.049 [+0.017, +0.081] at equal completeness (−0.001 [−0.011, +0.007]); ever-collided −0.075.
  So at N=150 the capped σ-CBF is to the upper right of the fixed frontier.
- N=200: safe −0.140, completeness −0.042. Under it on both.

**Closed-loop coverage of the σ margin:**
- At N=200 it is 0.77 on pairs < 2 m and 0.69 on rows where the margin is binding (binding on 75 % of rows), against
  0.95 offline.
- At N=150: 0.94 on pairs < 2 m, 0.89 on binding rows.
- The G-cal guarantee stays broken in closed loop even with the cap.

## Classification

**THESIS-CRITICAL, by the reading registered before the run: PRIMARY is still under the fixed frontier at N=200, so
Phase B fails as a whole.**
- The blow-up bug is ruled out as the cause of the N=200 failure. With d_safe ≤ 1.5 m and S2 bit-equal, every drone
  still collides at N=200 in every σ arm. With fixed margins, about half do.

**Fact for the PI (not a reinterpretation of the verdict).**
- At N=150 the bug was hiding a real, small effect: +0.04 to +0.05 safe-success over the fixed frontier, at the same
  completeness as fixed0.70.
- It does not survive at N=200, which is the density the method was meant for.
- So σ-CBF is at best a heuristic that helps at moderate density and fails at high density. The registered test does
  not license it.

**Chain:**
- "Compensation restores": **does not hold** for a predictor-driven CBF margin at N=200.
- "Each layer earns its place": the σ margin layer does not.

**STOP.** The queue (28 / 32 / 29 / 30) stays blocked until the PI decides.
