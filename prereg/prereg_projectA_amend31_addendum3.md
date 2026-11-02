# Amendment 31 — addendum 3 (2026-11-02): identity-gate failure diagnosed and fixed (PLUMBING), before any G-win data

## What happened
- Chain run 1 on the new A100 pod sbk37l6nvrg36q (HEAD f90bea6): stage 0 and stage 1 completed and were pushed
  (4e18618, 1349c30; G-cal PASS_AFTER_FIX). The identity gate then FAILED: k=0 real seed 0 was not bit-equal to amend22
  real_d0.70 seed 0 (collision_pair_steps 46844 vs 46961). The chain aborted as registered (e662d89). No σ-CBF or
  fixed-frontier rollout ran.
- The queue moved on to amend28. I stopped it during its first macro solve, before any rollout (5ff7869, logs only).

## Diagnosis, on the same pod
1. The plain fixed-0.70 real seed 0 is bit-equal to amend22 on all IDENTITY_KEYS, so the pod environment reproduces
   the A100 history. (Installed: jax/jaxlib 0.12.0, cublas 12.9.3.13, cudnn 9.27.0.42.)
2. The k=0 σ path is run-to-run deterministic: two runs are bit-equal.
3. The first divergence between k=0 and fixed 0.70 is at **step 0, in the CBF output u (8 ulp)**. row_violation is 0 vs
   2.4e-7, and u_nom is identical until step 7.
4. Cause: control/cbf.build_rows computes h = |x_i − x̂_j|² − d_safe². With a constant d_safe the square is a Python
   float64 0.7² cast to f32, giving 0.49f. With the per-pair array (σ mode) it is the f32 product 0.7f·0.7f =
   0.48999998f, 1 ulp lower. The chaotic closed loop amplifies that ulp.
- The margin logic max(0.70, 0.30 + k·σ) was correct: d_safe = 0.70 on every row at k=0 (p95 0.7).

## Fix (controller code, σ mode only; every other mode and the constant path are untouched)
- In margin_mode "sigma", rows whose d_safe equals the floor use the constant path's square: f32(margin_floor²) computed
  in Python. Rows above the floor are unchanged.
- Verified on the pod with the patched file: k=0 real seed 0 is **bit-equal to amend22 on all IDENTITY_KEYS**
  (all_equal true, collisions 46961).
- Classification: PLUMBING. A floating-point association artifact of the registered gate, not a controller-logic
  error. Fixed, not waived; the gate stays bit-equal vs amend22.

## Re-run rules
- **Stage 1 is reused, not re-trained:** the committed σ head (1349c30), calibration.json and G-cal verdict. The fix
  touches only the controller. Re-training on the GPU would give a second, non-bit-identical draw of the head and of
  G-cal, which would be selection.
- **Stage 0:** inputs are on the volume. A cache hit only loads; datagenL 60/60 is skipped.
- **Then:** identity (registered criterion vs amend22) → per-N rollouts → report, as registered (aneks + addenda 1–2).
- The queue then continues with amend28 → 32 → 29 → 30. amend28 restarts from its solves; it had no rollouts.
