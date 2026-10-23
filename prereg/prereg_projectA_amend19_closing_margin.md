# Amendment 19 — Closure-gated adaptive CBF margin (method v2a)

**Base:** amend15 s=1.05 / amend17 / amend18. **CONTROLLER CHANGE.** Frozen, pre-run; committed with SHA before any
run; results via `git -f`. Runs detached on the pod, self-pushes.

## 1. Method

Per pair, per timestep (cbf.margin_mode = "closing"):

**d_safe_ij = max(0.55, D_COLLISION + k · max(0, v_closing_ij) · h),  h = 0.3 s,  D_COLLISION = 0.30, no cap.**

- **v_closing_ij = (v_i − v̂_j) · e_ij**, positive part only (approach). e_ij = unit vector from i to i's estimate
  x̂_ij of j (the same x̂ the CBF row uses).
- **v_i** = i's own velocity over the last step, (p_i(t) − p_i(t − dt))/dt (0 at t = 0).
- **v̂_j** = finite difference of i's last two HEARD absolute positions of j over one TDMA frame:
  ((p_i(t_then) + r̂_now) − (p_i(t_prev) + r̂_prev)) / T_frame, where r̂ are the two most recent noisy relative
  observations (the cv estimator's inputs, same PRNG draws) and p_i(·) is i's own known position at those instants.
  Same estimator for every arm (the learned MLP outputs no velocity).
- Floor 0.55 = the fixed amend15 margin. Everything else in the controller unchanged.

## 2. Sanity (before the commit)

k = 0 ⇒ d_safe = max(0.55, 0.30) = 0.55 on every row ⇒ must be bit-identical to fixed d_safe 0.55, seed 0, learned
(+ real). Both sanity runs use `XLA_FLAGS=--xla_gpu_autotune_level=0`, because learned is not run-to-run deterministic
on the GPU with autotuning on (amend18 post-hoc), so a mismatch there could not be attributed to the code.

**Observed before this commit:** real — 19/19 logged arrays bit-identical. learned — d_safe = 0.55 on 100% of rows,
but NOT bit-identical: first difference 1 ulp (1.2e-7) in u_nom at step 1, upstream of the CBF, then chaotic growth
(seed-0 collision pair-steps 2000 vs 1179). Neither precision=highest nor an extra optimization_barrier removes it
(logs/amend19_pod/sanity_diagnostic.log). Taken as float-level compilation noise in the learned perception path, not a
code difference; the run proceeds on the code the sanity ran on.

## 3. Run

- **Fixed:** s = 1.05, gate = 2.309, N = 100, D_COLLISION = 0.30, all other amend15 params; learned = frozen amend11
  checkpoint (sha256 fc4a52a5…). s=1.05 Φ/m* cache, key f770143b36 (re-solved once on the new pod 8thhi6wqzzclov with
  the identical key, before the commit — pod k9nrewd0bhycsu could not get its GPU back; the run does no solve).
- **Arm:** learned. **k ∈ {0.5, 1.0, 2.0}. Seeds 0–4.** 15 rollouts, default XLA autotuning (same setting as the
  amend18 reference rows).
- **Metrics:** as amend18 (safe-success, completeness, ever_collided, p5 worst sep, CBF intervention, slack) + realized
  d_safe mean/p50/p95 + v_closing mean/p50/p95 and histogram (0.05 m/s bins) over the valid CBF neighbour rows.
- **Reference rows (reuse, no re-run):** learned fixed d_safe 0.55 → safe 0.226, 0.70 → safe 0.290 (amend18).

## 4. Predictions

None pre-registered.
