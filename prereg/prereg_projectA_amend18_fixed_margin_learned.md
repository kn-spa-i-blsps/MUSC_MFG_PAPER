# Amendment 18 — Fixed-margin sweep on `learned`

**Base:** amend15 cell s=1.05 / amend16 / amend17. **Measurement only**, no controller change: the constant-d_safe
CBF path (cbf.margin_k absent) is the pre-amend17 controller, bit-identical (amend17 k=0 sanity). Frozen, pre-run;
committed with SHA before any run; results via `git -f`. Runs detached on the pod, self-pushes.

**Gap filled:** amend16 Part 1 swept the uniform margin d_safe only on `real` (real_robust); amend17 swept the
adaptive margin on real + learned. The uniform-margin sweep on `learned` was never run.

## 1. Design

- **Fixed:** s = 1.05, gate = 2.309, N = 100, D_COLLISION = 0.30, all other amend15 params. Reuse the s=1.05 Φ/m*
  cache (key f770143b36, on the amend17 pod volume; d_safe is not in the cache key). No solve.
- **Arm:** `learned`, frozen amend11 checkpoint (predictor_weights.npz, sha256 fc4a52a5…).
- **d_safe (constant, not adaptive) ∈ {0.55, 0.70, 0.85, 1.00}. Seeds 0–4.** 20 rollouts.
- **Identity check:** the d_safe = 0.55 cell is exactly the amend15 learned s=1.05 config; its per-seed metrics are
  compared with the committed amend15 metrics (results/density_sweep_amend15_N100/s1.05/learned/metrics_seed*.json).
- **Reference row (reuse, no re-run):** learned @ fixed 0.55 (amend15): safe 0.240 / compl 0.674 / ever 0.606.

## 2. Metrics

Per d_safe, mean + 95% seed-bootstrap CI over seeds 0–4, at c = D_COLLISION = 0.30: safe-success (amend12
definition), completeness, ever_collided (fraction of agents ever with true d < 0.30), p5 worst-step separation,
CBF intervention rate, slack-active rate.

## 3. Predictions

None pre-registered (PI order is a measurement; the result is reported without interpretation).

## 4. Execution

Detached (tmux) on the amend17 RunPod A40 pod k9nrewd0bhycsu; on completion write report.json, `git add -f`, commit,
push, DONE/FAILED marker.
