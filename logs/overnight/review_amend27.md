# REVIEW — amend27: compensation-method ladder + MFG/CBF layer attribution (2026-11-01)

**Run:** amend27_chain_pod.sh, HEAD 01701ef, RunPod A100, result commit 2edfe32. Aneks sha256 d13f98ca… (matches
INDEX). 352/352 rollouts, all `matmul_precision: highest`. CIs: seed-level bootstrap, 10k, 95 %. Seeds 0–9.

## §3 verification
- **Identity gate:** passed, run first. N=100 seed 0 `real` and `split:real/real` are bit-equal to amend22
  real_d0.70 on all 7 IDENTITY_KEYS (results/amend27_ladder_layers/identity.json).
- **Wrong-knob:** cells differ from amend21_cells only in d_safe / run_name / amend27_arm.
- **Smith:** checked separately (below), because its completeness collapse looked like a possible artifact.

## Q1 — delay-free baseline
- **noise_only** (delay removed, sensor noise kept): safe at every N. ever<0.30 = 0.000 with CI [0, 0] at N 25–200.
  Completeness is 1.000, except 0.968 at N=200.
- **fresh** (oracle): safe at N 25–150 (ever 0; completeness 1.000 / 1.000 / 1.000 / 0.998).
- **fresh at N=200:** ever 0, completeness 0.763 [0.658, 0.822]. This is a liveness limit with zero collisions, as
  registered, and it replicates amend24.
- The delay-free two-layer controller is collision-free at every N.

## Damage of the TDMA delay (real, best of d 0.55/0.70)
| N | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| real safe-success | 1.000 | 0.284 | 0.000 | 0.000 | 0.000 |
| real ever<0.30 | 0.000 | 0.716 | 1.000 | 1.000 | 1.000 |
| noise_only safe-success | 1.000 | 1.000 | 1.000 | 1.000 | 0.968 |

## Q2 — method ranking (safe-success; each comparator at its better margin)
| N | real | cv | smith | **learned v3** | noise_only (oracle) |
|---|---|---|---|---|---|
| 25 | 1.000 | 1.000 | 0.908 | **1.000** | 1.000 |
| 50 | 0.284 | 0.000 | 0.068 | **0.992** | 1.000 |
| 100 | 0.000 | 0.000 | 0.006 | **0.880** | 1.000 |
| 150 | 0.000 | 0.000 | 0.002 | **0.479** | 1.000 |
| 200 | 0.000 | 0.000 | 0.001 | **0.137** | 0.968 |

- **Learned minus best comparator,** CI excluding 0 vs every comparator: N=50 +0.708 [0.638, 0.780] (vs real);
  N=100 +0.874 [0.842, 0.902] (vs smith); N=150 +0.477 [0.443, 0.509]; N=200 +0.137 [0.120, 0.154]. N=25 is a tie
  with real and cv (all 1.000); vs smith it is +0.092 [0.048, 0.136].
  **"NN best" holds at N = 50, 100, 150, 200 (registered reading).**
- **safe_N:** real 25, cv 25, smith none, **learned 50**, fresh 150, noise_only 200.
- **Recovered fraction of the (noise_only − real) safe-success gap by the NN:** 0.99 / 0.88 / 0.48 / 0.14 at N = 50 /
  100 / 150 / 200.
- **Cost:** learned completeness at N=200 is 0.599, vs 0.793 for real and 0.968 for noise_only.

## Q3 — layer attribution (d0.70; "none" = real, "both" = learned / noise_only)
| N | MFG-only learned | CBF-only learned | both learned | MFG-only oracle | CBF-only oracle | both oracle |
|---|---|---|---|---|---|---|
| 100 | 0.000 | 0.918 | 0.880 | 0.000 | 1.000 | 1.000 |
| 150 | 0.000 | 0.503 | 0.479 | 0.000 | 1.000 | 1.000 |
| 200 | 0.000 | 0.098 | 0.137 | 0.000 | 0.965 | 0.968 |

- **MFG layer:** compensation fed only to it changes safe-success by exactly 0.000 [0, 0] vs none, at every N, for
  both the NN and the **oracle**. The MFG layer does not carry the safety recovery.
- **CBF layer:** recovers 104 % / 105 % / 72 % (NN) and 100 % / 100 % / 100 % (oracle) of the both-layers gain.
- **N=100:** CBF-only NN is above both (+0.038 [0.006, 0.074]). Feeding the NN estimate to the MFG layer as well
  slightly lowers safe-success.
- **N=200:** CBF-only NN is below both (−0.039 [−0.059, −0.020]; ever +0.049 [0.026, 0.072]). Each single-layer split is
  below both, which is "both layers needed" by the registered reading. The MFG-layer share is small (0.039 of 0.137).
- **Completeness of MFG-only splits vs real:** 0.998 vs 0.994 (N=100), 0.909 vs 0.924 (150), 0.776 vs 0.793 (200),
  with overlapping CIs. Oracle MFG-only: 1.000 / 0.938 / 0.811.

## Smith: the collapse is real, not a code artifact
- **Symptom:** smith completeness 0.60 (N=25, d0.70), 0.14–0.33 (N=50) and 0.00–0.05 at N ≥ 100, while every other
  arm completes. Agents move fast (3.7 m/s mean at N=100) but only 9 % cross the wall. Closed-loop near-pair x̂ MSE is
  5.4 m² at N=100, vs cv 1.2.
- **Checks:**
  - The field sign matches the controller: `phi_grad` stores −∇Φ, used as velocity in both.
  - The time indexing matches: same dt and nearest index.
- **Open-loop diagnostic** (real-arm history N=100 seed 100, CPU, logs not committed):
  - At t=10 s (crossing phase), near-pair MSE is real 1.03 / cv 1.55 / **smith 4.60 m²**. Smith's predicted shift is
    1.38 m, vs a true 0.99 m motion. The cosine between the predicted shift and the true motion is 0.13.
  - At t ≥ 30 s (in the goal disc), smith ≈ real (0.64 / 0.41 / 0.33 vs 0.82 / 0.45 / 0.37).
- **Reading:** the routing-field model predicts neighbours moving at the clipped free-flow velocity toward the gate,
  while the real crowd is queueing, congestion-limited and CBF-limited. The model mismatch is largest exactly where the
  CBF needs accuracy. It places phantom neighbours in the gate and the swarm locks up.
- **Classification:** a property of this comparator (a routing-only model with no congestion), not a bug. This is
  **not** a statement about the thesis.

## Classification of non-positive results
| result | class | why / action |
|---|---|---|
| MFG-layer compensation has zero safety effect, even with the oracle | THESIS-RELEVANT | Holds for a perfect estimate too, so no predictor fix can change it. It is structural: the density term is a kernel-smoothed macro quantity. The MFG-layer benefit must be sought in macro fidelity (amend28 Q5, running), not collisions. |
| NN recovers only 14 % of the gap at N=200; completeness 0.60 | PLUMBING candidate (brief §C) | The predictor's error grows with T_frame (G-pred: mean e 0.22 m at N=200). Phase B σ-CBF (amend31) targets this. A higher-capacity / longer-window v5 is the next §C step if amend31 does not close it. |
| cv worse than no compensation at N=50 (0.000 vs 0.284) | comparator property | The finite-difference velocity of two noisy broadcasts, × age, amplifies noise. |
| Smith collapse | comparator property | Above. |

**Fairness consequence:** both classical comparators are weak for identifiable reasons (noise amplification; a
congestion-blind model). A claim "NN beats classical compensation" needs a noise-optimal classical estimator on the
SAME 4-broadcast input the NN sees: a constant-velocity least-squares / Kalman fit. That baseline is registered next
(amend32) and must be beaten before the "NN best" claim is stated without qualification.
