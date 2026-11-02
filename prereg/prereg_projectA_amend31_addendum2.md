# Amendment 31 — addendum 2 (2026-11-02, before any amend31 data exists)

Base: aneks amend31 (12678c58…) + addendum 1. Reason: there are no A100s in the RunPod catalog. The PI chose to poll for
GPUs (2026-11-02) in this order: the old A100 pod mc4847t30028lq, then a COMMUNITY RTX PRO 6000 (96 GB, ≤ $1.69/h), then
a new A100-SXM4-80GB. The first available one runs the queue.

## 1. Identity gate off the A100
- On an A100-SXM4-80GB the registered identity is unchanged: k=0 real seed 0 is bit-equal to amend22 real_d0.70 seed 0.
- On any other GPU type, the gating reference is the fixed-0.70 real seed 0 rollout run in the same chain on the same
  GPU (cell identity/N100_fixed0.70_real_ref.yaml; `identity --ref local`). The comparison uses the same IDENTITY_KEYS,
  bit-equal.
- The comparison with amend22 is still computed and reported (vs_amend22_reported). It does not gate, because the
  matmul kernels differ across GPU types.
- The identity tests the controller-code path (σ-CBF at k=0 ≡ fixed 0.70). That is a same-hardware property.

## 2. Hardware in the comparisons
- All G-win comparisons are within amend31: σ arms vs the fixed 0.70/0.85/1.00/1.20 frontier, all re-run in this chain
  on one GPU. So no comparison mixes hardware.
- Any comparison of amend31 numbers with earlier A100 amendments (amend22/23/27) is labelled cross-hardware, descriptive
  only.
- The GPU name is logged by the chain.
