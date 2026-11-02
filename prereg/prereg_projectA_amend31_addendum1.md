# Amendment 31 — addendum 1 (PI decision 2026-11-02, before any amend31 data exists)

Base aneks: prereg_projectA_amend31_sigma_cbf.md (sha256 12678c58087e2e4c2a5a342f715b76237ff336d7e9b5cc9e4249b04ebd2cc677),
unchanged. This addendum only adds; where it differs, it takes precedence. No amend31 output (σ head, calibration,
rollout) existed when it was written. The pod has been EXITED since 2026-11-01 03:00 UTC, and amend31 was last in the queue.

## 1. Run order
amend31 runs FIRST: amend31 → 28 → 32 → 29 → 30 (logs/overnight/queue_B_pod.sh). PI's reason: amend31 is the result
for thesis B. The rest only makes sense if B works, and the balance has stopped mid-queue three times.
amend31 does not read any output of amend28/29/30/32.

## 2. Stage 0 — inputs (infrastructure, no design change)
The pod volume may be lost. If so, the 5 per-N Φ/m* caches (amend21 geometry, unchanged keys) and the datagenL
histories (learned v3 d0.70, seeds 100–111, N 25–200) are regenerated before the σ head. Regeneration uses the unchanged
code and cells of the G-pred chain (commit 01701ef). A cache hit only loads.
- Unreadable datagenL npz files are deleted and regenerated.
- The learned closed loop is not bit-deterministic on the GPU (G-pred integrity: 8e-8 rel.), so a regenerated datagenL
  can differ from the G-pred copy at that level. The σ head is trained fresh in amend31 either way.

## 3. Staged chain, each stage pushed before the next (logs/overnight/push_lib.sh)
| stage | contents |
|---|---|
| 0 | inputs |
| 1 | σ head + G-cal (calibration.json, reliability.png) |
| 2 | identity gate |
| 3 | per-N rollouts, then the report |

G-cal is computed and reported in full whatever G-win does.

## 4. G-win — completeness condition added (PI: "przy NIE gorszej completeness")
- Comparator and safe-success criterion as in aneks §5: the best fixed margin per N by mean safe-success, and the CI95
  lower bound of the safe-success difference > 0.
- **Added — completeness non-inferiority** vs the same comparator: the CI95 lower bound of (primary − comparator)
  completeness must be ≥ −0.02. Seed-level bootstrap, as for safe-success. The 0.02 margin is absolute, as a fraction of
  agents.
- **Verdict (addendum 1, the one stated as G-win):**
  - PASS = both conditions at N = 150 AND N = 200;
  - PARTIAL = both at exactly one of them;
  - NOT_MET otherwise.
- The aneks §5 verdict (safe-success only) is still computed and reported next to it, labelled "registered v1".
- Completeness differences vs every fixed margin are reported as before.
- k = 1 as registered. The per-arm table is per α ∈ {0.10, 0.05, 0.01} plus binned. Other k are §C follow-ups, not part
  of this run.

## 5. G-cal reporting additions (no change to the gate)
calibration.json gains, per N, α and calibration (per-N and (N, σ-quintile)), per σ-decile:
- mean σ_c;
- the realised (1−α) quantile of e;
- n.

reliability.png plots, per N, (predicted σ_c vs realised quantile) and (per-decile coverage vs 1−α, with the 0.85 line).
The G-cal criteria (aneks §2) and the verdict logic are unchanged.
