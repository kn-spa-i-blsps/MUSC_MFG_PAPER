# Amendment 20 — Closure-gated margin: precision lock-in + k tuning

**Base:** amend19 (closure-gated margin v2a, the first method to beat the fixed-margin frontier:
k=0.5 → safe-success 0.482 / completeness 0.722 / ever 0.320, strictly above the constant frontier).
No controller-logic change; this run (a) fixes the numerical-precision setting and (b) tunes k around
the sweet spot with more seeds. Frozen, pre-run, SHA before run, detached, self-push.

**Motivation:** amend19 exposed that the `learned` closed loop is numerically chaotic — a 1-ulp
difference in u_nom at step 1 amplifies to macroscopically different trajectories, and
`precision=highest` alone shifted one seed's completeness 0.66→0.84. So (i) individual trajectories
are not bit-reproducible (chaos — expected for a dense multi-agent closed loop), only seed-ensemble
means+CIs are; (ii) the precision setting is a perturbation source that must be fixed and documented;
(iii) `precision=highest` is the more accurate reference and must be adopted before we trust the win.

---

## 1. Design

- **Precision: `jax.default_matmul_precision("highest")` set globally and logged** — this is now the
  standard for all learned-path runs. real re-run under it too (should stay ≈ bit-stable; it has no MLP).
- **Method (unchanged from amend19):** d_safe_ij = max(0.55, D_COLLISION + k·max(0,v_closing_ij)·h),
  h = 0.3 s, floor 0.55, D_COLLISION = 0.30.
- **Fixed:** s = 1.05, gate = 2.309, N = 100, reuse s=1.05 Φ/m* solve, learned = frozen amend11
  checkpoint.
- **Swept k ∈ {0.25, 0.50, 0.75}** (bracket and refine the amend19 sweet spot at 0.5; h fixed, since
  the effective gain is k·h — no need to sweep both). Arm: learned.
- **Reference arms re-run under precision=highest (apples-to-apples):** learned fixed d_safe ∈
  {0.55, 0.70}. (The amend18/19 fixed numbers were at default precision — re-establish them here.)
- **Seeds: 0–9 (10)** — more seeds to pin the mean under chaos. Seed-level cluster bootstrap.
- **Log:** safe-success, completeness, ever d<0.30, p5 worst sep, CBF intervention, slack, realized
  d_safe (mean/p50/p95), v_closing distribution.

## 2. Pre-registered questions

- **Q1 (robustness of the win):** does closure-gated k≈0.5 still beat BOTH fixed 0.55 and fixed 0.70
  on safe-success under precision=highest, 10 seeds? (amend19 at default precision: 0.482 vs 0.226 /
  0.290.) If yes → the method win is real and reproducible at the ensemble level.
- **Q2 (optimal k):** which k maximizes safe-success while keeping completeness ≳ 0.7 and pushing
  ever<0.30 as low as possible? Is a lower k (0.25) even better, or is 0.5 the knee?
- **Falsifier:** if under precision=highest the closure-gated advantage collapses to within CI of the
  fixed frontier → the amend19 win was a precision/noise artifact, not the method. Report honestly.

## 3. Reproducibility note (for the paper)

Dense multi-agent closed loops with an MLP in the loop are chaotic: bit-identical trajectories are not
achievable across precision settings / hardware, so all learned results are reported as seed-ensemble
means with seed-cluster-bootstrap CIs, under a fixed, documented precision (`highest`). This is the
standard going forward; single-seed trajectory comparisons are not used for claims.

## 4. Compute

3 k-values × 10 seeds + 2 fixed refs × 10 seeds = 50 rollouts, no new solve. ≈ 30–40 min on the A40
(precision=highest is slower), ~$0.5. Detached, self-push.
