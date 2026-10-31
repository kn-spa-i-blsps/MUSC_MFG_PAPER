# Amendment 31 — Phase B steps 2–3: conformal σ head (gate G-cal) and σ-CBF vs the fixed-margin frontier (gate G-win)

**Base:** AGENT_BRIEF_phaseB.md §A/§B; G-pred PASS on the registered criteria (review_gpred.md, commit c404384).
Frozen, pre-run, SHA before run, detached, self-push. Runs after amend30 on the same pod. Code: experiments/phaseB_sigma.py,
experiments/amend31_sigma_cbf.py, logs/overnight/amend31_chain_pod.sh; controller change = CBF `margin_mode: sigma`
(absent = unchanged path).

## 1. σ head (step 2)
- Error model g = the G-pred recipe D, unchanged: pooled N-balanced, near_frac 0.5, 60k windows / seed / N, MLP 2×128,
  heads log|e| (MSE) + q0.9 (pinball), inputs = v3 window (20) + v3 output (2) + |F|, stale range, |F − stale|, T_frame.
  Error e = ‖v3(X) − Y‖ of the frozen v3 (sha 9583cb7c…) on datagenL histories (closed-loop learned v3 d0.70).
- Seeds: TRAIN 100–105, VAL 106–107 (early stopping), CAL 108–109, TEST 110–111. Note: G-pred trained on 100–107. CAL
  and TEST windows are drawn uniformly (near_frac 0, 200k / seed / N) and restricted to true distance < 2 m.
- Primary calibration: normalized split conformal, Mondrian per N. Score r = e / exp(g_log); q_N(α) = the
  ⌈(n+1)(1−α)⌉-th smallest CAL score of N; σ_c = q_N(α)·exp(g_log); α ∈ {0.10, 0.05, 0.01}.
- **Registered §C calibration fix, computed every time:** Mondrian per (N, σ-quintile). It uses the CAL quintile edges
  of g_log within N, giving 5 bins with one q each.

## 2. Gate G-cal (criteria fixed now; TEST seeds)
- **Marginal:** |coverage − (1−α)| ≤ 0.02 for α 0.10 and 0.05, and ≤ 0.01 for α 0.01, at every N.
- **Conditional:** coverage ≥ 0.85 in every σ-decile within N, at α = 0.10, at every N.
- **Verdict:**
  - PASS = the per-N calibration meets both.
  - PASS_AFTER_FIX = only the σ-quintile Mondrian meets both. This is PLUMBING, fixed by the registered method.
  - PARTIAL = marginal met by either, conditional by neither.
  - FAIL = marginal met by neither. A THESIS-CRITICAL limitation for the guarantee. The σ-CBF results then stand as an
    uncalibrated margin.
- Reported, not gating:
  - coverage per age tercile;
  - coverage on pairs where the margin binds (0.30 + σ_c > 0.70);
  - pooled-q coverage;
  - **closed-loop coverage** in every σ-CBF rollout: the fraction of the CBF's valid neighbour rows with
    ‖x̂ − x‖ ≤ σ_c (all rows / true distance < 2 m / binding rows). The closed-loop coverage is the empirical check of
    the guarantee's assumption under the distribution shift the σ margin itself causes.
- Data-independent observation made before this commit: in a CPU test of the scripts on 4k windows (not the gate), the
  marginal coverage was within tolerance, but the lowest σ-decile was under-covered (0.82–0.87). That is why the binned
  fix is registered up front.

## 3. σ-CBF (step 3) and its identity
- d_safe_ij = max(0.70, D_COLLISION + k·σ_c,ij), with D_COLLISION = 0.30 and k = 1.
- σ_c is evaluated online per (i, j) from the same draws perceive() used: `learned_window_and_pred` with the same key,
  behind an optimization barrier.
- **Identity (gate, run first; aborts the chain):** k = 0 ⇒ d_safe = 0.70 on every row. N=100 seed 0, real arm: it
  must be bit-equal on IDENTITY_KEYS to the committed amend22 real_d0.70 seed 0. On a CPU check before this commit,
  18/18 logged arrays were bit-identical to fixed 0.70 for both the real and the learned arm (N=25, 80 steps).
- The learned arm is not run-to-run bit-deterministic on the GPU (amend18/19), so the GPU identity uses real.

## 4. Design (amend21 geometry, per-N caches unchanged, precision=highest, seeds 0–9, learned v3 arm)
| arm | margin |
|---|---|
| fixed0.70, fixed0.85, fixed1.00, fixed1.20 | constant d_safe (the fixed-margin frontier, re-run in this chain) |
| sigma_a0.1, sigma_a0.05, sigma_a0.01 | σ-CBF, per-N conformal q_N(α), k = 1 |
| sigma_bin_a0.05 | σ-CBF, (N, σ-quintile) Mondrian q, α = 0.05, k = 1 |

× N ∈ {25, 50, 100, 150, 200} = 40 cells, 400 rollouts + 1 identity rollout.

## 5. Gate G-win (criteria fixed now)
- **Primary arm:** sigma_a0.05 if G-cal = PASS; sigma_bin_a0.05 if G-cal = PASS_AFTER_FIX; sigma_a0.05 otherwise
  (then labelled uncalibrated).
- **Comparator per N:** the best fixed margin by mean safe-success among the four, chosen after the run. This hands
  the comparator the advantage.
- **PASS:** primary − best fixed safe-success > 0 with the CI lower bound > 0 at both N = 150 and N = 200.
- **PARTIAL:** at one of the two.
- **NOT_MET:** at neither.
- Also reported per N and arm:
  - differences vs best fixed and vs fixed0.70 on safe-success, ever<0.30 and completeness;
  - realised d_safe mean / p95, σ binding fraction and closed-loop coverage.
- Non-primary σ arms are secondary: a win only there is labelled exploratory. The relevant §C fixes are k ≠ 1,
  another α, and the binned or ensemble head. Only after those may a NOT_MET be read as the thesis-critical negative
  (brief §A); before them it is a result for this configuration only.

## 6. The conditional guarantee (to be stated in the review, from the data)
"If ‖x̂_ij − x_j‖ ≤ σ_c,ij holds on a CBF row, then h_ij ≥ 0 under d_safe_ij ≥ 0.30 + σ_c,ij implies a true
separation ≥ 0.30 on that row at that step." The review reports where its premise holds: TEST and closed-loop
coverage per N. It also reports where it does not cover the controller: discrete-time CBF, slack, k_nb truncation,
rows outside the CBF set.

## Review
logs/overnight/review_amend31.md (REVIEW MODE, PLUMBING vs THESIS-CRITICAL for every non-positive result).
