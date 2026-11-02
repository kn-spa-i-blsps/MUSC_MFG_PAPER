# Amendment 31 — addendum 4 (PI, 2026-11-02): supplementary ablations, replication and chaos ensemble

Registered BEFORE any amend31 G-win result was opened. HEAD at registration: fcc2f7f4310e65d1dd40979f73f5436fb64106db
(the last pod checkpoint was N=025; the run of 400 rollouts was in progress).

**Status of what was seen:** no G-win number has been read. Known were only:
- the number of metrics files (80 at 15:56 UTC);
- the checkpoint commit titles;
- the fact that `git pull` brought the N=025 metrics files into the local tree (commit fcc2f7f; not opened).

The automatic watch job that would have read the G-win report on arrival was cancelled first.

## Overriding rules (PI, verbatim summary)
1. Do not stop, restart or change the running 400-rollout amend31 run; do not recompute it.
2. **The addendum-1 G-win verdict stays exactly as registered.** This addendum does not change, replace or reinterpret it.
   Everything below is SUPPLEMENTARY.
3. No G-win result is opened before this addendum is committed.
4. No retraining of σ or of the predictor; no recalibration of sigma_bin. No change of seeds, thresholds, metrics,
   D_COLLISION, arena, horizon, precision or caches. No float16. precision=highest, as in amend31.
5. No amend28 or other queue job. The queue is blocked without touching running processes and is not unblocked.
6. New code is additive only. A technical repair is allowed only if it does not change the method and passes identity;
   it is described and classified PLUMBING. A repair needing a change of method, arm, threshold or data is not done:
   diagnosis and final report instead. Nothing is waived.
7. Pod, volume and artifacts are not deleted.

## Step 0 — results (no G-win data used)
- **a) Comparator scope.** [Ad1] §4: "the best fixed margin per N by mean safe-success". The code agrees:
  experiments/amend31_sigma_cbf.cmd_report computes `best` inside the loop over N. Applied as registered: per N.
- **b) d_safe on rows < 2 m is NOT logged.** The rollout logs have d_safe_sum / d_safe_n / d_safe_hist only over all
  valid CBF neighbour rows (env/rollout.py, `adaptive and cbf_on` block). σ coverage has a < 2 m variant, d_safe does
  not. Rollouts are not recomputed. **fixed_match therefore uses the logged mean over all valid CBF rows.**
  - Limitation: this mean includes far rows (≥ 2 m), where σ is usually smaller and d_safe sits at the 0.70 floor, so
    it is lower than a < 2 m mean would be.
  - Exact definition: fixed_match(N) = max(0.70, Σ_seeds 0–9 Σ_steps d_safe_sum / Σ_seeds Σ_steps d_safe_n), i.e.
    row-weighted, pooled over seeds 0–9, of sigma_bin_a0.05 at N.
- **c) Seeds 200–209 are unused.** Grep over prereg/, experiments/, logs/ and results/ (local and the pod volume) finds
  no rollout/datagen with seeds 200–209. The hits are N=200 or epochs=200, not seeds. Replication uses 200–209.
- **d) State.** amend31 run 2 identity PASS (commit 6ddfb66, bit-equal to amend22 on all IDENTITY_KEYS). The run was in
  progress (N=025 done). Steps 2–4 start only if amend31 ends with its DONE marker and a report.json. Otherwise:
  diagnosis and report.
- **Queue block (rule 5).** queue_B_pod.sh checks `/workspace/<id>_DONE` at the moment each `run` is called and skips
  that job. Markers were created for amend28 / amend32 / amend29 / amend30 (15:56 UTC), with the content "BLOCKED (not
  run) by PI 2026-11-02, addendum 4 … NOT a completion marker". The script file and the running processes were not
  touched (bash reads scripts lazily, so the file is not edited).

## Step 2 — after the 400 rollouts (PI text, implemented as below)
Generate the registered G-win report unchanged: the amend31 chain itself runs `amend31_sigma_cbf report`. Only then
freeze:
- **best fixed** per N from seeds 0–9 (report.json "best_fixed").
- **fixed_match(N)** per step 0b, from sigma_bin_a0.05 seeds 0–9 only.

Addendum-4 arms, N ∈ {150, 200} only, existing fixed path only (the amend31 fixed cell with only cbf.d_safe changed):
- **conf_const_a0.05:** d_safe(N) = max(0.70, 0.30 + q_raw(N)).
  - q_raw(N) = the finite-sample split-conformal quantile of level 0.95 of the raw error ‖x̂ − x‖ on the CAL windows:
    the ⌈(n+1)·0.95⌉-th smallest.
  - x̂ = the frozen v3 prediction. CAL = seeds 108–109, the same 200k uniform draws per seed per N as stage 1,
    true distance < 2 m.
  - No σ, no bins, no TEST, no rollout results.
- **fixed_match:** d_safe(N) = max(0.70, frozen value above).

If an arm's d_safe equals, in f32, an existing fixed arm at that N (0.70 / 0.85 / 1.00 / 1.20), it is not recomputed.
The existing amend31 rollouts are used and the ablation is recorded as identical to that arm. In the replication,
fixed-path arms with the same f32 d_safe share one run.

## Step 3 — rollouts, in this order (PI text)
- **Priority A, ablations:** conf_const_a0.05 and fixed_match × N {150, 200} × seeds 0–9. Up to 40 rollouts.
- **Priority B, replication with the frozen comparator:** sigma_bin_a0.05, best fixed, conf_const_a0.05, fixed_match
  × N {150, 200} × seeds 200–209. 80 rollouts, fewer after f32 de-duplication. Run regardless of the G-win result.
- **Priority C, chaos ensemble:** sigma_bin_a0.05 and best fixed × N {150, 200} × seeds 0–9 × members m = 1..5.
  200 rollouts.
  - Perturbation of member m: np.nextafter in f32 toward +inf on the x coordinate of the initial position of agent
    index m−1, applied right after scenario.initial_positions(key). No change to any RNG stream.
  - m = 0 is the existing base rollout.
  - Implementation: experiments/amend31_add4_run.py, an additive wrapper around experiments.ladder.main. The cell key
    `init_perturb_agent` absent means the scenario object is untouched.
  - CPU check before this commit: with the key absent, initial positions are bit-equal; with agent 2, exactly one entry
    changes, by exactly +1 ulp.

**Gates before the priorities, each must pass:**

| gate | test | if it fails |
|---|---|---|
| g1 | the add4 harness at fixed 0.70 is bit-equal to amend22 on all IDENTITY_KEYS (real, N=100, seed 0) | A, B and C are not run (all use the harness) |
| g2 | perturbation off gives bit-equal logs to the base for sigma_bin_a0.05, N=150, seed 0 (all logged arrays) | C is not run |
| g3 | perturbation on changes the trajectory; the first divergence step of `pos` is recorded | C is not run |

Note for g2, known before this commit: the learned arm is not run-to-run bit-deterministic on the GPU under default
autotuning (amend18/19, G-pred integrity 8e-8 rel.). g2 may therefore fail for that reason alone. It is not
re-engineered. If it fails, C is not run, and this is reported.

**Limit:** 6 hours from the start of step 3, or 15 USD of pod time accrued from the start of step 3 (1.79 USD/h).
The 6 h bound is the binding one, about 10.7 USD. A priority is not started if elapsed + its estimate exceeds the limit.

**Estimated cost before start:**
- per-rollout wall: fixed path ≈ 25 s at N=150 and ≈ 37 s at N=200 (amend22 learned logs); σ path ≈ 2× that
  (identity: 25 s vs 12 s at N=100); 2 processes in parallel;
- A ≈ 12 min, B ≈ 30 min, C ≈ 85 min, gates ≈ 3 min. All 320 rollouts ≈ 2.2 h ≈ 4 USD;
- the chain uses conservative estimates A 20 / B 40 / C 110 min for the limit check.

Results are written per seed (skip-existing) and pushed per priority, so an interrupted run resumes without recompute.

## Step 4 — analysis and report (PI text) → logs/overnight/review_amend31_add4.md
Everything is SUPPLEMENTARY unless it explicitly quotes the registered verdict.
1. **G-win verdict** exactly per [Ad1], plus "registered v1".
   - Raw means per arm and N, also N=25/50/100 as descriptive.
   - Base rates of safe-success and completeness.
   - If the values at N=150 or N=200 are near 0 or 1 in all arms, state that the metric does not separate the arms.
2. **Differences at N=150 and N=200:** sigma_bin_a0.05 minus each of best fixed, conf_const_a0.05 and fixed_match,
   for safe-success and completeness:
   - unpaired seed bootstrap (diff_ci, same procedure);
   - paired bootstrap over seeds, 10 000 replicates;
   - cross-arm correlation over seeds;
   - exact sign-flip permutation test of the mean paired difference: all 2^10 sign patterns, two-sided,
     p = P(|mean| ≥ |observed|);
   - Holm over N=150 and N=200 for each comparison and metric, applied to the sign-flip p.
   Seeds 0–9 and the replication 200–209 separately. Combined 20 seeds only as an extra, never in the main conclusion.
3. **Chaos:**
   - per arm, N and seed: the SD of safe-success across members 0–5;
   - variance split: within-seed (chaotic) = mean over seeds of the member variance; between-seed = variance of the
     member means;
   - then sigma_bin_a0.05 − best fixed on member means within seed, with a bootstrap over seeds (paired and unpaired).
4. **Closed-loop coverage** of sigma_bin_a0.05 at N=150/200 in the three registered row variants (all valid / true
   distance < 2 m / binding), with seed-bootstrap CIs.
   - Offline reference: G-cal TEST, binned α=0.05: coverage (pairs < 2 m) and coverage on σ_c > 0.40 m, the same
     condition as d_safe > 0.70. The change vs offline is reported.
   - Descriptive costs: makespan, time to 90 %, mean speed, mean and p95 d_safe, binding fraction, ever<0.30.
5. **Reading rules:**
   - "σ wins over a constant per-N margin schedule" only if, at BOTH N, on seeds 0–9 AND in the replication, the paired
     CI95 lower bound of safe-success is > 0 vs both conf_const_a0.05 and fixed_match, and the completeness lower bound
     is ≥ −0.02. Otherwise: "brak dowodu, że σ per sąsiad przewyższa stały margines per N".
   - "Effect above chaos" only if the point-3 lower bound is > 0 at that N. Otherwise: the effect is not distinguished
     from chaos.
   - "Calibration transfers to the closed loop" only if, at both N, the point coverage on rows < 2 m and on binding rows
     is ≥ 0.93. This is descriptive and does not change G-win; if it is not met, there is no recalibration and no
     iteration in this run.
   - PARTIAL is not a confirmation of the effect at an N without passing Holm.
   - The comparator chosen after the run is biased in its own favour: described, not corrected.
6. **Animations** for N=200, seed 0: sigma_bin_a0.05, best fixed, fixed_match. The paths go into the report. They are
   not treated as visually reviewed until a human has watched them.
7. **Final sections:**
   - FAKTY ZAREJESTROWANE;
   - WYNIKI SUPPLEMENTARY;
   - AWARIE I NAPRAWY (classification + commits);
   - CZEGO NIE ZROBIONO I DLACZEGO;
   - all commits, HEAD, spec SHA256, time and cost.

  Commit the report, the result JSONs and the animations. The pod is left untouched and the queue blocked.

## Code (additive only; no existing file modified)
- experiments/amend31_add4.py — qraw / freeze / write-cells / gate / analyze / animate.
- experiments/amend31_add4_run.py — ladder wrapper with the optional perturbation.
- logs/overnight/amend31_add4_chain_pod.sh — waits for the amend31 DONE/FAILED marker and the queue end, then steps 2–4.
