# MUSC-MFG Project A — Autonomous overnight brief

(PI covering note, verbatim: "Standing brief: AGENT_BRIEF_overnight.md (dostarczony). Przeczytaj go i działaj autonomicznie
przez noc wg niego. Zasady twarde: precision=highest wszędzie, prereg per raportowany run (aneks+SHA przed runem, wyniki
przez git -f), detached na podzie z self-push, i logs/overnight/progress.md dopisywany po każdym kroku +
logs/overnight/review_<amendNN>.md po każdym wyniku. Priorytet: FAZA 1 — staleness-conditioned predyktor (Δt per sąsiad
jako jawny input, trening na pooled N∈{25,50,100,150,200}), BRAMKA: open-loop MSE mały i płaski na wszystkich N zanim
ruszysz crossover (bez tego nie ruszasz). Potem crossover real vs learned fixed 0.70, 10 seedów. REVIEW MODE. Potem fazy
2–4 jak starczy czasu, każda w pełni zreviewowana. Nie overclaimuj. Dobrze wyjaśniony negatyw to wynik, nie porażka. Rano
chcę: progress.md, pliki review, CONFIRMED figury i listę wszystkiego, co poszło do kwarantanny i dlaczego.")

You are running unattended, probably all night. Use the time. This brief tells you WHAT we are trying
to achieve and WHAT TO DO at each outcome, so you can analyse your own results, impose your own
corrections, and iterate — without waiting for me. Two hard rules above everything: **(1) every result
must be shown to come from reality, not a bug or a precision/ OOD/ confound artifact — see VERIFICATION
and REVIEW MODE; (2) keep the pre-registration discipline for every reported run.**

Work on the pod, detached (tmux/nohup), self-push each result (report.json + `git add -f` + commit +
push + DONE marker). Leave a running `logs/overnight/progress.md` you append to after every step, so I
can read the whole night's reasoning in the morning.

---

## 0. Where we are (established, under precision=highest unless noted)

- **Thesis, confirmed on the baseline:** TDMA staleness grows with the swarm (T_frame = N·t_slot) and
  destroys decentralized-CBF safety as N grows. `real` (no compensation): safe-success 1.000 at N=25,
  0.284 at N=50, 0.000 from N=100; its xhat_mse rises monotonically 0.012→3.115. Clean staleness
  barrier.
- **Compensation gives large headroom:** at N=100 (big/sparse arena 43×19), `learned fixed d_safe=0.70`
  = safe-success 0.805 / completeness 0.98 / ever<0.30 0.18, versus `real` 0.000. Compensation makes a
  regime workable where the baseline is completely dead.
- **Selective margin is DEAD.** closure-gated margin loses to a plain constant margin (closing 0.75 =
  0.369 vs fixed 0.70 = 0.805 at N=100). Do not pursue closure-gating further. **The method is
  "compensation + a fixed safety margin."**
- **Predictor is OOD-brittle (critical limitation).** The N=100-trained predictor gives MSE 7.4 at
  N=25 (vs real 0.012), turning a trivially-safe case into a catastrophe. All learned results off its
  training condition are OOD-contaminated and cannot be trusted.
- **Density matters as much as N.** The same N=100 went from safe-success 0.41 (tight arena, amend20)
  to 0.805 (sparse arena, amend21). Operating density is a first-class axis; always report it.
- **The numerics are chaotic** (MLP in a dense closed loop): single trajectories are not reproducible;
  only seed-ensemble means with cluster-bootstrap CIs are. Default matmul precision produced a fake 2×
  win once (amend19). Always `precision=highest`.

## 1. The goal

Produce the **final, paper-ready sweeps** showing, honestly:
1. **N crossover (headline):** staleness barrier for `real`, and how far compensation pushes the safe
   swarm size. safe-success & ever<0.30 vs N, real vs learned.
2. **Operating-density sweep:** safe-success vs density (gate width / arena), real vs learned, at fixed
   N. Shows the density dependence and that compensation's advantage holds across densities.
3. **Pure-staleness sweep (τ = t_slot):** at fixed N and geometry, isolate staleness from density.
4. **Mechanism panel:** xhat_mse and worst-case separation vs N/τ (estimation error drives collisions),
   plus the predictor OOD-brittleness result as a stated limitation and the motivation for across-N
   training.

A sweep is "paper-ready" only when every cell in it is **CONFIRMED** by REVIEW MODE (§4).

## 2. Locked decisions (do not re-litigate)

- Method = learned delay-compensation + fixed CBF margin. Closure-gating dropped.
- precision=highest everywhere. 10 seeds minimum, seed-cluster bootstrap CIs.
- D_COLLISION = 0.30. Report completeness alongside safe-success always (safety-by-freezing must be
  visible). safe-success = reached r_goal AND never true d<D_COLLISION.
- Pre-reg per reported run: amendment file with sha256 committed BEFORE the run; results via `git -f`.
  Development/training iterations (predictor tuning) are exploratory and need not be pre-registered;
  the EVALUATION runs that produce reported numbers must be.

## 3. VERIFICATION — non-negotiable, run on EVERY result before you believe it

For each result, before reporting it as real, check it against the traps we have actually hit:

1. **Identity / k0 sanity.** Any code change must reduce to prior behaviour at its trivial setting;
   for the no-MLP (`real`) path this must be bit-identical. If not, the change altered more than
   intended — find out what.
2. **Precision.** Confirm precision=highest is active in the run (log it). If a result is surprising,
   re-run it under highest before believing it. Never compare a highest run to a default-precision one.
3. **OOD.** For any learned run, log open-loop predictor MSE at that condition. If learned MSE ≫ real's
   stale MSE, the predictor is OOD and the "result" is an artifact, not a property of compensation —
   quarantine it.
4. **Wrong knob.** If a swept lever does not move the outcome (e.g. ever<0.30 flat across the sweep),
   you are sweeping the wrong variable — find which variable actually controls collisions before
   concluding anything.
5. **Margin starvation.** For any margin scheme, log the realized d_safe distribution (mean/p50/p95).
   If p50 is below the working constant baseline, you starved the typical pairs.
6. **Geometry/density confound.** Hold geometry fixed within a sweep. Never compare numbers across
   different arenas/gates as if only one variable changed. Always report the operating density (initial
   spacing, arena size, gate width).
7. **Chaos / statistics.** Report only seed-ensemble means with cluster-bootstrap CIs. Never claim
   anything from a single seed/trajectory. For a claim "A beats B", require their CIs to separate.
8. **Completeness-as-sham.** Always report completeness with safe-success. A high safe-success at
   collapsed completeness is a frozen swarm, not a win.

A result failing any check is **quarantined**, not reported, until the cause is found.

## 4. REVIEW MODE — run after EVERY original result

After a run produces numbers, do NOT move on. Run an adversarial self-review pass and write it to
`logs/overnight/review_<amendNN>.md`:

**Step A — mechanism.** State the physical mechanism that WOULD produce these numbers. Check the data
is consistent with it (e.g. if collisions dropped, xhat_mse or the realized margin should explain how).
If you cannot name a mechanism that fits, the result is SUSPECT.

**Step B — artifact scan.** Run the result through all eight VERIFICATION checks (§3). List each as
pass/fail with the evidence.

**Step C — consistency.** Does this contradict anything we already believe (§0)? If yes, exactly one of
them is wrong — say which and how you will decide (usually a targeted re-run).

**Step D — statistics.** Do the CIs actually support the claim? Is 10 seeds enough for this effect size,
or do the CIs overlap? If overlap, the claim is not supported — add seeds or report as null.

**Step E — verdict + fix list.** Classify: **CONFIRMED** (mechanism + all checks + stats hold) /
**SUSPECT** (something doesn't fit → diagnose) / **ARTIFACT** (a check failed → quarantine). For SUSPECT
or ARTIFACT, emit a concrete FIX LIST: the exact next run(s) that would close each gap. Then execute
the top fix, and REVIEW again. Loop until CONFIRMED or a genuine, explained negative result.

**Only CONFIRMED cells feed the paper figures.** A clean negative result, fully explained, is a valid
outcome — report it as such, do not force a positive.

## 5. Plan for tonight (phases, with branches)

### Phase 1 — Fix the predictor (OOD), then the N crossover. PRIORITY.
Goal: a predictor that is in-distribution across swarm sizes, then the clean N crossover.
- Build a **staleness-conditioned predictor**: add per-neighbour staleness Δt (time since its last TDMA
  update) as an **explicit input feature**; train on **pooled full-horizon data across N ∈
  {25,50,100,150,200}** in the fixed arena, precision=highest. Freeze + hash.
- **Verify OOD is fixed (gate):** open-loop predictor MSE at each N must be small and flat (no N=25-style
  blowup). If some N still blows up → diagnose (is Δt actually fed and scaled? is the training set
  balanced across N? does the feature cover the Δt range?) and fix before proceeding. Do not run the
  crossover on an OOD predictor.
- **Run the crossover:** arms `real` and `learned fixed 0.70`, N ∈ {25,50,100,150,200}, 10 seeds, fixed
  arena, precision=highest. (Optionally one `closing 0.75` point to re-confirm it's dead — not required.)
- REVIEW MODE. Expected: real crosses between N=25 and ~40; learned holds safety to much higher N. The
  CONFIRMED crossover gap is the headline. Branch:
  - If learned also degrades early → compensation gives little headroom; report honestly, diagnose
    whether it's residual OOD or genuine.
  - If learned is near-perfect everywhere → check it isn't a frozen/completeness artifact (§3.8) or a
    density artifact (§3.6).

### Phase 2 — Re-tune the fixed margin on the in-distribution predictor.
We picked d_safe=0.70 on the old (OOD-contaminated, tight-arena) data. Re-sweep d_safe ∈
{0.55,0.70,0.85,1.00} with the new predictor at a representative N (100) and the paper's operating
density, 10 seeds, to pick the margin that best trades safe-success vs completeness. REVIEW MODE.

### Phase 3 — Operating-density sweep.
At fixed N (100) and the new predictor, sweep operating density via **gate width** {1.5,2.3,3.5,5.0,7.0}
(and/or arena size) — one geometry axis, held clean. arms real vs learned(best margin). Map safe-success
& ever<0.30 vs density. REVIEW MODE. This also tells us the honest operating density for the other sweeps.

### Phase 4 — Pure-staleness sweep (τ).
At fixed N and fixed geometry/density, sweep t_slot {0.0025,0.005,0.0075,0.010} (only staleness changes;
same Φ/m* cache). arms real vs learned. Isolates staleness from density/size. REVIEW MODE.

Order is priority order. Get Phase 1 CONFIRMED first; it is the headline and everything else assumes a
working, in-distribution predictor. Do later phases only as time allows, each fully REVIEWed.

## 6. Definition of done (paper-ready)

Stop and summarise for me when you have, each CONFIRMED under §4:
- N crossover (real vs learned) with the safe-N gap and CIs that separate.
- A chosen operating margin and density, justified by Phases 2–3.
- The mechanism panel (xhat_mse & worst-sep vs N) and the OOD-brittleness limitation, explained.
- A written list of every quarantined/ARTIFACT result and why, so nothing silently inflates the story.
- `logs/overnight/progress.md` and the per-step review files, so I can audit the whole chain.

Do not overclaim. The honest shape right now is: staleness barrier (strong), compensation headroom
(strong at N=100), selective margin dead, predictor OOD-brittle (fix in progress), no hard collision-
free guarantee under large staleness. Confirm or correct each of these with CONFIRMED evidence; a
well-explained negative is a result, not a failure.
