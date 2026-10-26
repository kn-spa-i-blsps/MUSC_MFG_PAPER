# Project A → Phase B brief: uncertainty-as-guarantor (conformalized σ-CBF)

(PI covering note, verbatim: "Nowy standing brief: AGENT_BRIEF_phaseB.md (dostarczony), rozszerza AGENT_BRIEF_overnight.md —
wszystkie jego zasady dalej obowiązują (precision=highest, prereg per eval run, detached self-push, VERIFICATION, REVIEW
MODE, progress.md + review_<id>.md). Faza B, w kolejności: 1. G-pred (make-or-break, TANIE, najpierw): czy per-sąsiad błąd
estymaty jest przewidywalny/heteroskedastyczny — wytrenuj model magnitudy błędu z inputu predyktora na pooled-N, zmierz R²
i przewidywalność OGONA. Zanim ogłosisz FAIL, wyczerp naprawy §C (pooled/balanced, capacity, near-pair oversampling, per-N
diagnostycznie). Sklasyfikuj wynik PLUMBING vs THESIS-CRITICAL. 2. Jeśli G-pred PASS: zbuduj conformal σ-predyktor
(heteroskedastyczny/kwantylowy + kalibracja), bramka G-cal (reliability diagram per N). 3. σ-CBF: d_safe_ij = max(0.70,
0.30 + k·σ_ij), identity przy k=0 = fixed 0.70. Sweep k. Ewaluacja na crossoverze + τ, porównanie z fixed-0.70 i v3.
Bramka G-win: schodzi z frontier przy N≥150 (CI rozdzielone)? Masz autonomię na kosztowny engineering, który POWINIEN
wyjść (trening na innych N itd.) — gap tam to hydraulika do naprawy, nie falsyfikacja tezy. Nie overclaimuj i nie
under-claimuj myląc hydraulikę z martwą tezą. Rano: progress.md, review per wynik, figury CONFIRMED, lista napraw PLUMBING i
kwarantann — żeby engineering i dowód tezy dały się rozdzielić.")

Extends AGENT_BRIEF_overnight.md. All its standing rules still hold: precision=highest, 10+ seeds,
seed-cluster bootstrap CIs, pre-registration per reported EVAL run (amendment + sha256 before the run),
detached on the pod with self-push, VERIFICATION §3 on every result, REVIEW MODE §4 after every result,
append `logs/overnight/progress.md` and a `review_<id>.md` per result. You have autonomy to run expensive
things you decide are needed (see §C) — use the night.

## The thesis (B)

Delay compensation in a two-layer MFG + CBF controller, **with a formalized per-neighbour predictor
uncertainty as the safety guarantor**: the predictor outputs a calibrated per-neighbour uncertainty σ_ij,
the CBF margin is d_safe_ij = max(d_floor, D_COLLISION + k·σ_ij), and conformal calibration turns this
into a conditional guarantee "collision-free with probability ≥ 1−δ given σ coverage holds". The method
targets the TAIL of the estimation error (the rare high-error moments that cause collisions), which
mean-improving predictors (v3, v4) could not kill, by making the CONTROLLER act on uncertainty rather
than chasing a perfect prediction.

## A. Thesis-critical gates (a clean failure here IS a real result about the thesis)

Only declare one of these failed AFTER exhausting the engineering fixes in §C. A survived failure is a
genuine negative and must be reported as such.

- **G-pred (make-or-break, do FIRST, cheap):** is the per-neighbour estimation error **predictable
  (heteroscedastic)**? Train a model to predict error magnitude from the predictor's own input, on the
  pooled-N data. Measure how much error variance it explains (R²), and specifically whether it predicts
  the TAIL (e.g. AUC for "error in top decile", or R² on the high-error subset). PASS ≈ it explains a
  substantial, tail-weighted share. FAIL (survives §C fixes) → σ carries little info → σ-CBF collapses to
  a fixed margin → the uncertainty-as-guarantor thesis has no teeth. That is a real negative; report it.
- **G-cal:** can σ be **calibrated** (conformal / reliability diagram) so predicted σ matches realized
  error per N and the coverage level holds out-of-sample? Required for the guarantee. A miscalibration
  that survives recalibration attempts is a real limitation.
- **G-win:** does the σ-CBF **beat the fixed-margin frontier** — lower ever<0.30 at EQUAL completeness,
  especially at N ≥ 150 where every fixed margin ties — i.e. get off the safety/completeness frontier
  that nothing has escaped? This is the method's value. CIs must separate.

## B. The method build (once G-pred passes)

1. Uncertainty head: heteroscedastic NLL and/or quantile-regression head on the v3 feature map, pooled-N
   training. Consider a small deep ensemble if a single head miscalibrates.
2. Conformal calibration of σ on held-out seeds, per N / per T_frame.
3. σ-CBF in the controller: d_safe_ij = max(d_floor=0.70, D_COLLISION + k·σ_ij); sweep k (coverage δ).
   k=0 / σ→const must reduce to fixed 0.70, bit-identical on the no-MLP path — identity gate.
4. Evaluate on the crossover (N ∈ {25..200}) and the τ sweep; compare to fixed-0.70 and v3. Win condition
   = G-win.

## C. Engineering latitude — "things that should just work" (a gap here is PLUMBING, not a thesis signal)

You are authorized to spend compute, on your own initiative, on work that SHOULD succeed and whose
weakness would be an artifact of incomplete engineering, NOT evidence against the thesis. Before you ever
attribute a weak or negative result to the thesis, you must first rule out — and, if plausible, FIX —
that it is one of these:

- **Training distribution / OOD.** The uncertainty head, like v1→v3, may be OOD if trained on too narrow a
  range. Train it POOLED and N-BALANCED across N ∈ {25,50,100,150,200} (and across T_frame if relevant).
  If a per-N gap appears, add that N's data / rebalance / **train per-N as a diagnostic ceiling** — these
  are expected engineering steps, not thesis tests. A learned curve that should be monotone (σ rising with
  T_frame, coverage flat across N) but isn't → suspect training coverage first.
- **Capacity / recipe.** Underfit uncertainty → wider/deeper head, near-pair oversampling (as v2→v3), more
  windows, better target (quantile vs NLL vs ensemble). Exhaust the obvious recipe space before concluding
  σ is uninformative.
- **Calibration method.** Raw σ miscalibrated → temperature scaling, conformal, per-bin recalibration.
  Miscalibration of a RAW head is expected; only miscalibration that survives recalibration is a result.
- **Seeds / statistics.** A surprising point may be seed noise → add seeds before interpreting.
- **Artifacts.** Always the VERIFICATION §3 checklist (precision, identity, numerical chaos, wrong-knob,
  floor starvation, geometry/density confound).

**Classification rule (add to REVIEW MODE Step E):** for every non-positive result, first classify it:
**PLUMBING** (an engineering fix in §C is plausible → do the fix, re-run, do NOT call it a thesis signal)
vs **THESIS-CRITICAL NEGATIVE** (it is a §A gate AND it survived the relevant §C fixes → report as a real
negative). Never let a plumbing gap read as "the thesis is wrong"; never let a genuine §A failure be
hidden as plumbing. State which it is and why in the review file.

## D. Definition of done for B

- G-pred resolved (predictable or not, with the §C fixes exhausted).
- If G-pred passes: a calibrated σ head (G-cal) and the σ-CBF crossover vs fixed-0.70/v3, with G-win
  resolved (escapes the frontier at N≥150, or does not — either way CONFIRMED).
- The conditional-guarantee statement, with its coverage assumption and where it holds / breaks.
- `progress.md`, per-result reviews, figures for CONFIRMED cells, and a list of every PLUMBING fix you ran
  and every quarantined/ARTIFACT item, so the engineering and the thesis evidence are separable in the
  morning.

Do not overclaim, and do not under-claim by mistaking plumbing for a dead thesis. A real, §C-exhausted
negative on a §A gate is a valid result; an unfixed engineering gap is not.
