# Amendment 28 — delay-free baseline validation, layer ablations, MFG macro fidelity

**Base:** amend21 geometry, amend22/24/27 arms. PI goal step 1 (2026-10-31): before showing delay damage, show the
delay-free two-layer controller works as the literature says, and that each layer earns its place. Also measure the
delay damage ON THE MFG LAYER itself (density tracking), not only collisions. Frozen, pre-run, SHA before run,
detached, self-push. Code: experiments/amend28_baseline_validation.py; ladder gains an optional cell key `cbf_on`
(absent = True = unchanged).

## Design (d_safe 0.70, N ∈ {25, 50, 100, 150, 200}, seeds 0–9, precision=highest, v3 weights sha 9583cb7c…)
| arm | perception | controller |
|---|---|---|
| fresh | fresh | MFG (c_w 1) + CBF |
| fresh_nocbf | fresh | MFG, CBF off (box clip only) |
| fresh_nomfg | fresh | decoupled routing: macro solved with c_w = w_bar = 0 (single-agent optimal control, no mean-field coupling), c_dens = 0, + CBF |
| noise_only | noise_only | MFG + CBF |
| real | real | MFG + CBF |
| learned | learned v3 | MFG + CBF |
300 rollouts + 5 new solves (fresh_nomfg caches, deleted after the run). fresh/noise_only/real/learned replicate
amend22/24/27 cells (reported side by side; not pooled).

## Macro fidelity (defined before data)
Every 1 s over the full horizon: agents' positions histogrammed on the solver grid and Gaussian-smoothed (σ = 1.0 m)
vs the c_w=1 equilibrium m*(t) smoothed identically, both normalized → total variation tv(t) ∈ [0,1]; tv_mean per
rollout; finite-N floor = tv of N i.i.d. samples from m*(t) (5 draws); goal-mass gap = mean |agents-in-goal fraction −
m*-mass-in-goal fraction|. All arms are compared to the SAME c_w=1 m* (the MFG prediction).

## Questions and pre-registered readings
- **Q1 baseline works:** fresh safe (ever<0.30 CI upper ≤ 0.10 and completeness ≥ 0.90) at every N. Failure → the
  delay study's reference itself is broken at that N (reported; N excluded from "damage" claims).
- **Q2 CBF needed:** fresh_nocbf − fresh ever<0.30 > 0 with CI excluding 0 at some N.
- **Q3 MFG layer earns its place:** fresh − fresh_nomfg on completeness or safe-success, CI excluding 0, sign
  reported per N. A null or negative result is reported as such (thesis-relevant, not plumbing, unless the decoupled
  solve fails to converge or NaNs).
- **Q4 MFG tracks its prediction:** fresh excess tv (tv − floor) reported per N; fresh tv < fresh_nomfg tv (CI) = the
  MFG equilibrium describes the realized swarm better than a decoupled policy does.
- **Q5 delay damages the MFG layer and compensation restores it:** real tv − noise_only tv > 0 (CI) = delay damage
  on macro tracking; learned tv − real tv < 0 (CI) = NN restores it; recovered fraction (real − learned)/(real −
  noise_only) reported. Same for the goal-mass gap.

## Review
logs/overnight/review_amend28.md.
