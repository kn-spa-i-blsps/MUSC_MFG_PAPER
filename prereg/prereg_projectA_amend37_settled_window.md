# Amendment 37 — settled-window success metric, log-only (stage 1, link 1: "delay-free works")

**Purpose (one sentence):** decide whether "delay-free works" holds when success means a drone has actually settled in
the goal, rather than being inside at the single instant T, and decide which MFG terminal cost (amend36 ON vs amend33
OFF) settles the swarm better.

**Status:** REGISTERED before computing any settled-window number. PI order 2026-11-04, after review_amend36 flagged that
the amend36 fresh pass rests on a last-second inflow (14–18 drones outside at T−1 s). Rule 5 (CLAUDE.md): this metric is
introduced after results were seen, **at the PI's request**. The completeness-at-T time series of the last seconds is
already known (review_amend36 table); the settled-window values themselves have not been computed. The window
sensitivity requirement (below) is the guard against choosing a window that favours one controller.

**No new runs.** Only existing logs are read. No code path of the controller changes.

## Data (existing logs, N=100 amend11 config, d_safe 0.55, CPU, seeds 0–4)

| controller | arm | logs |
|---|---|---|
| ON (terminal cost = goal potential, amend36) | fresh | results/amend36_terminal_goal/fresh/fresh |
| ON | noise_only | results/amend36_terminal_goal/noise_only/noise_only |
| ON | fresh_nomfg (CBF-only) | results/amend36_terminal_goal/fresh_nomfg/fresh |
| OFF (terminal cost 0, amend33) | fresh | results/amend33_n100_baseline_cpu/fresh/fresh |
| OFF | noise_only | results/amend33_n100_baseline_cpu/noise_only/noise_only |
| OFF | fresh_nomfg (CBF-only) | results/amend33_n100_baseline_cpu/fresh_nomfg/fresh |

Trajectory = `pos` (dt_exec 0.01 s) followed by `pos_final` (state at T).

## Metric (fixed now)

- **Settled in goal (window W):** an agent is settled if it is inside the goal disc (‖p − goal‖ ≤ r_goal, same disc as
  every earlier amendment) in **≥ 90 %** of the trajectory samples with t ≥ T − W (including the state at T).
- **Primary window W = 5 s.** Sensitivity windows W = 3 s and W = 10 s.
- **completeness_settled(W)** = fraction of the 100 agents settled. **safe_settled(W)** = fraction settled AND never
  collided (true separation < D_COLLISION = 0.30 m at any step, as amend12). Collision counts are reported unchanged.
- Reported alongside, unchanged: completeness at T (registered amend36 metric), collision pair-steps, ever_collided.
- Statistics: seed-level bootstrap CI95 (10 000 resamples, rng 0, as amend33); paired by seed for contrasts.

## Outcomes (fixed now, before any settled number)

**A. Link 1 ("delay-free works") under the settled metric, per controller, per arm (fresh, noise_only):**
- HOLDS: mean safe_settled(5 s) ≥ 0.99, every seed ≥ 0.98, 0 collision pair-steps in every seed (same thresholds as
  amend36, on the settled metric).
- NOT MET otherwise.

**B. Which controller settles the swarm better (decision rule given by the PI):**
- Contrast Δ = safe_settled(5 s, ON) − safe_settled(5 s, OFF), paired by seed, in fresh and in noise_only.
- **ON > OFF** iff the CI95 lower bound of Δ is > 0 in **both** fresh and noise_only. Then the terminal cost stays and the
  re-baseline goes ahead as planned in amend36.
- **Otherwise OFF ≥ ON:** the terminal cost is reverted (macro_cost.terminal unset), **no re-baseline**; amend11/33/34
  numbers stay the chain's reference. A point estimate of Δ > 0 with a CI that includes 0 counts as OFF ≥ ON (the change
  must earn its place).

**C. Window robustness:** the result is "window-robust" iff the verdicts of A (per controller, per arm) and of B are the
same at W = 3, 5 and 10 s. If any verdict differs between windows, the report says **WINDOW-DEPENDENT**, the 5 s
verdict is still stated, and the decision in B goes to the PI instead of being applied automatically.

**D. Reported, no gate:** MFG vs CBF-only, safe_settled(5 s) fresh − fresh_nomfg, for each controller (does the MFG
still earn its place on the settled metric).

## Output

- Script `experiments/amend37_settled_window.py report` → results/amend37_settled_window/report.json,
  logs/amend37/report.log.
- Review logs/overnight/review_amend37.md with classification and the link of the chain it supports or breaks.
- Then STOP: the queue 28/32/29/30 is reviewed together with the PI.
