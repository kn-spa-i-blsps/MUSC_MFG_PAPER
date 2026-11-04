# Amendment 38 — settled-window recompute of headline A + non-settler diagnosis, log-only

**Purpose (one sentence):** find out whether the compensation and layer results survive the honest (settled) metric, and
whether the ~8 % / ~20 % of drones that never settle in the delay-free baseline are a horizon artifact (cheap fix) or
structural (the success bar needs reframing). Both serve link 1 of the chain and the meaning of every later link.

**Status:** REGISTERED before any number below was computed. PI order 2026-11-04 after review_amend37. Metric = amend37
(settled: inside the goal disc in ≥ 90 % of the samples of the last 5 s; safe_settled = settled ∧ never true d < 0.30 m).
No new runs.

## Data availability (checked before registering)

- The raw trajectories of amend21/22 (N crossover), amend24 (noise_only/fresh vs N) and amend26 (τ / t_slot sweep) are
  **gone**: they lived on pod volumes deleted before 2026-10-31 (account had no pods and no volumes). amend27 reused those
  rows for real / learned / noise_only / fresh. Only per-seed summaries (report.json, metrics_seed*.json) are in git.
- Possibly on stopped pod mc4847t30028lq: amend27's own cells (cv, smith, split arms). Without the real/learned base
  rows they cannot rebuild the crossover or the decomposition contrasts, so the pod is not started for this amendment.
- Local raw logs: amend11 ladder N=100 (real / cv / learned / smith, d_safe 0.55, seeds 0–4), amend33 (fresh,
  noise_only, fresh_nomfg, fresh_nocbf), amend34 (split arms noise_only/real, real/noise_only, noise_only/noise_only).

## A — headline under the settled metric

**A1. Bound for the crossover, decomposition and τ sweep (every cell with a per-seed ever_collided):**
safe_settled ≤ 1 − ever_collided per seed (settled ∧ never collided ⊆ never collided). Reported per cell. Under the
amend22 safe-N rule (ever CI-upper ≤ 0.1 and completeness ≥ 0.9 at this and all smaller N), the ever_collided criterion
is unchanged by the metric and the completeness criterion can only get stricter, so safe-N under the settled metric is
≤ the registered safe-N. Whether it is equal cannot be decided without trajectories; this is stated, not estimated.

**A2. Exact recompute at N = 100, stage-1 config (amend11 geometry, d_safe 0.55, seeds 0–4, terminal cost 0):**
- arms: real, cv, learned, smith (amend11 ladder), fresh, noise_only, fresh_nomfg, fresh_nocbf (amend33), split arms
  (amend34).
- per arm: completeness at T, completeness_settled, safe_settled (5 s; 3 s and 10 s reported), ever_collided, collision
  pair-steps.
- **Safety check:** ever_collided and collision pair-steps recomputed from the same npz must equal the registered values
  (amend12/33/34 reports) exactly. Any difference is reported as a defect.
- **Relative claims (no gate, reported):** learned − real and each split contrast on safe_settled vs on the registered
  safe-success; "holds relatively" = same sign and CI excluding 0 on both metrics.
- Completeness drop = completeness at T − completeness_settled, per arm.

## B — non-settler classification (OFF controller = amend33, fresh and noise_only, seeds 0–4)

For each non-settled agent (5 s window), with d(t) = ‖p − goal‖ − r_goal (signed distance outside the disc), window
[T − 5 s, T], crossings = number of inside/outside switches in the window:
1. **oscillating** — crossings ≥ 2.
2. **in transit** — not (1) and net approach d(T) − d(T − 5 s) < −0.30 m (includes late arrivals that enter once).
   For these, report time-to-enter = d(T) / mean approach speed over the last 1 s (∞ if not approaching).
3. **parked** — not (1)/(2), 0 < d(t) ≤ 0.60 m for ≥ 90 % of the window and |Δd| ≤ 0.30 m.
   Sub-split by the cause visible in the logs, over the window:
   - **CBF-blocked**: median radial component of u_nom points inward (< −0.05 m/s) but |u| is small (median radial
     component of u > −0.05 m/s). The MFG asks to go in and the safety filter stops it (packing).
   - **MFG-stalled**: median radial component of u_nom ≥ −0.05 m/s. The MFG itself does not push in (crowd aversion or
     a flat potential).
4. **other** — everything else (far outside and not approaching, leaving), with d(T) and position reported.

**Reading, fixed now:**
- If (2) in transit with finite time-to-enter ≤ 10 s is ≥ 50 % of the non-settlers in fresh, the
  report says "horizon-limited": one run with a longer horizon is a reasonable test.
- If (3) parked is ≥ 50 %, the report says "structural" (named sub-cause), and a longer horizon is not expected to
  help.
- Otherwise "mixed", with the distribution.
No run is started either way. The decision on horizon vs reframe goes to the PI.

## Output

experiments/amend38_settled_recompute.py → results/amend38_settled_recompute/report.json, logs/amend38/report.log,
review logs/overnight/review_amend38.md. Then STOP.
