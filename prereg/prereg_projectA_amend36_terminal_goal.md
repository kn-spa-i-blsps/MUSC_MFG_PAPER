# Amendment 36 — MFG terminal cost = goal potential (stage 1, link 1: "delay-free works")

**Purpose (one sentence):** with no delay, find out whether every drone reaches the goal without collisions once the
MFG's terminal cost is no longer 0, because "delay-free works" is the first link of the chain and it currently fails
(13–15 / 100 drones end outside the disc, logs/overnight/review_goal_ceiling.md).

**Status:** REGISTERED before any rollout. PI decision 2026-11-04 (realign: Phase B closed, back to stage 1). This is
the bug fix named in the goal-ceiling diagnosis, not a new bet.

## The change (controller change → RE-BASELINE)

- The MFG HJB is solved with terminal condition V(T, x) = g(x) = V_goal(x), the scenario's goal potential (cone
  w_goal·r + hold step w_hold), in place of V(T, x) = 0. Weight 1, no new parameter; this is the terminal cost of the
  reference finite-horizon formulation (deviation c in prereg amend34 §5 is removed for the terminal part).
- The terminal condition is used in every HJB solve: in the fictitious-play iterations (so m* is the equilibrium of the
  corrected game) and in the final Φ solve against m*.
- Code: `CostParams.V_terminal` (mfg/hjb.py, separate jitted scan `_scan_hjb_backward_from`), `solve_macro(...,
  terminal="goal")` (mfg/equilibrium.py), config key `macro_cost.terminal: goal` (experiments/smoke.get_macro; it
  enters the cache key only when set, so old keys stay valid).
- With the key absent the old graph runs unchanged. Checked before this commit: the old path recomputes the cached
  amend11 Φ gradient (macro_N100_cw1_13e24892b8) bit-equal (logs/amend36/hjb_identity.json). With the terminal cost,
  max |∇Φ(T)| is 16.3 instead of 0, finite everywhere.
- **Re-baseline:** this changes the controller. Every confirmed Project A number (amend11 real/cv/learned/smith,
  amend33, amend34) was produced with terminal cost 0 and is superseded for the chain. They are recomputed under the
  new controller only after the PI reviews the queue (28/32/29/30) against the realigned chain — not in this
  amendment.

## Design

- Config: N=100 baseline, experiments/ladder_fargoal_amend11.yaml (d_safe 0.55, CPU, default precision), seeds 0–4.
  Not the amend21 crossover geometry.
- Arms (all delay-free), each = the amend33 cell + `macro_cost.terminal: goal` + run_name:
  - `fresh` — MFG + CBF, exact neighbour positions;
  - `noise_only` — MFG + CBF, sensor noise kept, no delay (the realistic delay-free arm);
  - `fresh_nomfg` — CBF-only control (c_w = w_bar = 0, c_dens = 0; HJB routing still solved, with the same terminal
    cost), the anti-deadlock comparison.
- Two new macro solves (c_w = 1 and c_w = 0, terminal goal), then 15 rollouts, 2 in parallel.
- References (old controller, same seeds, same initial positions): amend33 fresh / noise_only / fresh_nomfg.

## Outcomes (fixed now, before data)

**Link 1 verdict**, judged on `fresh` and `noise_only` separately:
- **HOLDS** — mean completeness ≥ 0.99 AND every seed ≥ 0.98 AND collision pair-steps = 0 in every seed.
- **IMPROVED, NOT MET** — collisions 0 in every seed and completeness higher than the amend33 row (paired seed-level
  bootstrap CI of the difference > 0), but the HOLDS bar is missed. The residual is then mechanism 2 (CBF-limited rim
  compaction unfinished at T), reported, not tuned.
- **FAIL** — completeness not higher than amend33 (CI includes or is below 0), or any collision pair-step.

**MFG earns its place (completeness axis):** `fresh` − `fresh_nomfg` completeness and safe-success, paired bootstrap CI
(amend33 Q3), plus the number of drones outside the disc at T in `fresh_nomfg`.

**Descriptive (diagnosis carry-over, no verdict):**
- non-arrivers added in the last 0.2 s (outside at T, inside at T − 0.2 s), per seed — mechanism 1 should go to ~0;
- before/after contrasts vs amend33 for completeness, safe-success, collision pair-steps, makespan, ∫W.

Metrics are the existing ones (completeness_final, collision_pair_steps, amend12 safe_success / ever_collided). Bootstrap:
10k, seed level, n = 5 (amend33 `paired`).

## Rules

- One run, one verdict. No tuning of the terminal weight or anything else after the numbers.
- If link 1 is not met, the review proposes at most one principled next step and STOPs.
- Push after the solves, after the rollouts and after the report (push_lib checkpoint).
- Then STOP: the queue 28/32/29/30 is reviewed with the PI against the realigned chain before any further compute.
