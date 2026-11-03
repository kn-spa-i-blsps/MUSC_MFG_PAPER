# REVIEW — amend34: through which layer does TDMA delay do its damage? (EXPLORATORY)

**Run**
- CPU (local). N=100, amend11 config, fixed d_safe 0.55, seeds 0–4.
- Aneks prereg/prereg_projectA_amend34_layer_delay_cpu.md (sha 9f2c0642…), committed and pushed in 02cc47a before any
  rollout.
- 12/12 rollouts finished, rc=0. Started 11:08 and ended 13:16 on 2026-11-03, after a first start at 10:22 was stopped
  by the user with no rollout finished.
- Numbers come from results/amend34_layer_delay_cpu/report.json and logs/amend34/report.log.

**Arms**
- **A** = delayed estimates to the MFG density term, delay-free to the CBF.
- **B** = delay-free to the MFG, delayed to the CBF.

Delay-free means the noise-only perception: same sensor noise and same PRNG key, age 0.

**Reference check (seed 0, split code path):**
- `split:real/real` was bit-equal to the amend11 real positions and all 7 metrics.
- `split:noise_only/noise_only` was bit-equal to amend33 noise_only.

So the existing real and noise_only rows are valid paired references.

**Statistics.** Exact two-sided sign-flip test over 2⁵ = 32 assignments. At n = 5 the smallest possible p is 0.0625,
so nothing here can reach p < 0.05. Seed-level bootstrap 95 % interval alongside.

**Disclosure.** While testing the report script, before the commit, I printed the per-seed facts of the two existing
reference rows (real, noise_only). No A or B data existed then. No definition was changed afterwards.

## Per-seed facts

"Not arrived" counts the drones outside r_goal at the end, split as still moving at the horizon / jammed at the gate /
other. Every "other" drone is stationary downstream of the wall, outside the disc. No drone in any arm was jammed at
the gate.

| arm | seed | fleet success | reached-goal | drones ever collided | collision pair-steps | t90 [s] | makespan [s] | not arrived (moving / gate / other) |
|---|---|---|---|---|---|---|---|---|
| real | 0 | 0 | 0.74 | 100 | 44881 | — | 32.07 | 26 (25 / 0 / 1) |
| real | 1 | 0 | 0.68 | 100 | 46165 | — | 26.07 | 32 (32 / 0 / 0) |
| real | 2 | 0 | 0.77 | 100 | 43296 | — | 25.41 | 23 (22 / 0 / 1) |
| real | 3 | 0 | 0.70 | 100 | 45299 | — | 26.56 | 30 (28 / 0 / 2) |
| real | 4 | 0 | 0.70 | 100 | 46271 | — | 24.87 | 30 (30 / 0 / 0) |
| noise_only | 0 | 0 | 0.87 | 0 | 0 | — | — | 13 (9 / 0 / 4) |
| noise_only | 1 | 0 | 0.87 | 0 | 0 | 67.97 | 67.79 | 13 (6 / 0 / 7) |
| noise_only | 2 | 0 | 0.85 | 0 | 0 | 67.59 | — | 15 (5 / 0 / 10) |
| noise_only | 3 | 0 | 0.85 | 0 | 0 | — | — | 15 (4 / 0 / 11) |
| noise_only | 4 | 0 | 0.87 | 0 | 0 | 69.47 | — | 13 (8 / 0 / 5) |
| **A** (delay-free CBF) | 0 | 0 | 0.85 | 0 | 0 | 67.63 | — | 15 (9 / 0 / 6) |
| **A** | 1 | 0 | 0.83 | 0 | 0 | 69.97 | — | 17 (9 / 0 / 8) |
| **A** | 2 | 0 | 0.85 | 0 | 0 | 68.92 | — | 15 (9 / 0 / 6) |
| **A** | 3 | 0 | 0.89 | 0 | 0 | 67.88 | — | 11 (6 / 0 / 5) |
| **A** | 4 | 0 | 0.86 | 0 | 0 | 69.49 | — | 14 (8 / 0 / 6) |
| **B** (delay-free MFG) | 0 | 0 | 0.75 | 100 | 37791 | — | 26.09 | 25 (23 / 0 / 2) |
| **B** | 1 | 0 | 0.76 | 100 | 39109 | — | 28.38 | 24 (23 / 0 / 1) |
| **B** | 2 | 0 | 0.78 | 100 | 38168 | — | 25.31 | 22 (22 / 0 / 0) |
| **B** | 3 | 0 | 0.68 | 100 | 37935 | — | 29.50 | 32 (31 / 0 / 1) |
| **B** | 4 | 0 | 0.75 | 100 | 39544 | — | 30.53 | 25 (24 / 0 / 1) |

**Note on makespan.** Makespan is the time the last drone first enters the disc; t90 is the first time 90 % are
inside it at once. In real and B every drone passes through the disc by 25–33 s, but 22–32 of them are outside it and
still moving at the end. A dash (—) means never reached.

## Paired contrasts

| contrast | metric | mean | 95 % interval | sign-flip p | per seed |
|---|---|---|---|---|---|
| A − real | reached-goal | +0.138 | [+0.102, +0.170] | 0.0625 | +.11 +.15 +.08 +.19 +.16 |
| A − real | drones ever collided | −100 | [−100, −100] | 0.0625 | −100 ×5 |
| A − real | collision pair-steps | −45182 | [−46013, −44187] | 0.0625 | all negative |
| A − noise_only | reached-goal | −0.006 | [−0.028, +0.020] | 0.75 | −.02 −.04 .00 +.04 −.01 |
| A − noise_only | collisions (both) | 0 | [0, 0] | 1.0 | 0 ×5 |
| A − noise_only | t90 | +1.1 s | [+0.02, +2.0] | 0.25 | 3 seeds with t90 in both |
| B − real | reached-goal | +0.026 | [−0.002, +0.054] | 0.3125 | +.01 +.08 +.01 −.02 +.05 |
| B − real | drones ever collided | 0 | [0, 0] | 1.0 | 0 ×5 (100 in both) |
| B − real | collision pair-steps | −6673 (−15 %) | [−7193, −5895] | 0.0625 | all negative |
| B − noise_only | reached-goal | −0.118 | [−0.148, −0.090] | 0.0625 | all negative |

Fleet success is 0 in every seed of every arm, so every fleet-success contrast is 0.

**Recovered fraction of the real → noise_only gap**

| arm | drones ever collided | collision pair-steps | reached-goal |
|---|---|---|---|
| A | 1.00 | 1.00 | 0.96 |
| B | 0.00 | 0.15 | 0.18 |

## Readings (as committed before the run)

- **A close to noise-only on collisions: YES.** A recovers 100 % of both collision measures and 96 % of reached-goal.
  The delay's safety damage goes through the CBF.
- **B moves reached-goal from real toward noise-only: NO** by the registered criterion (5/5 seeds positive).
  - 4 of 5 seeds are positive, mean +0.026, p = 0.31; this is 18 % of the gap.
  - Stale density costs no measurable flow at this n. An MFG-layer predictor has no measurable job on reached-goal
    here.
  - It does have a small, consistent effect on collision pair-steps: −15 %, 5/5 seeds, the same −15 % as in amend27.
    That effect is too small to keep a single drone from colliding.
- **Neither arm alone recovers it: NO.** A alone recovers essentially everything, so the layers do not interact in
  any measurable way here.

## The chain, one sentence per link

1. **Delay-free works: does NOT hold** for the PI's success definition. Without delay the swarm never collides, but
   13–15 of 100 drones end outside the goal disc in every seed. Most of them sit still downstream of the wall, next to
   the disc. So fleet success is 0 even with no delay, and no predictor can lift it above 0.
2. **Delay damages: holds.** With real TDMA delay every drone collides at least once (100/100, all seeds), and
   reached-goal falls from 0.86 to 0.72, because 22–32 drones are still moving at the horizon.
3. **Compensation restores: not tested here.** The only learned data at this config are the October amend11
   rollouts, which are not strictly paired with current code. This run shows only what perfect compensation would do,
   and where.
4. **Each layer earns its place:**
   - **CBF: holds.** Removing delay from the CBF's inputs alone gives back all the safety and 96 % of reached-goal.
   - **MFG density term: does not earn its place against delay.** Removing delay from it alone changes no drone's
     collision status and moves reached-goal by an amount this run cannot tell from zero.

**Thesis level.** The idea's core claim was that delay compensation matters for the MFG density layer. At N=100 on
this config, the damage from TDMA delay goes almost entirely through the CBF, not through the MFG density estimate.
That reproduces amend27's layer split on another geometry, margin and device, and it now covers flow as well as
safety. This stops the work and goes to the PI.

**One next step.** Find out from the existing noise_only and fresh logs (no new runs) why 13–15 drones stop outside
the goal disc with no delay. Until that is understood, the PI's success criterion is unreachable for every arm.
