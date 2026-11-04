# REVIEW — amend36: MFG terminal cost = goal potential (stage 1, link 1 "delay-free works")

**Purpose:** with no delay, find out whether every drone reaches the goal without collisions once the MFG's terminal
cost is the goal potential instead of 0.

**Records**
- Prereg prereg/prereg_projectA_amend36_terminal_goal.md (sha 66fd45f0…), committed f6f2415 before any run.
- Solves 671cb04, rollouts 6770441, report 50cbf6e.
- Numbers: results/amend36_terminal_goal/report.json, logs/amend36/report.log.
- CPU (this workstation), N=100 amend11 config, d_safe 0.55, seeds 0–4. 15 rollouts, all exit 0, wall time 3 h 53 min.
- **RE-BASELINE:** the controller changed. All earlier Project A numbers (amend11, 33, 34) used terminal cost 0 and are
  superseded for the chain. They are recomputed only after the queue review.

## Plumbing

- With the key absent, the old solver path recomputes the cached amend11 Φ gradient bit-equal
  (logs/amend36/hjb_identity.json).
- With the terminal cost, max |∇Φ(T)| is 16.3 (was 0). Everything stays finite.

## Registered result

| arm (delay-free) | completeness, new [per seed] | old (amend33) | Δ new − old [CI95] | collisions | link 1 |
|---|---|---|---|---|---|
| fresh (MFG+CBF) | **0.994** [1.00 .99 .99 .99 1.00] | 0.934 | +0.060 [+0.048, +0.072] | 0 | **HOLDS** |
| noise_only (MFG+CBF, sensor noise) | **0.888** [.88 .90 .87 .89 .90] | 0.862 | +0.026 [+0.016, +0.034] | 0 | **IMPROVED, NOT MET** |
| fresh_nomfg (CBF only) | 0.594 | 0.602 | −0.008 [−0.036, +0.012] | 0 | — |

- **MFG earns its place (completeness axis):** fresh − fresh_nomfg = **+0.400 [+0.346, +0.456]**.
  - CBF-only still leaves 34–49 drones frozen outside.
  - The terminal cost does not help it: its deadlock is not an end-of-horizon effect.
- **Mechanism 1 is gone:** drones leaving the disc in the last 0.2 s went from 2.2 (fresh) and 3.4 (noise_only) per run
  to 0.0.
- Collisions are 0 in every arm and seed. Safe-success equals completeness.

## Flag for the PI (descriptive, not a change to the verdict)

The registered metric counts drones inside the disc **at the instant T**. Looking at the last seconds of the logs (the
same time series as the goal-ceiling diagnosis), fresh reaches 99.4 % by a final-second inflow, not by drones having
settled:

| drones outside the disc at | T−3 s | T−1 s | T−0.5 s | T−0.2 s | T |
|---|---|---|---|---|---|
| fresh, new (seeds 0–4) | 16–20 | 14–18 | 9–14 | 4–6 | 0–1 |
| fresh, old | 5–8 | 2–6 | 3–6 | 3–7 | 4–9 |
| noise_only, new | 25–29 | 24–27 | 18–24 | 13–19 | 10–13 |

- One second before T, the new controller has **more** drones outside than the old one (14–18 vs 2–6). The terminal
  pull (|∇g| up to 16, against a rim pull of ~3 before) then draws them in during the last half second.
- Mean |u| in the last 0.5 s rises from about 0.12 to about 0.4 m/s (fresh) and to about 1.0 m/s (noise_only).
- So the fix removed the end-of-horizon dropout and replaced it with an end-of-horizon rush. Both are artifacts of
  scoring at one instant of a finite horizon.
- My reading, not verified: with a terminal reward for being in the disc at T, the equilibrium has no reason to enter
  the crowded disc early, so drones wait at the rim and enter at the end.

**Why this matters for the chain:** link 1 holds on the registered metric. It does not hold in the sense a reader
would assume ("drones arrive and stay"). Per CLAUDE.md rule 5, I think the success metric, completeness at the instant
T, is the wrong measure for "every drone reaches the goal" under either terminal cost. I have not added or computed a
replacement as a result.

## Classification

- **Link 1 (delay-free works), registered:**
  - fresh: HOLDS.
  - noise_only, the realistic delay-free arm: IMPROVED, NOT MET (0.888).
  - The residual 10–13 drones are mechanism 2: CBF-limited packing at the rim, wider under sensor noise (inside
    nearest-neighbour spacing 0.60–0.62 m vs 0.55–0.57 m exact).
- **THESIS-LEVEL (rule 2):** the fresh HOLDS rests on a last-second inflow that the instant-T metric rewards. Whether
  link 1 really holds depends on a metric decision that is the PI's, not mine.
- **"MFG earns its place":** holds, +0.40 completeness over CBF-only, zero collisions in both.

## One proposed next step (not run)

- PI decision on the success metric: whether "reaches the goal" means inside at T, or inside and staying inside over a
  final window (for example the last 1 s).
- If the latter is registered, recompute it from the existing logs, old and new controller alike. That is log-only, no
  new runs.
- No other change. No tuning of the terminal weight.

**STOP.** Queue 28/32/29/30 stays blocked until it is reviewed with the PI against the realigned chain.
