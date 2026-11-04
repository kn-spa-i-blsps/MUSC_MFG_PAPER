# REVIEW — amend38: headline A under the settled metric + non-settler diagnosis (log-only)

**Purpose:** do the compensation and layer results survive the honest (settled) metric, and is the delay-free
shortfall (8 % fresh, 20 % noise) a horizon artifact or structural?

**Records:** prereg cf93695 (sha 42318aa5…), committed before any number. Script
experiments/amend38_settled_recompute.py. Numbers in results/amend38_settled_recompute/report.json and
logs/amend38/report.log. No new runs.

## Data limit (read first)

- **The N crossover (amend21/22), the mechanism control (amend24) and the τ/t_slot sweep (amend26) cannot be
  recomputed.** Their raw trajectories were on pod volumes deleted before 2026-10-31. amend27 reused those rows.
- Only per-seed summaries survive. For those cells this review gives an upper bound (A1).
- The exact recompute (A2) is at N = 100 in the stage-1 config: amend11 geometry, d_safe 0.55, terminal cost 0. This is
  where local trajectories exist: the amend11 ladder, amend33 and the amend34 layer split.

## A2 — exact, N = 100 (5 seeds; settled = inside ≥ 90 % of the last 5 s)

| arm | compl at T | compl settled | safe at T | **safe settled 5 s** (3 s / 10 s) | ever collided |
|---|---|---|---|---|---|
| real (delayed) | 0.718 | **0.426** | 0.000 | **0.000** | 1.000 |
| cv | 0.730 | 0.440 | 0.000 | 0.000 | 1.000 |
| learned | 0.678 | 0.634 | 0.168 | **0.154** (0.160 / 0.144) | 0.734 |
| smith | 0.036 | 0.036 | 0.004 | 0.004 | 0.996 |
| noise_only (delay removed) | 0.862 | 0.800 | 0.862 | **0.800** | 0 |
| fresh | 0.934 | 0.920 | 0.934 | **0.920** | 0 |
| fresh_nomfg (CBF only) | 0.602 | 0.602 | 0.602 | 0.602 | 0 |
| fresh_nocbf (MFG only) | 1.000 | 1.000 | 0.002 | 0.002 | 0.998 |
| split: MFG clean, CBF delayed | 0.744 | 0.452 | 0.000 | **0.000** | 1.000 |
| split: MFG delayed, CBF clean | 0.856 | 0.802 | 0.856 | **0.802** | 0 |

**Safety is unchanged and identical.** For all 51 rollouts, the collision pair-steps and the completeness at T
recomputed from the npz equal the registered metrics_seed*.json values exactly. ever_collided does not depend on the
metric.

**Relative claims (paired, CI95):**
- **learned − real:** +0.168 [+0.118, +0.226] at T → **+0.154 [+0.106, +0.208]** settled (3 s +0.160, 10 s +0.144).
  **HOLDS relatively.**
- **Layer decomposition: HOLDS identically.**
  - CBF-layer compensation (MFG delayed, CBF clean) − real: +0.856 at T → **+0.802** settled. This equals noise_only
    (0.800), so the CBF layer recovers the whole gap.
  - MFG-layer compensation (MFG clean, CBF delayed) − real: **+0.000** under both metrics.
  - The amend27 finding stands: compensation fed to the MFG layer is worth exactly 0.
- **MFG vs CBF-only** (fresh − fresh_nomfg): +0.332 → **+0.318**. HOLDS.

**How much absolute completeness drops (at T → settled):**
- Clean arms lose little: fresh −0.014, noise_only −0.062, learned −0.044.
- **Delayed arms lose about 0.29:** real 0.718 → 0.426, cv −0.29, MFG-clean split −0.29.
- In the delayed arms almost half the drones counted as "arrived" at T were churning in and out of the disc. They are
  pushed around by collisions and stale CBF rows.
- So the old completeness numbers for delayed arms were strongly inflated. The safe-success numbers barely move,
  because collisions already zeroed them.

## A1 — bound for the lost cells (safe_settled ≤ 1 − ever_collided)

- **Crossover (amend22):**
  - Registered safe-N: real 25, learned 50.
  - Under the settled metric, safe-N can only stay or drop. The collision criterion is unchanged and the completeness
    criterion can only get stricter.
  - Bounds at the cells that decide it:

    | cell | bound |
    |---|---|
    | real N25 | 1.000 |
    | learned N25 | 1.000 |
    | learned N50 | 0.992 |
    | real N50 | 0.284 |

  - Real safe-N cannot rise above 25.
  - Learned safe-N = 50 survives **iff** learned settled completeness at N25 and N50 is ≥ 0.9. This **cannot be
    decided** without trajectories.
  - In that geometry, completeness at T was 1.000 in these cells and the swarm finished by about 32 s. A clean pass is
    plausible but **unverified**.
- **τ sweep (amend26):** no per-agent data and no ever_collided rows in the needed form. **Not recomputable, not bounded.**
- Exact numbers for the crossover and the τ sweep need re-runs of those cells (a decision for the PI, not proposed here).

## B — who does not settle (amend33, terminal 0, 5 s window)

| class | fresh (40 = 8 / run) | noise_only (100 = 20 / run) |
|---|---|---|
| oscillating (≥ 2 in/out crossings) | **20** (50 %) | **59** (59 %) |
| parked, MFG-stalled (u_nom not inward) | **12** (30 %) | 0 |
| parked, CBF-blocked (u_nom inward, u ≈ 0) | 3 (8 %) | 5 (5 %) |
| parked, other (u_nom inward, u partly cut) | 0 | 22 (22 %) |
| in transit (net approach > 0.30 m) | 5 (12 %) | 10 (10 %) |
| other | 0 | 4 (4 %) |

**Where they are:**
- Every fresh non-settler ends within **0.30 m of the rim**: d(T) from −0.42 to +0.30 m, median +0.18 m.
- Noise_only non-settlers end within 0.52 m, median +0.15 m.
- The rim is r_goal = 4.20 m. None of them is far away, stuck at the gate or deadlocked.

**Mechanism (medians of the radial command over the window, < 0 = inward):**
- **fresh:**
  - The oscillators (u_nom −0.03 m/s) and the MFG-stalled drones (u_nom −0.02 m/s) get **almost no inward push from the
    MFG** at the rim. That is 32 of 40.
  - So the swarm's outer ring sits on the boundary, where the goal attraction and the crowd-aversion term balance.
  - Only 3 are blocked by the CBF.
  - The 5 "in transit" drones are already inside at T: they are late arrivals, not drones still approaching.
- **noise_only:**
  - The MFG does push inward: oscillators −0.40 m/s, parked-other −0.28 m/s.
  - The filtered command is cut to −0.13 / −0.10 m/s. The oscillators cross the rim a median of 7 times in 5 s.
  - This is noise-widened CBF spacing making the outer ring jitter, consistent with mechanism 2 in
    review_goal_ceiling.md.

**Registered reading: "mixed"** (in transit 12 % < 50 %, parked 38 % < 50 %).
- The thresholds were written for "stuck far away" versus "still approaching". The data show a third case: a ring of
  drones sitting on the boundary.
- **The horizon is not the limit.** At most 12 % of non-settlers are approaching, and those are already inside at T.
- A longer horizon is **not expected to move the baseline toward 0.98**. The shortfall is structural: the outer ring of
  the equilibrium formation sits on the disc edge.
  - fresh: the MFG potential balances at the rim.
  - noise: CBF jitter at the rim.

## Classification

- **Link 2/3 ("delay damages", "compensation restores"):**
  - The learned-vs-real gap and the layer decomposition **HOLD** under the settled metric at N = 100.
  - Safety is identical.
  - The thesis-level amend27 finding (MFG-layer compensation = 0) is re-confirmed.
- **Headline crossover and τ sweep: UNVERIFIABLE from logs.** The data are lost. Bound given; learned safe-N 50 is
  plausible, not confirmed.
- **Absolute completeness of the delayed arms: about 0.29 lower than reported at T.** Earlier completeness numbers for
  real/cv/delayed arms overstate arrival.
- **Link 1, THESIS-LEVEL (rule 2):** the delay-free shortfall is **structural, not horizon**. In fresh, 8 drones per run
  form a ring on the boundary of the goal disc, at a median of 0.18 m outside. The MFG gives them almost no inward
  push there.
- Whether that counts as "not reaching the goal" is a question about the success bar, not the controller.

## Options for the PI (not run, not registered)

- **Reframe the bar.** The non-settlers are a boundary ring 0.1–0.3 m outside a 4.2 m disc. That is about one drone
  body. A goal tolerance of one drone body, or a goal defined by occupancy, would count them.
  - This is a metric change, so it needs your explicit order (rule 5). It can be computed from these same logs.
- **Or treat it as a controller property.** Then the honest statement is "the MFG equilibrium fills the goal up to its
  boundary; ~8 % of the swarm rests on the rim". Link 1 is then stated at 0.92 with exact positions and 0.80 with
  sensor noise.
- A longer-horizon run is **not recommended**. The data do not point to the horizon.

**STOP.** Decision on reframe vs horizon, plus the queue review 28/32/29/30, together with the PI.
