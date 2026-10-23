# Amendment 15 — Density sweep (the money plot), physically grounded

**Base:** prereg_projectA.md + amend9 (sparse-goal safe config) + amend11 (full ladder) +
amend13 (safe-success threshold curve). (amend14 = t_slot sweep, drafted then superseded by this
amendment; intentionally skipped — no amend14 run exists.)
**Status:** frozen, pre-run. Committed with SHA before any run; results added via `git -f`.
**Code change (single, declared):** goal geometry is **pinned** (see §3) — gate position, goal
x-offset, far-chamber size and horizon are fixed across all cells; only the goal-disk radius r_goal
(hence density) varies. This removes the path-length/horizon confound that a density-dependent
geometry would introduce. D_COLLISION is handled purely in analysis (`--d_collision`), no controller
or rollout change. `log_xhat: false` — estimation-error numbers in §2 are taken from amend11, not
re-logged here.

---

## 1. Why this sweep

The thesis result is a **safe-density law**: the maximum density at which the swarm flies
*collision-free* is bounded by perception quality, and delay-compensation raises that bound.
amend13 measured safe-success at **one** density (goal spacing 0.80 m) across the collision
threshold. This amendment sweeps the **density itself** at a fixed, physically-grounded collision
distance, producing the headline curve: **safe-success vs operating spacing, per arm.** It is the
calibration axis for the later staleness sweeps (τ and N both vary T_frame = N·t_slot): it fixes the
operating point at which the arms separate.

## 2. Physical grounding (drone size sets D_COLLISION, radio sets the error)

- **D_COLLISION = 0.30 m — a settable parameter tied to the real drone.** For a 5-inch quad
  (wheelbase ~0.22 m, prop radius 0.0635 m) two aircraft make prop contact at center-to-center
  ≈ 0.28 m side-by-side, ≈ 0.35 m worst-case diagonal; 0.30 m is the realistic two-drone collision
  distance (the **sum** of the two drones' collision radii; a single drone's enclosing radius is
  ~0.15–0.17 m). A 3-inch class would set D_COLLISION ≈ 0.15–0.18 m. Exposed so the law restates for
  any airframe.
- **Perception error is fixed in METERS by the radio/protocol, not by drone size:** range noise
  σ_r = 0.10 m (UWB-realistic), plus staleness displacement V·T_frame, T_frame = N·t_slot (0.5 s at
  N=100, t_slot=5 ms). **Measured closed-loop estimation error (RMS of x̂ error on close pairs,
  mean over seeds 0–4, from amend11):**
  - learned: **0.17 m**
  - real:    **0.82 m**
  - cv:      **0.94 m**
  So the neighbour's used position is off by ~0.6–3 body-lengths of a 0.30 m drone — this is *why*
  tight safe flight is impossible without compensation, and why it is relatively **worse for smaller
  drones** (fixed metre-error = more body-lengths).
- **Predicted safe-spacing law (threshold ≈ D_COLLISION + arm error):**
  - learned ≈ 0.30 + 0.17 ≈ **0.47 m** — inside the swept range, locatable.
  - real ≈ 0.30 + 0.82 ≈ **1.12 m**, cv ≈ 0.30 + 0.94 ≈ **1.24 m** — **beyond** the feasible arena
    range; reported as lower bounds, not located (see H1). That real/cv would need ~1.1–1.2 m
    (≈ 3.7–4.1 body-lengths) to fly safely is itself the headline cost of no compensation.

## 3. Design

- **Fixed:** scenario = far-goal single gate; N = 100; D_COLLISION = 0.30 m (analysis parameter);
  d_safe (CBF) = 0.55 m, **identical across all arms** (arm differences come purely from estimation
  quality feeding the CBF, not a per-arm margin); conical V_goal, no hold; C_DENS=1.0; all §3
  controller params from the base prereg. Learned predictor = frozen amend11 checkpoint
  (hash logged), no retrain. **All 5 Φ/m\* solves are computed fresh for the pinned geometry** — the
  amend11 solve cache is NOT reused, because with pinned geometry no cell matches the amend9/amend11
  configuration (there is no s=0.80 cell, and the arena is the large pinned one). The amend11 anchor
  is used only as a decoupled integrity check (§7), from amend11's own logs.
- **Pinned geometry (the one code change):** gate x, goal x-offset, far-chamber dimensions and
  horizon are all pinned to the **sparsest cell's** requirement (s = 1.05, r_goal = 5.51 m) so every
  cell runs in an **identical, fixed** arena — **≈ 29 m (length) × 13.4 m (height), horizon ≈ 87 s**
  — with identical path length and horizon. Only the goal-disk radius r_goal varies between cells.
  Exact dimensions confirmed from env/scenarios.py and logged at run time.
- **Swept — operating spacing s (the law's variable), with hex mapping ρ = 2/(√3·s²):**

  | s (m)        | 0.45 | 0.60 | 0.75 | 0.90 | 1.05 |
  | body-lengths | 1.5  | 2.0  | 2.5  | 3.0  | 3.5  |
  | ρ (hex)      | 5.70 | 3.21 | 2.05 | 1.43 | 1.05 |
  | r_goal (m)   | 2.36 | 3.15 | 3.94 | 4.73 | 5.51 |

  r_goal = √(N/(π·ρ)) = 5.25·s. Grid brackets learned's predicted 0.47 m threshold (0.45 below,
  0.60–0.75 through, 0.90–1.05 above). real/cv thresholds (1.12/1.24) sit above the grid by design —
  pushing the grid there balloons the arena/path for no added insight.
- **Arms:** real, cv, learned, smith. **Seeds:** 0–2 (3), seed-level cluster bootstrap. First pass
  to establish curve shape and range; extended to 5 seeds after the shape is confirmed (Gate 1).
- **Compute:** 5 spacings × 4 arms × 3 seeds = **60 rollouts** + 5 fresh Φ/m* solves (one per
  density, pinned geometry, cached across arms/seeds). **Runs on CPU** — amend11 (20 rollouts) took
  ≈ 1 h 45 on CPU, so 60 rollouts ≈ **6–7 h** (plus the pinned arena's slightly longer horizon) +
  solve time; fits a single night without GPU. GPU is optional (faster if available), not required.

## 4. Metrics (report BOTH — safety is not bought by emptying the goal)

Per (spacing, arm), full horizon:
1. **Safe-success** = fraction reaching r_goal **and never true d < D_COLLISION** (0.30). Primary.
2. **Completeness** = fraction reaching r_goal (regardless of collisions). Guards against winning
   safety by not filling. (Safe-success is upper-bounded by completeness — at s=0.80 learned
   completeness was 0.678, so the 0.47–0.60 safe-success band is only reachable if completeness
   holds as the arena grows; report both so the ceiling is visible.)
3. **% agents ever d < D_COLLISION** and **worst-case p5 true separation.**
4. **CBF intervention + QP-infeasible/slack rate** (checks the over-subscription regime as s → d_safe=0.55).

## 5. Pre-registered hypotheses (and what would falsify the thesis)

- **H1 (ceiling, asymmetric locatability):** safe-success rises with spacing; **learned has a
  locatable threshold inside the grid (predicted ≈ 0.47 m)**; real and cv stay ≈ 0 across the whole
  grid, their thresholds (≈ 1.1–1.2 m) reported as **lower bounds** beyond the feasible range.
  Learned's threshold is strictly tighter than real's/cv's.
- **H2 (law quantitative):** learned's located threshold tracks D_COLLISION + 0.17 ≈ 0.47 m; learned
  safe-success climbs from ≈ 0 near s≈0.45 to its completeness-bounded ceiling by s≈0.9–1.05.
- **H3 (joint regime):** there is a spacing where learned holds **both** completeness (≈ amend9's
  0.68) **and** materially higher safe-success than real at the **same** spacing.
- **Falsifiers (reported if they occur):** (a) learned's threshold is not meaningfully tighter than
  real's → compensation does not raise the ceiling, thesis fails; (b) no spacing keeps both
  completeness and safety for learned → method cannot fly dense safely, result is at most negative;
  (c) ordering non-monotone or seed-unstable → scenario too noisy for a law.

## 6. Known confound (stated up front)

The **path-length/horizon confound is eliminated** by the pinned geometry (§3) — every cell shares
arena, path and horizon. What remains is the one-sided inflow **crescent**: realized local density
at the goal is set by the inflow pile-up, so nominal spacing under-states true peak density (why
learned is already at 0.168 at s=0.80). The sweep therefore measures the ceiling **as realized under
the crescent** — conservative relative to clean formation flight. The crescent is an accepted
limitation of one-sided goal-filling, not a thesis claim; a clean formation/transit scenario (no
goal pile-up) is the follow-up that would measure the idealized law.

## 7. Consistency checks (run and reported, not optional)

1. **Pipeline integrity (decoupled anchor):** learned safe-success recomputed from amend11's own
   logs must be ≈ **0.168** (the amend13 point: s=0.80, c=0.30). This is NOT a sweep cell (the sweep
   geometry is pinned and matches no prior run); it only proves the analysis/pipeline did not break.
   Deviation > ~0.05 → stop and debug before trusting the sweep.
2. **"0.57" recovery (margin identity):** the margin s − c = 0.65 in amend13 (c=0.15, s=0.80) gave
   learned 0.57. The same margin at the honest c=0.30 is s=0.95, so learned safe-success should
   **bracket ~0.57 between the s=0.90 cell (margin 0.60) and the s=1.05 cell (margin 0.75).**
