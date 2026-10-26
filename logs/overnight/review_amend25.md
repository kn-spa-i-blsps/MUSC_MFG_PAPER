# REVIEW — amend25 (operating-density sweep via gate width, N=100, Phase 3)

Run: aneks sha256 0b7498fa11dbad6750cdb966152ace4b9a0e7869086c8061af6099e36a3481c7 (prereg commit 9fc6bf5, before any
run). Result commit 1ce2b21 (DONE, 150/150).
- **Pod:** NEW pod xby8jnv4zqwvn2 (A100-SXM4-80GB). The run was redone from scratch after the RunPod balance stop;
  outputs of the first, unreported attempt on biiz4t7r00oef8 were not used.
- **Geometry:** amend21 arena 43.379 × 19.081 m, N=100, min initial sep 0.788, free-area density 0.129 agents/m² at
  every gate. Gate width ∈ {1.5, 2.309, 3.5, 5.0, 7.0} m, i.e. gate flux N/gate = 66.7 / 43.3 / 28.6 / 20.0 / 14.3
  agents per m.
- **Caches:** one per gate, each converged without NaN (keys 11fc7f3ae5 / 14d351fc8e / a6378a47f8 / fdbf07e4f1 /
  6fbb7e451f).

## Data (mean [95% CI], 10 seeds, precision=highest)

| gate [m] | arm | safe-success | ever<0.30 | completeness | p5 sep | interv. | slack | x̂ MSE near (pair-w.) |
|---|---|---|---|---|---|---|---|---|
| 1.5 | real 0.55 | 0.000 | 1.000 | 1.000 | 0.025 | 0.592 | 0.166 | 0.476 |
| 1.5 | real 0.70 | 0.000 | 1.000 | 0.994 | 0.031 | 0.705 | 0.237 | 0.886 |
| 1.5 | learned 0.70 | 0.874 [0.853,0.893] | 0.126 [0.107,0.147] | 1.000 | 0.407 | 0.852 | 0.176 | 0.022 |
| 2.309 | real 0.55 | 0.000 | 1.000 | 1.000 | 0.032 | 0.586 | 0.159 | 0.459 |
| 2.309 | real 0.70 | 0.000 | 1.000 | 0.994 | 0.035 | 0.705 | 0.234 | 0.883 |
| 2.309 | learned 0.70 | 0.892 [0.862,0.919] | 0.072 [0.046,0.102] | 0.958 [0.949,0.967] | 0.464 | 0.903 | 0.192 | 0.023 |
| 3.5 | real 0.55 | 0.000 | 1.000 | 0.999 | 0.044 | 0.571 | 0.148 | 0.423 |
| 3.5 | real 0.70 | 0.000 | 1.000 | 0.994 | 0.042 | 0.698 | 0.227 | 0.851 |
| 3.5 | learned 0.70 | 0.928 [0.907,0.949] | 0.072 [0.051,0.093] | 1.000 | 0.483 | 0.839 | 0.144 | 0.023 |
| 5.0 | real 0.55 | 0.000 | 1.000 | 0.999 | 0.042 | 0.568 | 0.146 | 0.422 |
| 5.0 | real 0.70 | 0.000 | 1.000 | 0.997 | 0.041 | 0.697 | 0.224 | 0.838 |
| 5.0 | learned 0.70 | 0.885 [0.862,0.908] | 0.107 [0.086,0.128] | 0.991 | 0.468 | 0.877 | 0.160 | 0.023 |
| 7.0 | real 0.55 | 0.000 | 1.000 | 1.000 | 0.042 | 0.573 | 0.150 | 0.434 |
| 7.0 | real 0.70 | 0.000 | 1.000 | 0.995 | 0.045 | 0.697 | 0.225 | 0.860 |
| 7.0 | learned 0.70 | 0.927 [0.903,0.950] | 0.072 [0.049,0.096] | 0.998 | 0.471 | 0.881 | 0.167 | 0.022 |

- **Learned − real 0.70, safe-success:** +0.874 / +0.892 / +0.928 / +0.885 / +0.927 across the five gates. All CIs
  exclude 0.
- **Open-loop estimator check** (x̂ MSE on pairs < 1 m, on each gate's real histories):

  | gate [m] | 1.5 | 2.309 | 3.5 | 5.0 | 7.0 |
  |---|---|---|---|---|---|
  | real | 0.465 | 0.449 | 0.412 | 0.408 | 0.435 |
  | learned v3 | 0.242 | 0.235 | 0.220 | 0.218 | 0.230 |
  | ratio | 0.52 | 0.52 | 0.53 | 0.53 | 0.53 |

  There is no OOD flag at any gate.

## Step A — mechanism
- **Gate width is a weak lever at N=100 in this arena.**
  - Real is dead (ever = 1.000) at every gate. Its pair-weighted x̂ MSE barely changes with gate width (0.42–0.48 at
    d 0.55).
  - Learned's x̂ MSE is flat at 0.022–0.023, and its safety moves only within a narrow band (ever 0.07–0.13).
  - This is what the mechanism established in amend24 predicts. Collisions are driven by TDMA-delay estimation error,
    whose size is set by T_frame (N) and agent speeds, not by the gate geometry. A 4.7× change in gate flux does not
    change the estimation error, so it does not change safety much.
- **Second-order effects:**
  - The narrowest gate (1.5 m) is measurably worse for learned than 3.5 and 7.0. Ever is 0.126 [0.107, 0.147] vs
    0.072 [0.051, 0.093] and [0.049, 0.096], so the CIs separate. A plausible cause is more pairs forced into close
    contact at the gate.
  - The 2.309 gate shows lower completeness (0.958) than the others (≥ 0.991). This is the known amend22/23 value; a
    plausible cause is queueing at a mid-width gate.
  - The 5.0 m gate is no better than 2.309 (ever 0.107 vs 0.072, CIs overlap). The effect is non-monotone and lies
    within seed noise.

## Step B — artifact scan (§3)
1. **Identity:** PASS. Gate 2.309 real_d0.55 equals amend21 N=100 real on 7 metrics in all 10 seeds, on a third pod.
   Learned at gate 2.309 reproduces amend23's learned d0.70 N=100 to every printed digit (0.892 [0.862, 0.919],
   0.072 …) on a different pod.
2. **Precision:** PASS. All 150 rows are "highest".
3. **OOD:** PASS. Open-loop ratio is 0.52–0.53 at every gate.
4. **Wrong knob:** FLAG, and it is the finding. Gate width hardly moves ever<0.30 for either arm. Following the brief,
   this means gate width is not the variable that controls collisions here. amend24 already identified that
   variable: delay-estimation error, set by T_frame. The density axis that did matter in §0 (amend20 tight arena
   0.41 vs amend21 sparse arena 0.805) was a different arena and start packing, not the gate. It is therefore not
   tested by this sweep.
5. **Margin starvation:** n/a (constant margins).
6. **Geometry/density:** PASS. Only the gate width varies; the arena, start box, goal and N are fixed.
7. **Statistics:** PASS for "learned ≫ real at every gate" (all difference CIs exclude 0). For "safety varies with
   gate", only g1.5 vs g3.5/g7.0 separates. The other between-gate differences overlap.
8. **Completeness-as-sham:** PASS. Learned completeness is ≥ 0.958 at every gate.

## Step C — consistency
- Consistent with amend24 (the mechanism is estimation error) and with amend22/23 (same N=100 numbers, reproduced).
- **Refines §0 "density matters as much as N":** within one arena at N=100, gate width (flux) matters little. The
  amend20→21 difference came from the arena and start packing, which this sweep does not vary. An arena-size or
  start-spacing sweep would be needed to test that; it was not run.

## Step D — statistics
10 seeds resolve the arm difference (+0.87–0.93) easily. Between-gate differences for learned are ≤ 0.05 in ever and
mostly within CI. Only the 1.5 m effect is resolved.

## Step E — verdict
- **CONFIRMED:**
  - **The compensation advantage holds across gate widths 1.5–7.0 m at N=100:** learned 0.70 beats real 0.70 by
    +0.87 to +0.93 safe-success at every gate.
  - **Gate width is not a primary collision driver** (wrong-knob finding, mechanism consistent with amend24).
- **Pre-registered Q3 (narrowest gate at which learned is safe):** by the definition, gate 3.5 m is the narrowest
  (ever CI upper 0.093, completeness 1.000). Gate 5.0 fails it (0.128), and 7.0 passes (0.096). The safe/unsafe
  boundary is therefore non-monotone within seed noise. This is NOT a sharp operating-density threshold. Reported as:
  learned is at or near the 0.10 safety line at every gate ≥ 2.3 m; gate 1.5 m is measurably worse.
- **Operating density for the other sweeps:** the amend21 arena with gate 2.309 (as used) is representative. Results
  are insensitive to the gate within 2.3–7 m.
- **No quarantine.** Possible follow-up, not pre-registered: an arena-size / start-spacing sweep for the real density
  axis.
