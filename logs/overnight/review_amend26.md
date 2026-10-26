# REVIEW — amend26 (pure-staleness sweep via TDMA slot length, N=100 fixed, Phase 4)

Run: aneks sha256 d3afb507e4c1d76e9ea3f1414017f076f892e106bedd6f9e397b4412a95e8af9 (prereg commit 9fc6bf5, before any
run). Result commit ac32b09 (DONE, 120/120). Pod xby8jnv4zqwvn2.
- **Fixed for the whole sweep:** N=100, amend21 arena with gate 2.309, one cache (14d351fc8e), d_safe 0.70 for learned,
  free-area density 0.129 agents/m², min initial sep 0.788.
- **Swept:** t_slot ∈ {2.5, 5, 7.5, 10} ms, so T_frame ∈ {0.25, 0.5, 0.75, 1.0} s. This is the cell key t_slot, logged
  per run.

## Data (mean [95% CI], 10 seeds, precision=highest)

| T_frame | arm | safe-success | ever<0.30 | completeness | p5 sep | slack | x̂ MSE near (pair-w.) |
|---|---|---|---|---|---|---|---|
| 0.25 s | real 0.55 | 0.000 | 1.000 | 1.000 | 0.075 | 0.105 | 0.053 |
| 0.25 s | real 0.70 | 0.000 | 1.000 | 1.000 | 0.076 | 0.218 | 0.118 |
| 0.25 s | learned 0.70 | 0.986 [0.976,0.996] | 0.014 [0.004,0.024] | 1.000 | 0.493 | 0.226 | 0.011 |
| 0.5 s | real 0.55 | 0.000 | 1.000 | 1.000 | 0.032 | 0.159 | 0.459 |
| 0.5 s | real 0.70 | 0.000 | 1.000 | 0.994 | 0.035 | 0.234 | 0.883 |
| 0.5 s | learned 0.70 | 0.892 [0.862,0.919] | 0.072 [0.046,0.102] | 0.958 | 0.464 | 0.192 | 0.023 |
| 0.75 s | real 0.55 | 0.000 | 1.000 | 0.995 | 0.027 | 0.170 | 1.188 |
| 0.75 s | real 0.70 | 0.000 | 1.000 | 0.969 | 0.029 | 0.234 | 2.100 |
| 0.75 s | learned 0.70 | 0.545 [0.498,0.587] | 0.455 [0.413,0.502] | 1.000 | 0.322 | 0.184 | 0.031 |
| 1.0 s | real 0.55 | 0.000 | 1.000 | 0.977 | 0.020 | 0.174 | 2.146 |
| 1.0 s | real 0.70 | 0.000 | 1.000 | 0.917 | 0.023 | 0.231 | 3.638 |
| 1.0 s | learned 0.70 | 0.178 [0.160,0.195] | 0.822 [0.805,0.840] | 1.000 | 0.172 | 0.195 | 0.048 |

Learned − real 0.70 safe-success:

| T_frame | difference [95% CI] |
|---|---|
| 0.25 s | +0.986 [0.976, 0.996] |
| 0.5 s | +0.892 [0.862, 0.919] |
| 0.75 s | +0.545 [0.498, 0.587] |
| 1.0 s | +0.178 [0.160, 0.195] |

Open-loop check (near-pair x̂ MSE, each t_slot's real histories):

| T_frame | real | learned v3 | ratio |
|---|---|---|---|
| 0.25 s | 0.052 | 0.035 | 0.68 |
| 0.5 s | 0.449 | 0.235 | 0.52 |
| 0.75 s | 1.183 | 0.486 | 0.41 |
| 1.0 s | 2.120 | 0.678 | 0.32 |

There is no OOD flag.

## Q3 — decomposition against amend22 (same T_frame, different N/density; descriptive, as pre-registered)

| T_frame | learned ever<0.30, amend26 (N=100) | learned ever<0.30, amend22 (N) | learned completeness, amend26 / amend22 | real 0.70 ever<0.30, amend26 / amend22 |
|---|---|---|---|---|
| 0.25 s | 0.014 [0.004,0.024] | 0.008 [0.000,0.020] (N=50) | 1.000 / 1.000 | 1.000 / 0.944 |
| 0.5 s | 0.072 [0.046,0.102] | 0.076 [0.048,0.106] (N=100) | 0.958 / 0.950 | 1.000 / 1.000 |
| 0.75 s | 0.455 [0.413,0.502] | 0.473 [0.433,0.511] (N=150) | **1.000 / 0.857** | 1.000 / 1.000 |
| 1.0 s | 0.822 [0.805,0.840] | 0.827 [0.807,0.846] (N=200) | **1.000 / 0.599** | 1.000 / 1.000 |

## Step A — mechanism
- **Learned's collision rate is a function of T_frame alone.** At equal T_frame, learned's ever<0.30 at fixed N=100
  matches amend22's value at N = T_frame / 5 ms within CI at all four points, even though N, packing and the
  per-agent neighbour count differ by up to 2×. The same holds for learned's x̂ MSE: 0.011 / 0.023 / 0.031 / 0.048
  here vs 0.023 / 0.031 / 0.050 at N = 100 / 150 / 200 in amend23/22. This independently confirms amend24 (removing
  the delay removes all collisions): learned collisions are caused by the residual delay-estimation error, which
  grows with T_frame.
- **Completeness responds to density, not staleness.** At fixed N=100, learned completeness stays at 0.96–1.00 up to
  T_frame 1.0 s. At N = 150 / 200 it falls to 0.857 / 0.599. The completeness loss of the N crossover is therefore a
  density/congestion effect, while the collision rate is a staleness effect. This splits amend21's stated
  density–staleness confound.
- **Real collides at N=100 even at T_frame 0.25 s** (ever 1.000), while at N=50 with the same T_frame it reached
  0.716 (d 0.55) and 0.944 (d 0.70). For uncompensated estimates, more neighbours at the same staleness still adds
  collisions: more near pairs, each with the same stale error. Real is dead from T_frame 0.25 s at N=100 either way.

## Step B — artifact scan (§3)
1. **Identity:** PASS. t_slot 5 ms real_d0.55 equals amend21 N=100 real on 7 metrics in all 10 seeds, so the explicit
   key t_slot = 0.005 changes nothing. t_slot 5 ms learned reproduces amend23 and amend25 to every printed digit.
2. **Precision:** PASS. All 120 rows are "highest".
3. **OOD:** PASS. The open-loop ratio is 0.32–0.68 and monotone in T_frame. v3 was trained only at t_slot 5 ms but
   over T_frame 0.125–1.0 s; the check confirms the slot-phase lattice change does not hurt.
4. **Wrong knob:** PASS. t_slot moves learned's ever<0.30 from 0.014 to 0.822, so it is the controlling variable (in
   contrast to the gate in amend25).
5. **Margin starvation:** n/a (constant margins).
6. **Geometry/density:** PASS. This is the clean, fixed-density sweep by design.
7. **Statistics:** PASS. Learned − real differences exclude 0 at every T_frame. The Q3 comparison is descriptive, and
   the per-point CIs overlap in every case, i.e. they are statistically indistinguishable.
8. **Completeness-as-sham:** PASS. Learned completeness is ≥ 0.958 throughout, so the collision rise is not hidden by
   freezing.

## Step C — consistency
- **Consistent with:** amend24 (mechanism), amend25 (gate is not the knob), amend22/23 (reproduced N=100 point).
- **Resolves the §0 caveat** that amend21/22 couple staleness and density:
  - for learned, collisions track staleness and completeness tracks density;
  - for real, both matter, and it is dead either way from N=50 or T_frame 0.25 s at N=100.

## Step D — statistics
Arm differences are 0.18–0.99 against CI widths ≤ 0.09, so they are resolved. The Q3 equalities are within CI
(differences ≤ 0.018 in ever<0.30).

## Step E — verdict
- **CONFIRMED:**
  - **Pure staleness alone** (fixed N, density and geometry) drives learned's safety from 0.986 to 0.178 as T_frame
    goes from 0.25 to 1.0 s. Real is dead at every T_frame at N=100. Compensation helps at every T_frame
    (+0.18 to +0.99).
  - **Decomposition:** learned's collision rate in the N crossover is explained by T_frame (staleness), quantitatively,
    at all four matched points. The completeness loss at N ≥ 150 is a density effect.
- **No quarantine.** Fix list: none. This strengthens the case for better compensation at large T_frame (v4, running).
