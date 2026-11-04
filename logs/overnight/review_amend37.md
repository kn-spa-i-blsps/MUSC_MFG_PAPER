# REVIEW — amend37: settled-window success metric, log-only (stage 1, link 1 "delay-free works")

**Purpose:** decide whether "delay-free works" holds when success means a drone has settled in the goal, and which
terminal cost (amend36 ON vs amend33 OFF) settles the swarm better.

**Records**
- Prereg prereg/prereg_projectA_amend37_settled_window.md (sha d4aa87ec…), committed fe5db70 before any settled number
  was computed. The PI asked for the metric (rule 5).
- Script experiments/amend37_settled_window.py. Numbers: results/amend37_settled_window/report.json,
  logs/amend37/report.log.
- No new runs. Existing logs only: N=100 amend11 config, d_safe 0.55, CPU, seeds 0–4.
- Metric: an agent is settled if it is inside the goal disc in ≥ 90 % of the samples of the last W s.
  safe_settled = settled and never collided. W = 5 s primary, 3 s and 10 s sensitivity.

## Result (safe_settled, mean over 5 seeds; collisions 0 pair-steps in every cell)

| controller | arm | at T (amend36 metric) | settled 3 s | **settled 5 s** | settled 10 s |
|---|---|---|---|---|---|
| ON (terminal = goal) | fresh | 0.994 | 0.830 | **0.814** | 0.800 |
| ON | noise_only | 0.888 | 0.724 | **0.702** | 0.676 |
| ON | CBF-only | 0.594 | 0.594 | 0.594 | 0.594 |
| OFF (terminal 0) | fresh | 0.934 | 0.930 | **0.920** | 0.900 |
| OFF | noise_only | 0.862 | 0.826 | **0.800** | 0.756 |
| OFF | CBF-only | 0.602 | 0.602 | 0.602 | 0.602 |

**ON − OFF, paired by seed, safe_settled 5 s [CI95]:**
- fresh −0.106 [−0.116, −0.096]
- noise_only −0.098 [−0.124, −0.074]
- At 3 s and 10 s the signs and the CIs are the same (fresh −0.100 and −0.100, noise_only −0.102 and −0.080, all CIs below 0).

**MFG vs CBF-only, safe_settled 5 s:**
- OFF +0.318 [+0.274, +0.370]
- ON +0.220 [+0.164, +0.280]
- Positive at every window.

**Window robustness:** every verdict is identical at 3, 5 and 10 s, so the result is window-robust.

## Registered verdicts

- **B (decision): OFF ≥ ON → revert the terminal cost, no re-baseline.**
  - ON is not better. It is **worse by about 10 drones in 100** in both arms, at every window.
  - The amend36 0.994 was the last-second rush flagged in review_amend36. On a settled metric it vanishes.
  - amend11/33/34 numbers stay the chain's reference. `macro_cost.terminal` is left unset in every future cell. The code
    path stays in the repo (absent key = old graph, bit-equal).
- **A (link 1): NOT MET, for both controllers, both arms, every window.**
  - The best case is OFF fresh: 0.920 settled. About 8 drones in 100 do not settle in the goal even with exact
    positions and no delay.
  - The realistic delay-free arm (OFF noise_only) settles 0.800.
- **D:** the MFG still earns its place over CBF-only on the settled metric (+0.32 with OFF), at zero collisions.
  CBF-only is unaffected by the window: its drones are frozen, not moving in and out.

## Classification

- **THESIS-LEVEL (rule 2): link 1 "delay-free works" does not hold on an honest metric, with either controller.**
  - Success in the vision is "every drone reaches the goal and never collides".
  - The never-collides half holds: 0 collisions everywhere.
  - The reaches-the-goal half fails without any delay: 0.92 with exact positions, 0.80 with realistic sensor noise.
  - The chain's later links (delay damages, compensation restores) are measured against this baseline. Their meaning
    depends on what the baseline is.
- **Terminal cost (amend36): reverted** by the registered rule. It was a principled fix for one mechanism (the
  end-of-horizon dropout). It replaced that mechanism with a worse one: drones wait at the rim and do not settle.
- **MFG earns its place:** holds (+0.32 settled over CBF-only).

## What this does not decide (for the PI, not proposed as a run)

- Why ~8 % (fresh) and ~20 % (noise) of drones do not settle with the OFF controller. review_goal_ceiling.md named
  mechanism 2: CBF-limited compaction at the rim, wider under noise. That diagnosis was made at the instant T. It has
  not been checked on the settled metric.
- Whether the horizon T = 70.4 s, the goal disc size, or the MFG crowd-aversion itself (which pushes agents away from a
  dense goal) is the limit. Each of these is a design question for the PI, not something to tune past the gate (rule 3).

**STOP.** The queue 28/32/29/30 is reviewed together with the PI, against this result.
