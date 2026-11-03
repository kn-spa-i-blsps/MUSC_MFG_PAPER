# REVIEW — completeness ceiling without delay (log-only diagnosis, DESCRIPTIVE)

**Purpose (one sentence):** find out why 13–15 / 100 drones end outside the goal disc even with no delay, because
"every drone reaches the goal" is unreachable for every arm until this is understood.

**Run.**
- No rollouts, no training, no parameter change. Log-only diagnosis, PI request 2026-11-03.
- Data: existing npz logs of amend33 (fresh, fresh_nocbf, fresh_nomfg, noise_only) and amend34 A (delay-free CBF),
  amend11 config, N=100, d_safe 0.55, seeds 0–4. Macro cache `macro_N100_cw1_13e24892b8`.
- Scripts:
  - experiments/diag_goal_ceiling.py → results/diag_goal_ceiling/diag.json;
  - experiments/diag_goal_ceiling_timeseries.py → timeseries.json;
  - experiments/diag_goal_ceiling_last.py → last_seconds.txt.
- These are descriptive quantities for a diagnosis. None of them is a new success metric.

**Setting.** r_goal = 4.20 m, T = 70.38 s. Success region: inside r_goal at T.

## What it is NOT

- **Not the gate or an inlet deadlock.**
  - 0 non-arrivers upstream of the wall, in every arm and seed.
  - 0 jammed at the gate.
- **Not distance or travel time.**
  - 98–99 % of drones were inside the disc at some time.
  - Mean time to goal is ~20 s (noise_only 20.5 s, fresh 23.6 s) out of 70 s.
- **Not the MFG plan.** m* keeps 99.86 of 100 agents inside the disc from t = 10 s to T.
- **Not a lack of space.** The disc is sized for 100 drones at 0.80 m hex spacing. Without the CBF (fresh_nocbf),
  100/100 end inside, but collapsed onto each other.

## What it is

The non-arrivers are drones that churn on the rim of the disc. They sit 0.02–0.6 m outside it (median 0.15–0.35 m).
- noise_only: 68 of 69 are within 0.5 m of the rim.
- fresh: all are within 0.3 m.
- Almost all were inside earlier and stepped out late: median last time inside 63–70 s.

Two mechanisms, both at the end of the horizon.

**1. The MFG pull switches off in the last half second (terminal-time artifact).**
- The MFG has terminal cost 0, so Φ(T) = 0 and ∇Φ(T) = 0.
- Median |∇Φ| in the rim band (r_goal ± 1 m):

  | time before T | 10 s | 1 s | 0.5 s | 0.2 s | 0.1 s | 0 |
  |---|---|---|---|---|---|---|
  | \|∇Φ\| | 3.3 | 3.3 | 2.9 | 0.9 | 0.5 | 0 |

- At that moment the net radial velocity of rim drones flips from inward to outward (+0.06 … +0.26 m/s). The CBF
  repulsion is no longer balanced by the pull.
- Non-arrivers added in the last 0.2 s alone:

  | arm | added in last 0.2 s, per seed | total |
  |---|---|---|
  | noise_only | +1 +7 +3 +1 +5 | 17 of 69 non-arrivers |
  | fresh | +1 +3 +2 +3 +2 | 11 of 33 |
  | A | +3 +6 +1 +4 +2 | 16 of 72 |

**2. Compaction at the rim is slow under the CBF and has not finished at T.**
- Inside drones are pressed together at the CBF distance. Nearest-neighbour median: 0.550 m for fresh, 0.61–0.62 m
  for noise_only.
- They crowd the rim instead of filling the core:
  - 31–35 drones in the 3.5–4.0 m ring (~2.7 /m², vs 1.8 /m² planned);
  - only 7–14 drones at r < 2 m, where m* puts ~33.
- At the rim the MFG asks for inward motion (u_nom radial −0.3 … −0.6 m/s for noise_only). The CBF passes only
  −0.0 … −0.2 m/s.
- So the outside count falls slowly and is still falling at T. noise_only went 18–23 (T−10 s) → 13–19 (T−5 s) →
  6–14 (T−0.5 s).
- Within the last 5 s it also swings by 6–12 drones (min 5–12, max 17–20), because drones hover on the boundary.

**Why noise costs ~7 more drones (fresh 4–9 → noise_only 13–15).** With 0.1 m range noise the CBF holds neighbours
apart at a larger effective spacing: 0.61–0.62 m instead of 0.55 m. The same disc holds fewer drones at the same
pressure, so more are left on the rim.

**Contrast: without the MFG density coupling (fresh_nomfg) it is a real CBF deadlock.**
- 34–49 out, frozen: speed ~5e-5 m/s, pressed at exactly 0.55 m.
- u_nom points inward at 1–3.5 m/s and the CBF cancels all of it.
- With the MFG the rim is not frozen: drones keep moving at 0.1–0.4 m/s and the count keeps falling. They simply
  run out of time.

## Named cause

**Horizon end, not gate distance and not inlet deadlock.**
- At T the CBF-limited compaction at the rim is still under way.
- In the last 0.5 s the MFG routing field drops to zero, because the terminal cost is 0. That releases the rim drones
  and pushes ~25 % of the non-arrivers out of the disc in the final 0.2 s.
- Sensor noise widens the CBF spacing and roughly doubles the count.

**Chain link:** "delay-free works" fails at the end of the horizon. A predictor cannot fix this.

**At most one principled fix (not run, needs PI go):**
- Give the MFG the terminal cost of the reference formulation, g(x) = the goal potential, in place of 0
  (deviation c in prereg amend34 §5). Then ∇Φ does not vanish at T.
- This removes mechanism 1 directly.
- Whether mechanism 2 (unfinished compaction) is then still above 0 drones must be measured. It cannot be decided
  from these logs.
