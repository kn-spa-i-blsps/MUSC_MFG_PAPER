# Amendment 33 — CPU baseline validation + layer ablations at N=100 (amend11 locked config)

**Why:** PI 2026-11-02 reset the focus to the basic chain: the delay-free MFG+CBF works → realistic delay damages it → a predictor restores it, and each layer earns its place. amend28 asks this, but its amend21 geometry does not fit in this machine's 30 GB RAM, and tonight is CPU-only. This is the CPU-feasible version on the amend11 locked config, where real/cv/learned/smith already exist on the same CPU (seeds 0–4).

## Design
- Config: experiments/ladder_fargoal_amend11.yaml (N=100, far goal, goal spacing 0.80, d_safe 0.55, default CPU precision). Seeds 0–4.
- Cells: experiments/amend33_cells/.
- New arms:

| arm | perception | controller |
|---|---|---|
| fresh | fresh | MFG (c_w 1) + CBF |
| fresh_nocbf | fresh | MFG, CBF off (box clip only) |
| fresh_nomfg | fresh | macro c_w = w_bar = 0 (decoupled single-agent routing), c_dens = 0, + CBF |
| noise_only | noise_only | MFG + CBF |

- Reused arms: amend11 real / cv / learned / smith (results/ladder_fargoal_amend11_N100).
- Identity: real seed 0 with the current code vs amend11 real seed 0 on 7 metrics. If it is not equal, the old arms are compared descriptively only (not as paired contrasts).
- log_xhat false in the new arms. This is logging only.

## Metrics
- safe success (amend12: agent in r_goal at the end AND never true d < 0.30);
- ever-collided;
- completeness;
- collision pair-steps;
- ∫W;
- makespan.

Seed-level paired bootstrap (10k), 95 % CI.

## Questions and readings
- **Q1 baseline works:** fresh ever-collided CI upper ≤ 0.10 and completeness ≥ 0.90.
- **Q2 CBF needed:** fresh_nocbf − fresh ever-collided > 0, CI excludes 0.
- **Q3 MFG earns its place:** fresh − fresh_nomfg on safe success or completeness, CI excludes 0. The sign is reported. A null or negative result is thesis-relevant and is reported as such.
- **Q4 delay is the damage:** noise_only − real safe success > 0, CI excludes 0.
- **Q5 compensation:** learned − real safe success (already known: +0.168); recovered fraction = (learned − real)/(noise_only − real).

5 seeds: underpowered for small effects; reported as such.
