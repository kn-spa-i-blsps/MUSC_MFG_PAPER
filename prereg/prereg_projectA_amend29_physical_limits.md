# Amendment 29 — compensation ladder under physical limits (sensor noise, packet loss, process noise)

**Base:** amend21 arena/geometry, amend27 (method ladder; margins). PI goal step 2–3 (2026-10-31): the damage of
physical limitations beyond pure TDMA staleness, and which compensator removes most of it. Frozen, pre-run, SHA
before run, detached, self-push. Code: experiments/amend29_physical_limits.py (--exp amend29); perception/ladder cell
keys `sigma_scale`, `p_loss`, `nu` (commit 2f796e8; absent = unchanged path).

## Physical-limit models
- `sigma_scale s`: UWB range σ_r = 0.10·s m and bearing σ_θ = 2°·s.
- `p_loss p`: every scheduled TDMA broadcast (frame f, slot j) is lost independently with probability p
  (deterministic integer hash of (f, j, salt), salt = 1000 + seed); a neighbour's age = time since its last
  SUCCESSFUL broadcast (≤ 7 extra frames). cv's previous observation and the learned window's older blocks keep
  their one-frame spacing behind that broadcast (stated simplification). Checked: mean age at p=0.2, N=100 = 0.372 s
  vs theory T/2 + T·p/(1−p) = 0.375 s.
- `nu ν`: process noise on the agents' motion, displacement std √(2ν·dt) per step.

## Design (one knob at a time; all other settings default)
settings sigma 0.5 / 2 / 4, loss 0.1 / 0.2 / 0.4, nu 0.01 / 0.05 × N ∈ {100, 150} × arms {real, cv, smith (each at
its better d_safe from amend27 at the same N by mean safe-success; tie → 0.70), learned v3 d0.70 (frozen, trained at
σ×1, p=0, ν=0 → every non-default setting is out of its training distribution), noise_only d0.70 (delay-free
reference under the same noise / process noise; packet loss does not affect it)} × seeds 0–9, precision=highest.
800 rollouts; the default setting's rows are amend27/amend22/amend24 (read side by side, not pooled).

## Questions and pre-registered readings
- **Q1 damage:** per setting, noise_only − real safe-success (from the report rows), sign and size.
- **Q2 ranking:** learned − each other compensator safe-success with CI; "NN best at a setting" = CI excludes 0 vs
  every comparator. Recovered fraction (method − real)/(noise_only − real) per method.
- **Q3 robustness of the frozen NN:** where learned loses its advantage (CI includes 0 or reverses) under a
  perturbation it was not trained on, the review classifies it PLUMBING (training-distribution gap → retrain v5 with
  the perturbation in the data, pre-registered separately) — not "the NN does not work".

## Review
logs/overnight/review_amend29.md.
