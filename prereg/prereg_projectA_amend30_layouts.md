# Amendment 30 — compensation ladder and MFG value on generic layouts

**Base:** amend21 arena, start box, goal disc and horizon rule; amend27 margins; amend28 arms. PI goal (2026-10-31):
generalization beyond the single centred gate. Frozen, pre-run, SHA before run, detached, self-push. Code:
env/scenarios.LayoutA + scenario key `layout` (absent = SingleGateA, unchanged), experiments/amend29_physical_limits.py
(--exp amend30).

## Layouts (geometry_N 230 arena, gate width 2.309)
- `two_gates`: two gates at the source chamber's quarter points (z = z_off + H0/4, z_off + 3H0/4), no centre gate.
- `offset_gate`: one gate at z = z_off + H0/4.
- `pillar`: the centred gate + a 4 m square pillar midway between the wall and the goal disc's near edge, z = H/2
  (CBF: its 4 edges as wall segments; V_obs: obs_penalty·exp(−(distance/thickness)²)).
One Φ/m* solve per (layout, N) with c_w = 1, plus the decoupled solve (c_w = w_bar = 0) at N=100; caches deleted after
each layout.

## Design
3 layouts × N ∈ {100, 150} × arms {real, cv, smith (amend27 margin rule, same as amend29), learned v3 d0.70 (frozen;
never trained on these layouts), noise_only d0.70, fresh d0.70} + fresh_nomfg d0.70 (decoupled routing, c_dens 0, CBF)
at N=100 × seeds 0–9, precision=highest. 390 rollouts.

## Questions and pre-registered readings
- **Q1 baseline per layout:** fresh safe (amend28 definition) at N=100/150.
- **Q2 MFG value per layout:** fresh − fresh_nomfg safe-success / completeness at N=100 (CI); two_gates is the case
  where a mean-field equilibrium should split the flow and decoupled shortest-path routing should not.
- **Q3 ranking per layout:** as amend29 Q2 (learned vs each comparator; recovered fraction).
- **Q4 transfer:** a learned loss of advantage on an unseen layout is classified PLUMBING (layout data → retrain), not
  thesis.

## Review
logs/overnight/review_amend30.md.
