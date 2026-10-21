# Amendment 2 to prereg_projectA.md — process noise in the rollout dynamics

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a), §3 (locked parameters: process noise was not
listed) and §9 (smoke gate run specification). The original file is NOT edited.
**Status:** approved by the PI before any Project-A run (no outcome observed).

## Reason
The ported rollout (MUSC_MFG @ physical-refactor-final, phys_rollout.py) adds Brownian process
noise NU = 0.091 m^2/s (sigma ~= 4.3 cm per 0.01 s step per axis) to the true position update.
This term is not in the prereg, is not modelled by the CBF, and is not modelled by the MFG solver
(first-order, deterministic, no viscosity). Under tight packing it makes collisions > 0 near-certain
regardless of the CBF.

## Change
- Primary smoke run (§9): NU = 0 — deterministic dynamics; noise only in sensing (SIGMA_R,
  SIGMA_THETA, TDMA staleness), consistent with the deterministic solver. §9 (4) zero-collisions is
  evaluated on this run.
- One additional smoke run with NU = 0.091 (legacy, ported), reported as FACTS only, not gated:
  collision count, collision rate, worst-case p5 inter-agent separation, CBF intervention rate.
  Purpose: measure the robustness gap of the deterministic CBF against unmodelled process noise.

## Open design decision (to be resolved AFTER the smoke, not now)
A hard no-collision guarantee is well-defined only for the dynamics the CBF models. Process noise
requires either (a) narrowing the guarantee to deterministic dynamics, or (b) a robust CBF with a
margin for a physically pinned noise level sigma (plus, optionally, a matching viscosity term in
the MFG solver so that m* stays consistent with the rollout). NU = 0.091 is a legacy value that
must be re-pinned physically before it is used in any reported result.
