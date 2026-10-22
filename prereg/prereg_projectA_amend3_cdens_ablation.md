# Amendment 3 to prereg_projectA.md — ablation of the online micro density term (c_dens = 0)

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a), §2 (macro layer:
nominal control) and §3 (locked parameter C_DENS) — as a DIAGNOSTIC ablation only. The original and
amendments 1 (2c080dd4…) and 2 (4c5015a3…) are NOT edited. No official smoke (§9) or factorial run is
made under this amendment; the §9 gate spec awaits a separate reframe.
**Status:** written before any c_dens = 0 run (no outcome of this ablation observed).

## Context (facts already observed, diagnostic runs)
- Fresh perception, legacy solver: completeness 0.68, mean speed 0.25 m/s, 0 collisions.
- Decomposition of u_nom in front of the gate (P3a): routing pull without congestion ~6 m/s; the macro
  W(m) term inside Phi removes ~3.4–4.8 m/s; the online micro term
  -C_DENS * grad[c_w * relu(m_hat - m*)^2] removes a further up to ~1.8 m/s, because m* ~ 0 before the
  wall after t ~ 5 s so every remaining agent sees excess ~ m_hat ~ 2 and is pushed away from the crowd
  at the gate. Net u_nom 0.16–0.23 m/s.
- The micro barrier term w_bar * relu(m_hat - m_bar)^2 contributed 0.00 m/s there.

## Hypothesis (Arch 2)
With a hard online CBF the micro term C_DENS * grad relu(m_hat - m*)^2 is redundant with the CBF for
safety and anti-throughput: removing it (C_DENS = 0) raises completeness and lowers the number of
agents stranded before the gate, without collisions under fresh perception.

## Test
- Single-gate, N = 100, CBF on (qp_solver exact_hard, P1 fix), NU = 0, seed 0.
- Phi / m* from the existing cache (c_w = 1, unchanged), CBF parameters unchanged.
- Only change: C_DENS = 0 in the nominal control (control/macro.py). Note: C_DENS multiplies both
  the excess-congestion term and the micro barrier term; the latter was measured at 0.00 m/s (above).
- Run A: fresh perception, C_DENS = 0 vs C_DENS = 1 (same solver, same seed) as baseline.
- Run B: real perception (TDMA delay + sensor noise), C_DENS = 0 vs C_DENS = 1 (same solver).
- Reported as facts: completeness, arrived fraction over time, agents left of the wall at T,
  CBF intervention rate, QP infeasibility (slack-active rate), collisions, min separation,
  worst-case p5 separation; visualizations of all runs.

## Decision rule
None pre-committed for the gate: this is a diagnostic informing the reframe. The hypothesis is
supported in A if completeness rises and stranded agents fall with 0 collisions at C_DENS = 0.
