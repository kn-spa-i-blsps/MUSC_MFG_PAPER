# Pre-registration — Project A (MUSC-MFG redo)

**Status:** frozen design, pre-run. No outcome has been observed under this design.
**Supersedes:** the physical_refactor line (density-gating thesis, settled-only metric window, settled-only NN training). Those results are retired as the paper's basis; mechanism findings (CV harmful, Smith pathological) are retained as prior evidence, to be re-confirmed under this design.

## 1. Thesis (v2 — emergent density)
Decentralized MFG routing of a TDMA swarm that trades congestion against goal-progress. Density is emergent, not a tracked setpoint: the swarm concentrates only where and when it pays off (to reach the goal faster/more efficiently), shaped by obstacles, under (i) a hard no-collision requirement and (ii) a completeness requirement — all agents reach the goal region. At fixed nominal (free-space) density, as the swarm grows (N) the TDMA staleness grows and degrades the quality of this trade-off (worse achieved cost, risked collisions, degraded completeness); learned delay-compensation recovers it. Delay and density are varied independently (factorial), answering prior reviewer point C3.

## 2. Architecture (two layers)
- Macro — soft MFG (routing). Offline-solved equilibrium gives value/routing field Phi and equilibrium density m*. Running cost = goal-progress + monotone congestion W(m) + soft collision barrier. Soft barrier keeps routing collision-aware. Solved once per (scenario, N); N enters rollout only via delay. m* = Cesaro mean over FP iterates.
- Micro — hard CBF (safety guarantee). Per-agent, decentralized CBF-QP on realized control, from local sensing. Enforces no-collision as a true guarantee, not a penalty. Liveness: goal-biased priority to prevent deadlock. Feasibility fallback: soft-slack (bounded violation minimized), never emergency-stop, so an infeasible QP never freezes an agent and breaks completeness.
- Delay compensation feeds BOTH layers: predictor's local-neighbour estimate consumed by macro density gradient AND micro CBF neighbour estimates. This dual coupling is the core novelty (distinct from predictor->CBF-only prior work).
- §8 invariant held: NN = delay predictor only; no learned policy. (Learned field amortization for scenario-free online operation is Project B, not A.)

## 3. Locked parameters
RHO_NOM=1.0 agents/m^2 (nominal free-space density, ~1.0 m spacing). Scaling: density-constant via arena_for_N(N) with RHO_NOM=1.0 (arena grows with N, density fixed). D_COLLISION=0.30 m (hard, CBF). D_STAR=0.60 m (desired spacing, soft). R_BARRIER=0.40 m. R_SENSE=5.0 m (prior H-run: 2.5 m near-identical — robustness noted). V_MAX=8.0 m/s. V_NOM=1.0 m/s (sizes horizon only). C_DENS=1.0. H_BANDWIDTH=0.60 m. SIGMA_R=0.10 m. SIGMA_THETA=0.035 rad. T_SLOT=0.005 s (nominal; varied on delay axis). gate/width ratio=0.20 (fixed across N). goal region r_goal=1.0 m. horizon=3*W/V_NOM per scenario/N; FULL-horizon logging (no settled-only truncation).
Arena per N (RHO_NOM=1.0): N=25 -> 5.77x4.33 m; N=50 -> 8.16x6.12; N=100 -> 11.55x8.66; N=150 -> 14.14x10.61; N=200 -> 16.33x12.25. Gate width = 0.20*W.

## 4. Objective / cost (macro MFG)
J = w_goal*V_goal + W(m) + w_bar*V_barrier, with: goal-progress V_goal strong enough to dominate crowd-aversion locally at the goal so agents reach AND hold (fixes prior in_goal~2% dispersal), represented so a local goal direction (bearing/range) is available to the agent independently of Phi (B-readiness, §8-clean in A); congestion W(m)=c*m^2 monotone; soft barrier = collision-aware routing only (hard guarantee is CBF). Uniqueness NOT forced: report FP+Cesaro selection + convergence diagnostics; do not claim uniqueness where it does not hold (asymmetric double-gate, room).

## 5. Experimental design (factorial)
Axis A (swarm size / coupled delay): N in {25,50,100,150,200} at fixed RHO_NOM (arena grows; T_frame=N*t_slot grows with N). Axis B (pure delay): t_slot in {0.0025,0.005,0.0075,0.010} s at fixed N (isolates staleness from swarm size). Arms: real, cv, learned, smith (optional baseline age-weighted KDE if time). Seeds: 20 per cell, seed-level cluster bootstrap. Scenarios (4): single-gate (core), asymmetric double-gate (non-unique equilibrium test), corridor (transit-heavy), room-with-obstacles (richest field — highest solver/liveness risk, hardest zero-shot transfer target; informative either way). Compute plan: full factorial (A x B) on single-gate; Axis A only (at nominal t_slot) on the other three. Escalate to full factorial on all four IF time permits (secondary).

## 6. Learned predictor
Trained on FULL-horizon rollouts (incl. transit) with a warm-started history buffer — fixes the settled-only OOD failure of the prior line. Trained on single-gate only; zero-shot transfer to the other three reported as generalization result (and Project-B feasibility evidence). Disjoint train/eval seeds; frozen checkpoint hash recorded before eval runs.

## 7. Metrics
Completeness: fraction of agents within r_goal at horizon end (primary success criterion). Safety: collision count (target ~0 with CBF — the guarantee); worst-case p5 inter-agent separation; CBF intervention rate + QP-infeasibility/slack rate. Congestion: achieved integral W(m) over full horizon + transit/settled decomposition (windows from measured crossing curve, not a fixed tail fraction). Throughput: time-to-goal / makespan. Headline: arms x delay-axis x scenario on the four dimensions.

## 8. Visualization (standing control — mandatory before trusting any number)
Per scenario x representative seeds: m(t) field animation, agent trajectories, collision events, arrived-fraction over time, CBF intervention markers. Acceptance checklist per cell: concentration appears where/when cost-beneficial; reported fraction of agents reach the goal region; zero collisions with CBF on; CBF interventions bounded (no permanent freezing). A cell whose numbers disagree with its animation is quarantined, not reported.

## 9. Smoke gate (MUST pass before the full factorial)
Run single-gate x N=100 x real arm, with full viz, and check: (1) mean-field: W(m) on vs off changes behaviour materially (else geometry too loose); (2) density-constant: measured mean free-space density ~ RHO_NOM=1.0 across swept N (confirms the scaling path is in use, same arena feeds solve and rollout); (3) goal-holding/completeness: arrived-fraction high (>=90%) at N=100 real (confirms goal term holds agents, fixing in_goal~2%); (4) CBF safety: collisions=0 with CBF on; (5) liveness: completeness stays >=90% WITH CBF on (CBF does not deadlock the swarm); (6) viz: animation passes §8 checklist by eye. Any failure -> STOP, diagnose, do not proceed.

## 10. B-readiness / port path (amortized-MFG, Project B)
Phi is a cleanly isolated module (global oracle router), replaced in B by a learned field amortizer; controller interface treats Phi and the learned module as swappable. Local goal direction (bearing/range) is a first-class agent input in A, so routing can go local in B. NN training data = across-scenario offline MFG solves on local-observation inputs — exactly what the B amortizer needs. CBF is already local/online/scenario-free — ports to B unchanged. Generalization / zero-shot transfer framed as A-evidence that the approach ports to scenario-free operation. A and B are two papers; A is B's foundation and motivation, not to carry B.

## 11. Kept vs changed
Kept: two-layer controller skeleton, TDMA/perception model, compensation ladder arms, prereg + seed-bootstrap discipline, CV/Smith mechanism findings (to re-confirm). Changed: density-constant scaling (was fixed-arena rising-density); emergent-density cost with goal-progress + goal-holding (was congestion-only, settled window); hard CBF layer (was soft-barrier only; macro density-constraint dropped); full-horizon NN training (was settled-only); 4 scenarios + zero-shot transfer; mandatory per-cell visualization.
