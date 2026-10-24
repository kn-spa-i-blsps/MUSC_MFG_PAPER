# Amendment 21 — N crossover (fixed arena), under precision=highest

**Base:** amend20 (precision lock-in; the amend19 closure win was a precision artifact — under
`highest`, best single method is fixed d_safe=0.70 at safe-success 0.412/compl 0.529; closure k=0.75
beats fixed 0.55 by ~+0.10 at matched completeness 0.85 but does not dominate the frontier; nothing is
collision-free at N=100). No controller change. Frozen, pre-run, SHA before run, detached, self-push.

**Motivation:** N=100 in this gate is revealed as a fundamentally hard regime — staleness
T_frame = N·t_slot = 0.5 s is large and no margin scheme reaches collision-free. The real question is
the **crossover**: at what swarm size does the system stay safe, and does the method push that size
higher than the baseline. This is the "how many drones fit in this field" deployment view.

---

## 1. Design

- **Fixed arena — ONE geometry for all N**, sized so the largest N (200) fits (start box + goal disk)
  at initial spacing ≥ d_safe=0.55. Same arena, gate, goal disk, start box across every N. Report the
  dimensions. (Fixed geometry ⇒ a single Φ/m\* solve reused for all N — N enters only as agent count.)
- **Swept N ∈ {25, 50, 100, 150, 200}.** t_slot = 0.005 fixed, so **T_frame = N·t_slot** grows with N
  (0.125 → 1.0 s), AND density = N/arena grows with N — both staleness and packing rise together
  (the deployment framing; the two are not separated here, by choice).
- **Arms:** real (baseline), learned fixed d_safe=0.70 (best constant from amend20), learned
  closing-gated k=0.75 (best selective from amend20). D_COLLISION=0.30, floor 0.55, h=0.3.
- **precision=highest** (standard). **Seeds 0–9 (10).** Seed-cluster bootstrap.
- **Log:** safe-success, ever d<0.30, completeness, p5 worst sep, CBF intervention, slack, realized
  d_safe (for closing), closedloop_xhat_mse (staleness error grows with N — confirm the mechanism).

## 2. Pre-registered questions

- **Q1 (safe regime exists):** is there an N at which learned (either variant) reaches ≈ collision-free
  (ever d<0.30 → near 0, safe-success → high)? Expected at small N (T_frame small → estimation error
  small). This is the "it works here" anchor to show the supervisor.
- **Q2 (crossover / method value):** at what N does each arm cross from safe to unsafe, and does
  learned (fixed 0.70 and/or closing 0.75) hold safety to a **higher N** than real? The gap in that
  crossover N is the method's deployment value.
- **Q3 (mechanism):** does closedloop_xhat_mse rise monotonically with N (staleness), tracking the
  rise in collisions? Confirms staleness as the driver.
- **Falsifier:** if even N=25 is not safe (ever d<0.30 stays high) → collisions are not staleness/size
  driven in this scenario and the whole framing is wrong; report honestly. If real and learned cross at
  the same N → compensation gives no deployment headroom.

## 3. Note on the confound (stated)

Fixed arena couples density and staleness: both rise with N, so this measures the combined deployment
limit, not staleness in isolation. The decomposition (density alone = amend15/16 gate sweeps;
staleness alone = a future t_slot sweep at fixed N) is handled elsewhere. Here the question is simply
"how large a swarm stays safe in this field, and does the method enlarge it."

## 4. Compute

5 N × 3 arms × 10 seeds = 150 rollouts + 1 solve. Rollout cost grows with N (N=200 ~2× N=100). ≈ 2 h
on the A40 under highest, ~$1.5. Detached, self-push.
