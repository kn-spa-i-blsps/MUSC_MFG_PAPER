# Note to the agent working on this repo (PI review, 2026-11-02)

Read this before proposing or registering anything.

## What went wrong (amend1–32)

Between 2026-10-21 and 2026-11-02 this repo collected 32 amendments, about 70 experiment scripts, bit-identity
gates, GPU-determinism investigations and four addenda to a single amendment. Each step was locally reasonable and
honestly reported. Taken together, they moved the project away from its idea without anyone deciding that it should.

- **The thesis changed without being named.**
  - The idea: an MFG crowd-aversion controller whose local density estimate goes stale under TDMA delay that grows
    with N, and a predictor that compensates.
  - Where it ended up: "an NN predicts neighbour positions for a CBF safety filter." The project's own prereg
    (§2) calls that the prior work it wanted to be distinct from.
  - amend27 showed that compensation fed to the MFG layer changes safety by exactly 0.000, even with an oracle.
    That result should have stopped the work and been raised with the PI as a thesis-level finding. Instead it
    became one more row followed by the next amendment.
- **Symptom-fixing instead of stepping back.**
  - amend1–10 tuned the goal radius, goal shape, hold-on-arrival, d_safe and the attractor position to get past the
    smoke gate.
  - amend15–20 tuned CBF margins: adaptive, closure-gated, σ/conformal.
  - None of these asked whether the change served the basic question.
- **Metrics were redefined after the results were in.** amend12 and amend13 added "safe success" and threshold
  curves on logs already seen. They were pre-registered before analysis, which is correct. But the pattern
  (result disappoints → new metric) is goalpost-moving and should be called out, not normalised.
- **Process outgrew the science.**
  - An aneks + SHA for every tweak, identity gates to the last ulp, and GPU nondeterminism became their own
    research threads.
  - Pre-registration exists to protect the main claims. It is not meant to make every knob turn look rigorous.
- **Scope crept.** Phase B, conformal σ-CBF, cvls, physical-limit sweeps, layouts and chaos ensembles were each
  justified by the previous result, never by the goal.

## What to do instead

The goal, from the PI (2026-11-02):
- An MFG controller as used in the swarm literature.
- Realistic physics: drone size, UWB range noise, vision bearing noise, TDMA delay.
- Predictors that compensate those errors.
- Success = every drone reaches the goal and never collides.
- The CBF stays as a named safety filter.
- Stage 1 is the offline MFG. The fully online per-drone MFG with dynamic waypoints and unknown terrain comes later.
- Kept for now: 5" drones, single-gate geometry.

Rules:
1. **Before any new experiment, write one sentence on how it serves that goal.** If you can't, don't run it.
2. **A thesis-level result stops the work and goes to the PI in plain words before the next amendment.** Examples:
   a layer doesn't matter, the baseline doesn't work, the effect is somewhere other than expected.
3. **Don't tune your way past a failing gate.** If the baseline fails, report why and propose at most one
   principled fix.
4. **One prereg per real question.** Logging and infra changes don't need an aneks. Identity gates only when a
   comparison actually depends on them.
5. **No new metrics after seeing results,** unless the PI asks. Flag it explicitly if you think a metric is wrong.
6. **Keep the chain visible:** delay-free works → delay damages → compensation restores → each layer earns its
   place. Every result should say which link it supports or breaks.
