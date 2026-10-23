# Amendment 17 — Staleness-aware adaptive CBF margin (method v1)

**Base:** amend15/amend16. **This is a CONTROLLER CHANGE** (the method contribution), not a pure
measurement. Frozen, pre-run; committed with SHA before any run; results via `git -f`. Runs detached
on the pod, self-pushes.

**Motivation (from amend16):** collisions are stale-tail events — at the collision instant the CBF's
estimated neighbour distance is ~0.8 m wrong for real (est 1.04 vs true 0.22), ~0.26 m for learned.
A *uniform* robust margin failed (PART 1: d_safe=1.90 → still 100% collide, completeness 0.86→0.24):
it over-inflates everywhere yet is still mistimed at the tail. Fix: put the margin **where the error
is**, computed online from what we KNOW — the TDMA staleness and the neighbour's speed.

---

## 1. Method

Replace the constant CBF safety distance d_safe with a **per-pair, per-timestep** margin:

**d_safe_ij(t) = D_COLLISION + k · σ̂_ij(t),  capped at d_max**

- **D_COLLISION = 0.30** (physical, fixed).
- **σ̂_ij(t) = v̂_j · Δt_stale_ij** — the bound on how far neighbour j's position estimate can be
  wrong: its estimated speed v̂_j times the time since i last heard j over TDMA.
  - **Δt_stale_ij**: time since j's last TDMA slot update, known exactly from the schedule / the
    per-neighbour last-update timestamp already tracked by the perception model.
  - **v̂_j**: neighbour j's speed estimate from the arm's own estimator (cv/learned carry velocity;
    for `real`, finite-difference of its last two heard positions for j).
- **k**: safety factor on the staleness bound, **swept ∈ {1.0, 1.5, 2.0}**.
- **d_max = 1.5 m**: hard cap so the QP stays feasible (beyond this the pair is unrecoverable anyway).
- Same formula for **all arms** in v1 (no per-arm tuning). This is deliberately conservative for
  `learned` (its estimate already compensates staleness, so v̂_j·Δt over-bounds its true residual) —
  if learned's completeness survives the over-margin, v1 is enough; if it freezes, that is the signal
  to move to v2 (§4).

## 2. First validation run

- **Fixed:** s = 1.05, gate = 2.309, N = 100, D_COLLISION = 0.30, all other amend15 params. Reuse the
  s=1.05 Φ/m* solve (geometry unchanged; the margin is a CBF-runtime quantity, no re-solve). Learned =
  frozen amend11 checkpoint.
- **Arms:** real, learned (the informative contrast). **k ∈ {1.0, 1.5, 2.0}. Seeds 0–4.**
- **Reference rows (reuse, no re-run):** learned @ fixed 0.55 (safe 0.240 / compl 0.674 / ever 0.606),
  real_robust @ fixed 1.90 (safe 0.000 / compl 0.244 / ever 1.000), real @ fixed 0.55 (0.000 / 0.860 /
  1.000).
- **Log:** safe-success, completeness, % ever d<0.30, p5 worst sep, CBF intervention + slack rate,
  **and the realized d_safe distribution** (mean/p50/p95 of the adaptive margin per arm — shows how
  selective it is vs the uniform 1.90).

## 3. Pre-registered predictions (falsifiable)

- **P1:** adaptive-margin `real` drops ever_collided **below 1.000 for the first time** (the margin
  now inflates exactly on its fast-stale pairs) — but at a completeness cost **less severe than the
  uniform 1.90 freeze** (compl > 0.244) because the inflation is selective, not global.
- **P2:** adaptive-margin `learned` raises safe-success **above its fixed-0.55 value (0.240)** and
  drops ever_collided **below 0.606**, while keeping completeness **well above 0.244**. This is the
  win: safety recovered without the freeze.
- **Falsifier:** if learned's completeness collapses toward the uniform-1.90 level (~0.24), v1
  over-margins learned → the staleness term double-counts learned's compensation → go to v2.

## 4. Staging (do not run yet beyond v1)

- **v2 (next iteration, only if v1 shows learned over-margined):** σ̂_ij for `learned` comes from the
  **predictor's own uncertainty**, not raw staleness — retrain the predictor with a variance/quantile
  head so it outputs per-neighbour residual uncertainty, feed that as σ̂. This is the polished method
  (uncertainty-aware learned compensation feeding the CBF).
- **Full sweeps come AFTER collisions are driven sensibly down** (transit-density money plot over gate
  width, then τ, then N, then scenarios) — not before.

## 5. Honest ceiling

Under unbounded staleness no static or adaptive margin gives a HARD zero (a neighbour stale 2 s that
manoeuvred is uncatchable). The adaptive margin gives a **conditional guarantee**: collision-free
provided v̂_j·Δt_stale ≤ realized margin and the constant-velocity extrapolation holds over Δt_stale.
This is exactly the robust/conditional-guarantee thesis — v1 is its first constructive instance.

## 6. Compute & execution

3 k-values × 2 arms × 5 seeds = 30 rollouts, no new solve. ≈ 20–30 min on the A40, ~$0.3. Launch
detached (tmux/nohup) on the pod; on completion write report.json, `git add -f`, commit, push.
