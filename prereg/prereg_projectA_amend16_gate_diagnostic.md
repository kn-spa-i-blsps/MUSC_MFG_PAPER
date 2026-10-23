# Amendment 16 — Baseline fairness (robust margin) + gate-width diagnostic + §8 viz

**Base:** amend15 (density sweep, pinned geometry). No controller code change.
**Status:** frozen, pre-run. Committed with SHA before any run; results via `git -f`.
**Runs detached on the pod** (tmux/nohup) to completion, then commits results `-f` and pushes — so it
survives the local machine going offline.

**Why:** amend15 showed only `learned` gets any collision-free arrivals; real/cv/smith collide 100%.
Before concluding "only learned works" we must rule out that the baselines are **strawmen**. The
measurement-robust-CBF literature (Cosner-Yue-Ames 2021; Dean-Taylor-Cosner 2020) prescribes
inflating the CBF margin to the error bound. Our baselines ran at a fixed small margin
(d_safe−D_COLLISION = 0.25 m) far below their estimation error (real RMS 0.81, cv 0.97). Part 1 tests
the fair non-learned baseline. Parts 2–3 keep the earlier gate-knob and §8 viz questions.

---

## 1. Part 1 — robust-margin frontier (the fairness test, PRIORITY)

Question: is `learned`'s safety unique, or does a non-learned CBF with a properly inflated margin also
reach safety — and at what completeness cost?

- **Fixed:** s = 1.05 (amend15 best cell), gate = 2.309 (baseline), N = 100, D_COLLISION = 0.30, all
  amend15 params. **No new solve** — d_safe is a CBF param, geometry unchanged; reuse the s=1.05
  Φ/m* cache.
- **Arm `real_robust`: d_safe swept ∈ {0.55, 0.90, 1.30, 1.90} m** (0.55 = amend15 real; 1.90 ≈
  D_COLLISION + 2·RMS_real, covering ~p97 of real's staleness error). Perception = real (stale).
- **Reference points (reuse amend15, no re-run):** real @ d_safe 0.55 and learned @ d_safe 0.55.
- **Seeds:** 0–4. **Report per d_safe:** safe-success, completeness, % ever d<0.30, p5 worst sep.
- **Pre-registered reading:**
  - If some d_safe makes `real_robust` safe (safe-success ↑) **only by collapsing completeness** →
    the honest claim is *"learned achieves safety at a deployable margin; the non-learned fix buys
    safety only by freezing the swarm"* — a safety/conservatism trade-off, learned dominates it.
  - If `real_robust` reaches safety **with completeness comparable to learned** → "only learned
    works" is FALSE; the contribution shrinks to "learned ≈ a well-tuned robust margin, cheaper to
    tune." We report that honestly.
  - If no d_safe (incl. 1.90) makes `real_robust` safe → staleness error is so large even a 1.9 m
    margin cannot cover it → compensation is necessary, strongly supported.

## 2. Part 2 — gate-width diagnostic (is transit density the knob?)

- **Fixed:** s = 1.05, d_safe = 0.55, arms {real, learned}, seeds 0–4.
- **Swept gate_width ∈ {1.5, 2.3, 3.5, 5.0, 7.0} m** (7.0 ≈ half the 13.39 m cross-section). Re-solve
  Φ/m* per width (5 solves); geometry otherwise pinned at amend15.
- **Log** incl. closedloop_xhat_mse_near (catch learned OOD at wide gates — single-gate-trained
  predictor).
- **Reading:** learned safe-success ↑ / ever_collided ↓ as gate widens → transit density is the knob
  (Path A). Flat → collisions are estimation-error-driven regardless of density (Path B).

## 3. Part 3 — §8 viz sanity (quarantine rule)

Render trajectory animations for **real seed 0, real_robust(d_safe=1.90) seed 0, learned seed 0** at
s=1.05, gate=2.309, with collision markers (true d<0.30), CBF-intervention markers, and the CBF-active
flag per agent. `log_xhat: true` for these. **Confirm the CBF is actually engaged for real** (neighbor
set non-empty, d_safe applied, QP solved). If the filter is bypassed → amend15's baseline rows are a
wiring artifact, quarantine them and fix the CBF. If the CBF is acting and collisions still occur via
stale estimates → numbers are physical (error ≫ margin), consistent with robust-CBF theory.

## 4. Compute & execution

Part 1: 4 d_safe × 5 seeds = 20 rollouts, no new solve. Part 2: 5 gates × 2 arms × 5 seeds = 50
rollouts + 5 solves. Part 3: 3 render rollouts. ≈ 1.5 h on the A40, ~$1. **Launch detached on the pod
(tmux/nohup); on completion write the JSON report, `git add -f` results, commit, and `git push` — so
the local machine can go offline.** Resume pod 6k5omnvf9h2ulr.
