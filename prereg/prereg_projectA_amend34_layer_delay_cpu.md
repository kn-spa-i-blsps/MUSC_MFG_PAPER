# Amendment 34 — through which layer does TDMA delay do its damage? (EXPLORATORY, CPU, N=100)

**Purpose (one sentence):** find out through which layer TDMA delay does its damage (the CBF safety filter or the MFG
density term), so that the predictor gets built for the layer where compensation can matter.

**Status:** EXPLORATORY. Written and committed before any rollout of this amendment. PI request 2026-11-03.
Not authorised here: no retraining, no 0.70 m margin change, no Φ-informed predictor, no training, and no parameter
change.

## Pre-run facts (from code and committed results only)

1. **amend27 "exact 0.000"**
   - **Cause.** amend27 never established the cause. The review said "structural: the density term is a
     kernel-smoothed macro quantity", but no experiment tested that.
   - **What the committed per-seed data show** (amend27 N=100, d0.70; amend22 real):
     - Every agent collides at least once in both `real` and the MFG-only oracle split (ever-collided 1.000 in
       10/10 seeds). The per-agent metric safe-success is therefore pinned at 0 in both arms, so the difference is
       0.000 by construction: a **floor effect**.
     - Collision pair-steps did move. MFG-only oracle (`split:noise_only/real`) 39.6–41.6 k vs real 44.6–50.1 k,
       lower in 10/10 seeds, about −15 %.
     - MFG-only NN gave 42.6–45.2 k.
   - **Do compensated positions reach the MFG density term? Yes.** In the split arm `x_hat_mfg` is passed to
     `nominal_control` → `local_density_and_grad`, both in m̂ and in its gradient (env/rollout.py, control/macro.py).
     The hold-on-arrival branch uses it too, but hold is off in every config used here.
   - **What amend27 measured.** The report has safe-success, ever-collided<0.30, completeness, p5 worst-step
     separation, CBF intervention rate, slack-active rate and closed-loop x̂ MSE (near / all).
     - Collision pair-steps, makespan and t90 were saved per seed but not reported.
     - Real completeness on that geometry is already 0.994, so the run could not show a flow (reached-goal) effect.
2. **d_safe of each row of amend33** (last night's table, config ladder_fargoal_amend11.yaml):
   - Fixed d_safe = **0.55 m** in fresh, fresh_nomfg, noise_only, the identity real run, and the reused amend11 rows
     real, cv, learned and smith.
   - fresh_nocbf has the CBF off, so d_safe is not used.
   - No row uses an adaptive margin.
3. **amend11 learned predictor**
   - Weights: results/ladder_fargoal_N100/predictor_weights.npz, sha256 fc4a52a5…, record in
     experiments/ladder_fargoal_predictor_record.json.
   - Trained with experiments/train_predictor.py: a 2×64 MLP on a 4-broadcast window of stale, noisy observations.
     The target is the neighbour's current relative position.
   - Training data: 2.0 M windows from `real`-arm closed-loop datagen rollouts of **experiments/ladder_fargoal.yaml**
     (amend6), seeds 100–107 train, 108–109 val, 110–111 test. N=100, T_SLOT 0.005 s (T_frame 0.5 s), CBF
     exact_hard.
   - **How that differs from this test config (ladder_fargoal_amend11.yaml):**

     | | training (amend6) | this test (amend11) |
     |---|---|---|
     | d_safe | 0.40 | 0.55 |
     | goal hex spacing | 0.60 | 0.80 (larger r_goal, so a different goal disc) |
     | chamber | — | taller downstream chamber (goal_clearance 1.18) |
     | Φ / m* solve | its own | different |
     | controller the data came from | `real` arm (delayed estimates) | the predictor is used in closed loop |

     The delay model (T_SLOT, N) and the sensor noise are the same in both.
   - Open-loop test MSE was 0.225 m² on amend6 test seeds, and 0.357 m² near pairs on amend11 test seeds
     (real 0.686).
4. **Identity check and pairing**
   - **What it covered.** Last night's identity check (amend33) covered the **real arm only, seed 0, on this CPU**
     (jax 0.11.1, default precision). It compared 7 metrics with amend11 real seed 0 and all 7 were equal.
   - **The learned arm was not checked.** The October learned rollouts (amend11, same CPU, Oct 23) used code that
     has changed since in perception, rollout and cbf (amend17/19/27/29/31 branches, all claimed inactive by
     default).
   - **Pairing.** They share seeds, and therefore initial positions and noise keys. They can be paired by seed, but
     a **strict pairing with last night's runs is not verified**: nobody re-ran learned seed 0 on current code.
   - The real and noise_only reference rows are what this amendment uses, and they get the one-seed check below.
5. **Reference MFG formulation**
   - **Closest standard formulation.** The repo never names a single literature reference. The closest standard
     formulation is the deterministic (first-order) finite-horizon MFG with quadratic Hamiltonian H = ½|p|² and a
     local, monotone congestion coupling W(m) = c·m² (Lasry–Lions; crowd aversion as in Lachapelle–Wolfram). It is
     solved by fictitious play with Cesàro averaging (Cardaliaguet–Hadikhanloo 2017, cited in mfg/cesaro.py).
   - **Deviations of this code from that formulation:**
     a. extra soft barrier w_bar·relu(m − m_bar)², with w_bar 10 and m_bar 6.25;
     b. coupling capped at f_cap = 300;
     c. the running cost has a goal term (cone + tanh "hold" step, w_goal 2, w_hold 8) and a soft wall penalty
        (V_obs 200) in place of state constraints; terminal cost 0;
     d. FP is damped (mixing 0.02), not the 1/n weights. The last iterate does not converge (residual 0.18 at 800
        iterations). m* is the Cesàro mean over iterations 200–800 (Cesàro residual 0.0016), and Φ is one more HJB
        solve against m*;
     e. numerics: explicit Lax–Friedrichs HJB with local dissipation, upwind finite-volume transport, 0.2 m grid,
        dt 0.004;
     f. the agents do not apply the equilibrium feedback −∇Φ alone. u_nom adds an online one-sided term
        −c_dens·∇[c_w·relu(m̂ − m*)² + w_bar·relu(m̂ − m_bar)²], where m̂ is a 0.6 m Gaussian KDE (5 m range) of the
        estimated neighbours. This term is not part of the MFG equilibrium;
     g. u_nom is clipped to 8 m/s, then filtered by the CBF-QP (d_safe, per-axis box 5.657 m/s);
     h. 100 finite agents with sampled initial positions, process noise ν = 0, success = inside r_goal.

## Design

- **Config, margin and seeds:** exactly amend33 / amend11. experiments/ladder_fargoal_amend11.yaml, N=100,
  d_safe 0.55, seeds 0–4, CPU, default precision. Only `--arm` and run_name/log_xhat differ.
- **Arms** use the existing amend27 split code, `split:<mfg>/<cbf>`. Both estimates come from the same PRNG key, so
  the delay-free input carries exactly the noise-only arm's sensor noise.
  - **A** = `split:real/noise_only`: delayed x̂ to the MFG, delay-free x̂ to the CBF.
  - **B** = `split:noise_only/real`: delay-free x̂ to the MFG, delayed x̂ to the CBF.
- **References:** `real` = amend11 real seeds 0–4; `noise_only` = amend33 noise_only seeds 0–4.
- **Check (seed 0), current code through the split path:**
  - `split:real/real` must give pos arrays bit-equal to amend11 real s0, plus the 7 amend33 IDENTITY_KEYS. log_xhat
    true, as in amend11.
  - `split:noise_only/noise_only` must be bit-equal to amend33 noise_only s0 (log_xhat false).
  - If either check fails, that reference is used descriptively only and the failure is reported.
- **Run size:** 10 + 2 rollouts, 2 in parallel, nothing trained.

## Per-seed facts (all arms)

- **fleet success:** 1 if every drone is inside r_goal at the end AND no pair is ever closer than
  D_COLLISION = 0.30 m, else 0.
- **reached-goal:** fraction inside r_goal at the end.
- **drones ever collided:** count of agents with any true d < 0.30.
- **collision pair-steps.**
- **makespan:** first arrival of the last drone; none if not all arrived.
- **t90:** first time 90 % are inside r_goal.
- **Non-arrivers** (not inside r_goal at the end), each placed in exactly one bucket:
  - *still moving at the horizon*: net displacement over the last 5 s ≥ 0.5 m;
  - *jammed at the gate*: not moving AND within 3.0 m of the gate centre (wall_x, gate_z);
  - *other*: not moving, elsewhere. Upstream / downstream of the wall is reported.

## Statistics

- Paired by seed (n = 5).
- Exact two-sided sign-flip test on the mean paired difference, all 2⁵ = 32 sign assignments.
  - **Resolution:** the smallest possible p is 2/32 = 0.0625, reached only when all 5 differences share a sign. No
    contrast can reach p < 0.05 at n = 5.
- Seed-level bootstrap 95 % interval (10 k) alongside.
- Contrasts: A − real, A − noise_only, B − real, B − noise_only.
- Recovered fraction = (arm − real)/(noise_only − real).

## Interpretation, committed before the run

- **A close to noise-only on collisions** ⇔ A recovers ≥ 90 % of the real → noise_only gap in mean drones-ever-collided
  AND in mean collision pair-steps. ⇒ The delay's safety damage goes through the CBF.
- **B moves reached-goal from real toward noise-only** ⇔ B − real reached-goal > 0 in 5/5 seeds (sign-flip p = 0.0625,
  the minimum). ⇒ Stale density costs flow, and an MFG-layer predictor has a measurable job.
- **Neither arm alone recovers it** ⇔ no single arm reaches ≥ 90 % of the real → noise_only gap on BOTH mean
  drones-ever-collided AND mean reached-goal. ⇒ The layers interact. This goes to the PI before any next step.
- The whole result is labelled exploratory and goes to the PI in plain words, with one sentence per chain link.
