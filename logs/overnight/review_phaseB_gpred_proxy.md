# Review — G-pred PROXY (exploratory, cannot pass/fail the gate)

Setup: N=100 only, amend11 geometry (results/ladder_fargoal_N100/datagen, real-arm histories), the old N=100 predictor
(results/ladder_fargoal_N100/predictor_weights.npz, trained on seeds 100–107 of these histories → in-distribution).
Test seeds 110–111, pairs with true distance < 2 m (n = 20348). precision=highest. Files: results/phaseB/gpred_proxy_N100*.json.

| variant | AUC top10 (val) | lift top1 | Spearman | R² log | corr² abs | Spearman in tail | raw q0.9 coverage | top/bottom σ-decile error |
|---|---|---|---|---|---|---|---|---|
| D (2×128, nf 0.5, 60k) | 0.794 (0.791) | 0.451 | 0.517 | 0.250 | 0.224 | 0.189 | 0.902 | 4.66 |
| cap 3×256 | 0.796 (0.798) | 0.475 | 0.524 | 0.256 | 0.229 | 0.169 | 0.890 | 4.71 |
| near_frac 0.8 | 0.791 (0.800) | 0.390 | 0.509 | 0.243 | 0.211 | 0.134 | 0.894 | 4.51 |
| near_frac 0 | 0.796 (0.797) | 0.459 | 0.510 | 0.238 | 0.213 | 0.191 | 0.903 | 4.52 |
| 200k windows | 0.808 (0.804) | 0.407 | 0.537 | 0.270 | 0.244 | 0.123 | 0.900 | 4.78 |
| age-only (D) | 0.775 (0.765) | 0.456 | 0.445 | 0.186 | 0.181 | 0.311 | 0.916 | 3.89 |
| range-only (D) | 0.548 | 0.338 | 0.065 | 0.005 | — | — | 0.899 | 1.48 |

Step A — mechanism: error of a delay compensator grows with the neighbour's staleness (age, sawtooth per slot) and with
relative motion; the age-only model capturing most of the signal is exactly that mechanism. Consistent.
Step B — artifacts: precision highest (logged); held-out seeds; no closed-loop code touched (identity n/a); val ≈ test
(no selection optimism); range-only baseline ≈ chance (sanity: the score is not just "near pairs").
Step C — consistency: agrees with amend26 (learned collisions track T_frame) and amend24 (delay error is the mechanism).
Step D — statistics: n=20k pairs but only 2 test seeds (correlated windows) → treat third decimals as noise; the
variant spread (±0.01 AUC) is the honest resolution.
Step E — verdict: PROXY only. Signal: heteroscedastic, tail-predictive at the "which pairs" level (AUC ~0.80, lift ~4.5×),
plateaued over the recipe ladder → not capacity-limited. Weakness: within the top decile the model barely ranks
(Spearman 0.12–0.19), and R² stays ~0.25 → σ says WHICH pairs are risky, not HOW large their error is. Classification:
not a gate result, neither PLUMBING nor THESIS-CRITICAL. Open question for the real gate: on the closed-loop learned-v3
histories (calm swarm, small errors) the heteroscedastic share may be smaller; that is the deciding measurement.
