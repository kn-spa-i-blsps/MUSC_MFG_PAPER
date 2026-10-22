# Pre-registration index

| file | sha256 | scope |
|---|---|---|
| prereg_projectA.md | 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a | frozen original |
| prereg_projectA_amend1_rgoal.md | 2c080dd4c133f2fba0db49080c908d62dffd4d2991738e82bab055f18b39b97f | §3 r_goal -> r_goal(N); §9 (3),(5) |
| prereg_projectA_amend2_process_noise.md | 4c5015a3a10776e821e492b86f61facff2fbd638773dddd2f69f95d528b57ae4 | §3 process noise NU; §9 run spec |
| prereg_projectA_amend3_cdens_ablation.md | 54edaa364fa8b7f04679da3fa3226312178ecfeb922d81fa8b461c1ea13d1f50 | §2/§3 C_DENS = 0 diagnostic ablation |
| prereg_projectA_amend4_far_goal.md | 0869f717d89cfb8364ae4e9f13817e5dee05c0aae5923f0bd88e62ecbabec087 | single-gate downstream chamber, goal waypoint 3 r_goal behind the gate |
| prereg_projectA_amend5_horizon.md | 4a9c86b91e458d1df3b690ad5506fee4b44190f21a6b6fc80af3942866fca24e | horizon = HORIZON_MULT * path length to goal / V_NOM (far goal) |
| prereg_projectA_amend6_ladder.md | 8e88e0dc3f87926232351feefbaa02b6ff523f8d8b690efee7a3c196780c6300 | ladder {real,cv,learned,smith} on delayed perception, far goal; claim + go/no-go; NN retrain full-horizon |
| prereg_projectA_amend7_safety_margin_goalshape.md | a2fa24dca74c5cb89dfee1345a7a5aad6ff2ab72dfa2b500fbbe02070855e315 | d_safe 0.40 -> 0.55; flat-bottomed V_goal + hold-on-arrival; x_hat logging; learned arm, far goal |
| prereg_projectA_amend8_dsafe_revert.md | a46ee261a50af158c6a18ef5330fa606b10b207994e274a3e65fafe20c2d2790 | ablation of amend7: d_safe 0.55 -> 0.40 only; flat V_goal + hold-on-arrival kept; learned arm, far goal |
| prereg_projectA_amend9_sparse_goal.md | d8eab51fde996cb811d563a42f4a8a98a16cfa4818e98d0c434d83d953902e13 | sparse goal: spacing 0.80 (RHO_GOAL 1.80, r_goal 4.20), taller downstream chamber, d_safe 0.55; conical V_goal, no hold; learned arm |
| prereg_projectA_amend10_far_backstop.md | 03e895c339c89091667e68c0e85195bdd1e55cdc78565c21bce3ae8b25c47db4 | far backstop: V_goal cone attractor at the far disc edge (x_goal + r_goal); region, hold step, obs, CBF priority unchanged; else amend9; learned arm |
| prereg_projectA_amend11_ladder_safe.md | 39188e89f4939e3d8616787a8877c06f7f01feba3534ea84c3af6d0d68818ce1 | full ladder {real,cv,learned,smith} on the amend9 safe config (locked); open-loop MSE on datagen test seeds 110-111 of this config; best/worst-arm viz rule |

Amendments take precedence over the original for the items they name; the original is never edited.
