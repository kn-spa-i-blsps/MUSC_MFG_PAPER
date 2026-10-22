# Amendment 11 to prereg_projectA.md — full compensation ladder on the amendment-9 safe config

**Amends:** prereg/prereg_projectA.md (sha256 70a591b2c7d7f2a8c2a8d039807213974e4e9bb5b72df8de804a55b5ccdfc33a): the
setup on which the amendment-6 ladder and its go / no-go are evaluated. The original and amendments 1 (2c080dd4…),
2 (4c5015a3…), 3 (54edaa36…), 4 (0869f717…), 5 (4a9c86b9…), 6 (8e88e0dc…), 7 (a2fa24dc…), 8 (a46ee261…),
9 (d8eab51f…), 10 (03e895c3…) are NOT edited. No N sweep, no t_slot sweep, no other scenario and no factorial are
run under this amendment.
**Status:** written after the amendment-10 run and before any ladder run on this config (only the learned arm on
this config has been observed, as amendment 9). Config and report script are committed together with this file.

## Locked config (PI decision after amendments 6–10)
The amendment-9 config is the safe config: sparse goal (spacing 0.80 m, RHO_GOAL 1.80, r_goal 4.20 m at N = 100),
taller downstream chamber (clearance 1.18 m), d_safe 0.55, conical V_goal to the disc CENTRE, no hold-on-arrival,
horizon 70.4 s, Phi / m* from cache macro_N100_cw1_13e24892b8 (the amendment-9 solve; same key, no re-solve).
The amendment-10 far attractor is NOT used. Partial disc filling (the inlet-side crescent) is accepted as a known
limitation; goal filling is closed. (Facts for reference, learned arm, mean of 5 seeds: amendment 9 completeness
0.678, amendment 10 0.720.)

## Run
- experiments/ladder_fargoal_amend11.yaml: identical to experiments/ladder_fargoal_amend9.yaml except run_name
  (ladder_fargoal_amend11_N100) and the header comment. Every arm uses this file; only --arm changes.
- Arms real, cv, learned, smith (amendment-6 definitions, unchanged code), N = 100, delayed perception as in
  amendment 6, evaluation seeds 0–4 paired across arms, per-pair x_hat logged.
- The learned predictor is the amendment-6 checkpoint (SHA256 fc4a52a5…, experiments/ladder_fargoal_predictor_record.json),
  not retrained. The learned arm is re-run under the new run_name; its agreement with the amendment-9 learned run
  (same config, same seeds) is reported as a reproducibility check.
- Open-loop MSE (amendment-6 definition, experiments/openloop_mse.py unchanged): on `real`-arm histories of the
  datagen TEST seeds 110 and 111 generated in THIS config (2 extra `real` datagen rollouts, used for nothing else),
  400 sampled steps per seed, near pairs (< 1 m) and all pairs. The amendment-6 open-loop values are listed beside.

## Reported facts (experiments/amend11_report.py; no verdict, the PI evaluates against the amendment-6 go / no-go)
Arm × metric table, mean ± sd and per seed:
- completeness (final, at 57.8 s), agents outside r_goal at the end, agents in the disc (t ≥ 28 s, final 10 s);
- collision pair-steps total and in the disc; depth bands 0–0.10–0.20–0.25–0.30 m in the disc and overall;
- **leading safety metric:** worst-step p5 nearest-neighbour separation and the deep-collision rate = true pair-steps
  with d < 0.10 m per control step (overall and in the disc), always shown next to completeness: an arm does not
  count as safer through failing the task;
- min separation, CBF intervention rate, QP infeasibility (slack-active rate) overall / in the disc / outside;
- closed-loop and open-loop x_hat MSE; per-pair collision attribution (amendment 7).
Paired bootstrap over seeds (10 000 resamples, amendment-6 statistics) of arm − real for completeness, collision
pair-steps (total, in the disc, deep), worst p5, min separation, intervention, slack; reduction of mean collision
pair-steps vs real.

## Viz (rule fixed here)
- **Best arm:** among the arms whose mean completeness ≥ mean completeness of real − 0.03 (amendment-6 tolerance),
  the highest mean worst-step p5 (ties: lower deep-collision rate).
- **Worst arm:** over all arms, the lowest mean worst-step p5 (ties: higher deep-collision rate).
- For both: snapshots and animation of seed 0; one summary panel with all four arms (seed 0).
