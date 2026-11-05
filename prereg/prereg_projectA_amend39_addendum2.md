# Amendment 39 — addendum 2: the same band metric on the Config S cells (log-only, before any band number)

**Status:** written 2026-11-05 before any Config S band number was computed. Base aneks sha256 4b4ee8e8…, addendum 1
sha256 296e700e….

**How it serves the goal (rule 1):** the paper's Config S numbers (layer split +0.802 / +0.000, MFG vs CBF-only
+0.318, learned − real +0.154, delay-free 0.920 / 0.800) are on settled b = 0. Addendum 1 makes settled + 0.30 m band
the paper's headline variant; without this the paper would mix two metrics across configurations.

**Data:** the existing amend38 A2 trajectories on this workstation (amend11 ladder real/cv/learned/smith, amend33
fresh/noise_only/fresh_nomfg/fresh_nocbf, amend34 splits; 5 seeds each). Review_draft §2 said these npz were missing;
they were on this workstation's checkout (found 2026-11-05). No new rollout.

**Metric:** exactly the amend39 definition (`experiments.amend39_settled_regen.settled_metrics`): settled_{W,b} = inside
r_goal + b for ≥ 90 % of the last W s, W ∈ {3, 5, 10}, b ∈ {0, 0.30}. b = 0 must reproduce amend38 A2 exactly
(check; if not equal, stop and report).

**Analyses:** the amend38 A2 arms and the amend38 CONTRASTS list, paired by seed, bootstrap 10k (amend33 helpers).
Descriptive; no threshold, no new reading. Registered amend38 readings stand on b = 0. Where b = 0.30 and b = 0 give a
different qualitative statement, the paper says so explicitly.

**Output:** `results/amend39_configS_band/report.json`, script `experiments/amend39_configS_band.py`.
