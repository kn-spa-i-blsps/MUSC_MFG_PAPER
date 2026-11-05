# Amendment 39 — addendum 1: which settled variant the paper headlines (registered before any amend39 data)

**Status:** written 2026-11-05 before the amend39 chain was started. No amend39 rollout exists. Base aneks sha256
4b4ee8e84f65721814f955badd6aa2858d327e36301a5ad031a37f1893e5fd37 (unchanged, not edited).

**Why:** the PI brief of 2026-11-05 (NEW_AGENT_ONBOARDING §3, §5.2) states that the paper's completeness standard is
*settled with the 0.30 m tolerance band* ("goal region = disc + 0.30 m"). The base aneks registers b = 0 as primary and
b = 0.30 as secondary. Choosing the headline variant after the data exist would be goalpost-moving (CLAUDE.md rule 5),
so it is fixed here, now.

**What changes (paper presentation only):**
1. Table 1, Fig. 1, Fig. 5 and the abstract report **safe_settled and completeness_settled at W = 5 s, b = 0.30 m**
   (the PI's standard), with the b = 0 value **reported next to it** (table column or second marker) wherever space
   allows, and always in the review file and report.json.
2. safe_N is reported for both b = 0 and b = 0.30, side by side.

**What does NOT change:**
- Every registered reading of the base aneks keeps its metric. In particular the Q1 *confirmation/refutation* of
  learned safe_N = 50 is still decided at b = 0. If safe_N differs between b = 0 and b = 0.30, both are stated, and the
  text says which one is the registered test.
- No cell, seed, arm, threshold, window or analysis changes. Ever-collided is metric-independent.
- Rule-5 note of the base aneks stands: no claim may rest on b = 0.30 alone *where b = 0 disagrees*; such a
  disagreement is written out explicitly.
