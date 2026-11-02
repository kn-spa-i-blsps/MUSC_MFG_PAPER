"""Amendment 31 addendum 4 — rollout runner = experiments.ladder unchanged, plus one optional cell key.

Usage: identical to experiments.ladder (python -m experiments.amend31_add4_run --config ... --mode eval --arm ... --seeds ...).
If the cell has `init_perturb_agent: a` (int), the initial x coordinate of agent a is moved by ONE f32 ulp toward +inf
(np.nextafter semantics, jnp.nextafter in the agent's dtype) right after scenario.initial_positions(key). Nothing else
changes: same PRNG key, same splits, same streams. Key absent -> ladder.main() runs byte-for-byte the existing path
(the scenario object is not touched). No existing file is modified (additive, addendum 4 rule 6)."""
from __future__ import annotations

import jax.numpy as jnp

import experiments.ladder as L

_orig_build = L.build_scenario


def _build_scenario(cfg, N):
    sc = _orig_build(cfg, N)
    a = cfg.get("init_perturb_agent")
    if a is not None:
        a = int(a)
        base = sc.initial_positions

        def initial_positions(key):
            p = base(key)
            return p.at[a, 0].set(jnp.nextafter(p[a, 0], jnp.asarray(jnp.inf, p.dtype)))

        sc.initial_positions = initial_positions   # instance attribute shadows the method (plain class)
        print(f"[add4] init_perturb_agent = {a}: x of agent {a} -> nextafter(+inf) in f32", flush=True)
    return sc


L.build_scenario = _build_scenario

if __name__ == "__main__":
    L.main()
