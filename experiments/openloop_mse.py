"""Open-loop estimation MSE per arm (prereg amend6 sub-test): on held-out `real`-arm histories of the
datagen TEST seeds, at 400 uniformly sampled control steps (k >= 4 T_frame), every arm's x_hat from the
SAME perceive() used in closed loop; MSE over ordered pairs with true distance < 1 m, and all pairs.

    python -m experiments.openloop_mse --config experiments/ladder_fargoal.yaml --weights <npz>
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from compensation.perception import perceive
from control.params import t_frame
from experiments.ladder import field_params_for, load_predictor, setup
from experiments.smoke import ROOT

ARMS = ["fresh", "real", "cv", "learned", "smith"]
TEST_SEEDS = [110, 111]
N_SAMPLE_STEPS = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc, grid, sol, rcfg, _ = setup(cfg)
    N, dt = cfg["N"], cfg["dt_exec"]
    pp, fp = load_predictor(a.weights), field_params_for(sol)
    run_dir = os.path.join(ROOT, "results", cfg["run_name"])
    k_min = 4 * int(round(t_frame(N) / dt))
    out = {arm: {"near": [], "all": []} for arm in ARMS}
    fns = {arm: jax.jit(lambda h, k, key, arm=arm: perceive(arm, h, k, dt, N, key, predictor_params=pp,
                                                            field_params=fp)) for arm in ARMS}
    for s in TEST_SEEDS:
        h = jnp.asarray(np.load(os.path.join(run_dir, "datagen", f"pos_seed{s}.npz"))["pos"])
        ks = np.random.default_rng(7 + s).integers(k_min, h.shape[0], N_SAMPLE_STEPS)
        for k in ks:
            p = np.asarray(h[k])
            d = np.linalg.norm(p[:, None] - p[None], axis=-1)
            offd = ~np.eye(N, dtype=bool)
            near = offd & (d < 1.0)
            key = jax.random.fold_in(jax.random.PRNGKey(s), int(k))
            for arm in ARMS:
                e2 = np.sum((np.asarray(fns[arm](h, int(k), key)) - p[None]) ** 2, -1)
                out[arm]["near"].append(e2[near]); out[arm]["all"].append(e2[offd])
    res = {arm: {"mse_near1m_m2": float(np.mean(np.concatenate(v["near"]))),
                 "mse_all_m2": float(np.mean(np.concatenate(v["all"]))),
                 "n_near_pairs": int(sum(x.size for x in v["near"]))} for arm, v in out.items()}
    res["_meta"] = {"test_seeds": TEST_SEEDS, "steps_per_seed": N_SAMPLE_STEPS, "k_min": k_min, "weights": a.weights}
    json.dump(res, open(os.path.join(run_dir, "openloop_mse.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
