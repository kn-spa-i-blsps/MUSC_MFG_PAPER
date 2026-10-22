"""Learned delay predictor (prereg amend6): windows from the ladder datagen histories (full horizon,
warm 4-broadcast buffer), then the frozen phase2b recipe (ported unchanged: 2x64 ReLU MLP, Adam 1e-3,
batch 256, max 200 epochs, early stop on val MSE, patience 20, init seed 0, train-fitted normalisation
folded into the weights). Writes weights .npz + a JSON with open-loop MSE (train/val/test) and the
checkpoint SHA256.

    python -m experiments.train_predictor --run results/ladder_fargoal_N100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from control.params import T_SLOT, SIGMA_R, SIGMA_THETA, t_frame

DT_EXEC = 0.01
K_WINDOW = 4
SAMPLES_PER_SEED = 200_000
SPLITS = {"train": list(range(100, 108)), "val": [108, 109], "test": [110, 111]}
IN_DIM, HIDDEN, OUT_DIM = 20, 64, 2
LR, BATCH, MAX_EPOCHS, PATIENCE, INIT_SEED = 1e-3, 256, 200, 20, 0


def extract_windows(history, N, rng, n=SAMPLES_PER_SEED):
    """history (n_steps, N, 2) TRUE positions. Same features/target as phase2b_datagen.extract_windows,
    but k over the FULL horizon from the first warm-buffer step k_min = K_WINDOW * T_frame_steps."""
    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / DT_EXEC)))
    k_min = K_WINDOW * T_frame_steps
    k = rng.integers(k_min, history.shape[0], size=n)
    i = rng.integers(0, N, size=n)
    j = (i + rng.integers(1, N, size=n)) % N
    age = np.mod(k * DT_EXEC - j * T_SLOT, T_frame)
    k_then = k - np.round(age / DT_EXEC).astype(np.int64)
    pos_i_now = history[k, i]
    Y = history[k, j] - pos_i_now
    blocks = []
    for m in range(K_WINDOW):
        km = k_then - m * T_frame_steps
        assert np.all(km >= 0)
        rel = history[km, j] - history[km, i]
        rng_m = np.linalg.norm(rel, axis=-1) + SIGMA_R * rng.standard_normal(n)
        brg_m = np.arctan2(rel[:, 1], rel[:, 0]) + SIGMA_THETA * rng.standard_normal(n)
        ego = pos_i_now - history[km, i]
        blocks.append(np.stack([rng_m, brg_m, age + m * T_frame, ego[:, 0], ego[:, 1]], -1))
    return np.concatenate(blocks, -1).astype(np.float32), Y.astype(np.float32)


def load_split(run_dir, seeds, N):
    Xs, Ys = [], []
    for s in seeds:
        h = np.load(os.path.join(run_dir, "datagen", f"pos_seed{s}.npz"))["pos"]
        X, Y = extract_windows(h, N, np.random.default_rng(1_000_003 + s))
        Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


def init_params(key):
    def layer(k, fi, fo):
        kw, _ = jax.random.split(k)
        return jax.random.normal(kw, (fi, fo)) * jnp.sqrt(2.0 / fi), jnp.zeros((fo,))
    k1, k2, k3 = jax.random.split(key, 3)
    (W1, b1), (W2, b2), (W3, b3) = layer(k1, IN_DIM, HIDDEN), layer(k2, HIDDEN, HIDDEN), layer(k3, HIDDEN, OUT_DIM)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def forward(p, X):
    h1 = jax.nn.relu(X @ p["W1"] + p["b1"])
    h2 = jax.nn.relu(h1 @ p["W2"] + p["b2"])
    return h2 @ p["W3"] + p["b3"]


def mse(p, X, Y):
    return jnp.mean(jnp.sum((forward(p, X) - Y) ** 2, -1))


def stale_pred(X):
    return np.stack([X[:, 0] * np.cos(X[:, 1]), X[:, 0] * np.sin(X[:, 1])], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--N", type=int, default=100)
    a = ap.parse_args()
    data = {sp: load_split(a.run, seeds, a.N) for sp, seeds in SPLITS.items()}
    Xtr, Ytr = data["train"]
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-6
    nrm = {sp: (jnp.asarray((X - x_mean) / x_std), jnp.asarray((Y - y_mean) / y_std)) for sp, (X, Y) in data.items()}
    params = init_params(jax.random.PRNGKey(INIT_SEED))
    opt = optax.adam(LR)
    state = opt.init(params)

    @jax.jit
    def step(p, s, Xb, Yb):
        loss, g = jax.value_and_grad(mse)(p, Xb, Yb)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss

    rng = np.random.default_rng(INIT_SEED)
    best, best_p, since, hist = np.inf, params, 0, []
    Xn, Yn = nrm["train"]
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        perm = rng.permutation(Xn.shape[0])
        nb = Xn.shape[0] // BATCH
        tl = 0.0
        for b in range(nb):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            params, state, l = step(params, state, Xn[idx], Yn[idx])
            tl += float(l)
        vl = float(mse(params, *nrm["val"]))
        hist.append({"epoch": ep, "train_mse_norm": tl / nb, "val_mse_norm": vl})
        if vl < best - 1e-6:
            best, best_p, since = vl, jax.tree_util.tree_map(lambda x: x.copy(), params), 0
        else:
            since += 1
        print(f"epoch {ep:>3}: train={tl / nb:.6f} val={vl:.6f} best={best:.6f} since={since}", flush=True)
        if since >= PATIENCE:
            break
    wall = time.time() - t0
    W1, b1 = np.asarray(best_p["W1"]), np.asarray(best_p["b1"])
    W3, b3 = np.asarray(best_p["W3"]), np.asarray(best_p["b3"])
    folded = {"W1": W1 / x_std[:, None], "b1": b1 - np.sum(W1 * (x_mean / x_std)[:, None], 0),
              "W2": np.asarray(best_p["W2"]), "b2": np.asarray(best_p["b2"]),
              "W3": W3 * y_std[None, :], "b3": b3 * y_std + y_mean}
    folded = {k: v.astype(np.float32) for k, v in folded.items()}
    wpath = os.path.join(a.run, "predictor_weights.npz")
    np.savez(wpath, **folded)
    sha = hashlib.sha256(open(wpath, "rb").read()).hexdigest()
    fj = {k: jnp.asarray(v) for k, v in folded.items()}
    report = {"weights": wpath, "sha256": sha, "epochs": len(hist), "best_val_mse_norm": best, "wall_s": wall,
              "n": {sp: int(d[0].shape[0]) for sp, d in data.items()}, "splits": SPLITS,
              "recipe": dict(hidden=HIDDEN, lr=LR, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                             init_seed=INIT_SEED, samples_per_seed=SAMPLES_PER_SEED, k_window=K_WINDOW)}
    for sp, (X, Y) in data.items():
        err = np.asarray(forward(fj, jnp.asarray(X))) - Y
        es = stale_pred(X) - Y
        near = np.linalg.norm(Y, axis=-1) < 1.0
        report[f"mse_m2_{sp}"] = float(np.mean(np.sum(err ** 2, -1)))
        report[f"mse_m2_{sp}_near1m"] = float(np.mean(np.sum(err[near] ** 2, -1)))
        report[f"stale_mse_m2_{sp}"] = float(np.mean(np.sum(es ** 2, -1)))
        report[f"stale_mse_m2_{sp}_near1m"] = float(np.mean(np.sum(es[near] ** 2, -1)))
    report["history"] = hist
    json.dump(report, open(os.path.join(a.run, "predictor_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
