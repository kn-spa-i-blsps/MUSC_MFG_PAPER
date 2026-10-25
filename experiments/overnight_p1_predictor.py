"""Overnight Phase 1 (EXPLORATORY, not a reported run): staleness-conditioned predictor trained on POOLED real-arm
histories across N in {25, 50, 100, 150, 200} on the amend21 fixed geometry (geometry_N 230, gate 2.309),
precision=highest. Evaluation runs that produce reported numbers are pre-registered separately (amend22).

Staleness input: the amend6/phase2b 20-dim window already carries each neighbour's staleness explicitly, per block m,
as age_m = tdma_age_j + m*T_frame (feature index 2 + 5m); T_frame = age_1 - age_0 is therefore also explicit. The
N=100-only predictor never saw ages outside N=100's range [0, 4*0.5 s) on the T_frame=0.5 s lattice -> OOD at N=25
(T_frame 0.125 s). The fix tested here is the training distribution (pooled, N-balanced), same feature map and same
perceive() code path, so the closed loop needs no code change (only new weights).

    python -m experiments.overnight_p1_predictor write-cells            # experiments/overnight_p1_cells/*.yaml
    python -m experiments.overnight_p1_predictor train --tag v1 [--hidden 64] [--near_frac 0] [--samples 100000]
    python -m experiments.overnight_p1_predictor openloop --weights <npz> [--weights_old <npz>] --out <json>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from control.params import T_SLOT, t_frame
from experiments.smoke import ROOT
from experiments.train_predictor import DT_EXEC, K_WINDOW, stale_pred
from control.params import SIGMA_R, SIGMA_THETA

NS = [25, 50, 100, 150, 200]
SPLITS = {"train": list(range(100, 108)), "val": [108, 109], "test": [110, 111]}
CELL_DIR = os.path.join(ROOT, "experiments", "overnight_p1_cells")
RES = os.path.join(ROOT, "results", "overnight_p1")
OLD_WEIGHTS = os.path.join(ROOT, "results", "ladder_fargoal_N100", "predictor_weights.npz")
LR, BATCH, MAX_EPOCHS, PATIENCE, INIT_SEED = 1e-3, 256, 200, 20, 0


def datagen_dir(N, src="datagen"):
    """src 'datagen' = real-arm histories (v1-v3); 'datagenL' = closed-loop learned-v3 d_safe 0.70 histories (v4+)."""
    return os.path.join(RES, f"{src}_N{N:03d}", "datagen")


def cmd_write_cells(a):
    """datagen cells = the amend21 real cells (same geometry, cache key, d_safe 0.55, precision highest), new run_name;
    identity cells = the amend21 real cells with a new run_name (re-run of amend21 real seed 0 on this pod)."""
    os.makedirs(CELL_DIR, exist_ok=True)
    for N in NS:
        src = os.path.join(ROOT, "experiments", "amend21_cells", f"N{N:03d}_real.yaml")
        for kind in ("datagen", "identity", "datagenL"):
            c = copy.deepcopy(yaml.safe_load(open(src)))
            c["run_name"] = f"overnight_p1/{kind}_N{N:03d}"
            if kind == "datagenL":
                c["cbf"]["d_safe"] = 0.70
            p = os.path.join(CELL_DIR, f"{kind}_N{N:03d}.yaml")
            with open(p, "w") as f:
                f.write(f"# overnight P1 {kind} cell: amend21_cells/N{N:03d}_real.yaml with run_name changed\n")
                yaml.safe_dump(c, f, sort_keys=False)
            print(p)


def extract_windows(history, N, rng, n, near_frac=0.0, near_r=3.0):
    """experiments.train_predictor.extract_windows generalised to any N (T_frame = N*T_SLOT) with optional near-pair
    oversampling: a fraction near_frac of the (k, i) draws takes j among i's neighbours with true current distance
    < near_r (if i has any), the rest uniform j != i (the original recipe). Feature order unchanged."""
    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / DT_EXEC)))
    k_min = K_WINDOW * T_frame_steps
    k = rng.integers(k_min, history.shape[0], size=n)
    i = rng.integers(0, N, size=n)
    j = (i + rng.integers(1, N, size=n)) % N
    if near_frac > 0:
        sel = np.nonzero(rng.random(n) < near_frac)[0]
        P = history[k[sel]]                                        # (m, N, 2)
        d = np.linalg.norm(P - P[np.arange(sel.size), i[sel]][:, None], axis=-1)
        d[np.arange(sel.size), i[sel]] = np.inf
        ok = d < near_r
        has = ok.any(1)
        w = ok / np.maximum(ok.sum(1, keepdims=True), 1)
        u = rng.random(sel.size)
        jn = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), N - 1)
        j[sel[has]] = jn[has]
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


def load_split(seeds, n, near_frac, sources=("datagen",)):
    Xs, Ys, Ns = [], [], []
    for N in NS:
        for si, src in enumerate(sources):
          for s in seeds:
            h = np.load(os.path.join(datagen_dir(N, src), f"pos_seed{s}.npz"))["pos"].astype(np.float64)
            X, Y = extract_windows(h, N, np.random.default_rng(1_000_003 + 7919 * N + s + 104_729 * si),
                                   n // len(sources), near_frac)
            Xs.append(X); Ys.append(Y); Ns.append(np.full(len(X), N, np.int32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ns)


def init_params(key, hidden, in_dim=5 * K_WINDOW, out_dim=2):
    def layer(k, fi, fo):
        kw, _ = jax.random.split(k)
        return jax.random.normal(kw, (fi, fo)) * jnp.sqrt(2.0 / fi), jnp.zeros((fo,))
    k1, k2, k3 = jax.random.split(key, 3)
    (W1, b1), (W2, b2), (W3, b3) = layer(k1, in_dim, hidden), layer(k2, hidden, hidden), layer(k3, hidden, out_dim)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def forward(p, X):
    h1 = jax.nn.relu(X @ p["W1"] + p["b1"])
    h2 = jax.nn.relu(h1 @ p["W2"] + p["b2"])
    return h2 @ p["W3"] + p["b3"]


def cmd_train(a):
    jax.config.update("jax_default_matmul_precision", "highest")
    out = os.path.join(RES, f"predictor_{a.tag}")
    os.makedirs(out, exist_ok=True)
    data = {sp: load_split(seeds, a.samples if sp == "train" else a.samples // 2, a.near_frac, tuple(a.sources))
            for sp, seeds in SPLITS.items()}
    Xtr, Ytr, _ = data["train"]
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-6
    nrm = {sp: (jnp.asarray((X - x_mean) / x_std), jnp.asarray((Y - y_mean) / y_std)) for sp, (X, Y, _) in data.items()}
    loss = lambda p, X, Y: jnp.mean(jnp.sum((forward(p, X) - Y) ** 2, -1))
    params = init_params(jax.random.PRNGKey(INIT_SEED), a.hidden)
    opt = optax.adam(a.lr)
    state = opt.init(params)

    @jax.jit
    def epoch(p, s, Xn, Yn, perm):
        def body(c, idx):
            p, s = c
            l, g = jax.value_and_grad(loss)(p, Xn[idx], Yn[idx])
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), l
        (p, s), ls = jax.lax.scan(body, (p, s), perm)
        return p, s, ls.mean()

    val_loss = jax.jit(loss)
    rng = np.random.default_rng(INIT_SEED)
    best, best_p, since, hist = np.inf, params, 0, []
    Xn, Yn = nrm["train"]
    nb = Xn.shape[0] // BATCH
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        perm = jnp.asarray(rng.permutation(Xn.shape[0])[:nb * BATCH].reshape(nb, BATCH))
        params, state, tl = epoch(params, state, Xn, Yn, perm)
        vl = float(val_loss(params, *nrm["val"]))
        hist.append({"epoch": ep, "train_mse_norm": float(tl), "val_mse_norm": vl})
        if vl < best - 1e-6:
            best, best_p, since = vl, jax.tree_util.tree_map(lambda x: x.copy(), params), 0
        else:
            since += 1
        print(f"epoch {ep:>3}: train={float(tl):.6f} val={vl:.6f} best={best:.6f} since={since}", flush=True)
        if since >= PATIENCE:
            break
    W1, b1 = np.asarray(best_p["W1"]), np.asarray(best_p["b1"])
    W3, b3 = np.asarray(best_p["W3"]), np.asarray(best_p["b3"])
    folded = {"W1": W1 / x_std[:, None], "b1": b1 - np.sum(W1 * (x_mean / x_std)[:, None], 0),
              "W2": np.asarray(best_p["W2"]), "b2": np.asarray(best_p["b2"]),
              "W3": W3 * y_std[None, :], "b3": b3 * y_std + y_mean}
    folded = {k: v.astype(np.float32) for k, v in folded.items()}
    wpath = os.path.join(out, "predictor_weights.npz")
    np.savez(wpath, **folded)
    sha = hashlib.sha256(open(wpath, "rb").read()).hexdigest()
    fj = {k: jnp.asarray(v) for k, v in folded.items()}
    rep = {"weights": os.path.relpath(wpath, ROOT), "sha256": sha, "epochs": len(hist), "best_val_mse_norm": best,
           "wall_s": time.time() - t0, "NS": NS, "splits": SPLITS, "matmul_precision": "highest",
           "recipe": dict(hidden=a.hidden, lr=a.lr, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                          init_seed=INIT_SEED, samples_per_seed_per_N=a.samples, near_frac=a.near_frac,
                          sources=a.sources,
                          k_window=K_WINDOW), "window_mse_m2": {}}
    for sp, (X, Y, Nv) in data.items():
        err = np.asarray(forward(fj, jnp.asarray(X))) - Y
        es = stale_pred(X) - Y
        near = np.linalg.norm(Y, axis=-1) < 1.0
        for N in NS:
            m = Nv == N
            rep["window_mse_m2"][f"{sp}_N{N}"] = {
                "learned": float(np.mean(np.sum(err[m] ** 2, -1))), "stale": float(np.mean(np.sum(es[m] ** 2, -1))),
                "learned_near1m": float(np.mean(np.sum(err[m & near] ** 2, -1))),
                "stale_near1m": float(np.mean(np.sum(es[m & near] ** 2, -1))), "n_near1m": int((m & near).sum())}
    rep["history"] = hist
    json.dump(rep, open(os.path.join(out, "predictor_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=1), flush=True)


def cmd_openloop(a):
    """Open-loop x_hat MSE per N with the SAME perceive() used in closed loop, on held-out datagen TEST seeds 110-111
    of this geometry, 400 sampled steps per seed (k >= 4 T_frame), pairs with true distance < 1 m ('near') and all."""
    jax.config.update("jax_default_matmul_precision", "highest")
    from compensation.perception import perceive
    from experiments.ladder import load_predictor
    preds = {"learned_new": load_predictor(a.weights)}
    if a.weights_old:
        preds["learned_old"] = load_predictor(a.weights_old)
    arms = ["real", "cv", "noise_only"] + list(preds)
    res = {"src": a.src, "weights": a.weights, "weights_sha256": hashlib.sha256(open(a.weights, "rb").read()).hexdigest(),
           "weights_old": a.weights_old, "test_seeds": SPLITS["test"], "steps_per_seed": 400,
           "matmul_precision": jax.config.jax_default_matmul_precision, "per_N": {}}
    for N in NS:
        k_min = 4 * int(round(t_frame(N) / DT_EXEC))
        fns = {}
        for arm in arms:
            cond, pp = ("learned", preds[arm]) if arm.startswith("learned") else (arm, None)   # noqa
            fns[arm] = jax.jit(lambda h, k, key, cond=cond, pp=pp, N=N: perceive(cond, h, k, DT_EXEC, N, key,
                                                                                   predictor_params=pp))
        acc = {arm: {"near": [], "near2": [], "all": []} for arm in arms}
        for s in SPLITS["test"]:
            h = jnp.asarray(np.load(os.path.join(datagen_dir(N, a.src), f"pos_seed{s}.npz"))["pos"], jnp.float32)
            ks = np.random.default_rng(7 + s).integers(k_min, h.shape[0], 400)
            for k in ks:
                p = np.asarray(h[k])
                d = np.linalg.norm(p[:, None] - p[None], axis=-1)
                offd = ~np.eye(N, dtype=bool)
                key = jax.random.fold_in(jax.random.PRNGKey(s), int(k))
                for arm in arms:
                    e2 = np.sum((np.asarray(fns[arm](h, int(k), key)) - p[None]) ** 2, -1)
                    acc[arm]["near"].append(e2[offd & (d < 1.0)])
                    acc[arm]["near2"].append(e2[offd & (d < 2.0)])
                    acc[arm]["all"].append(e2[offd])
        res["per_N"][str(N)] = {arm: {"mse_near1m_m2": float(np.mean(np.concatenate(v["near"]))),
                                      "p99_near1m_m2": float(np.percentile(np.concatenate(v["near"]), 99)),
                                      "p999_near1m_m2": float(np.percentile(np.concatenate(v["near"]), 99.9)),
                                      "mse_near2m_m2": float(np.mean(np.concatenate(v["near2"]))),
                                      "mse_all_m2": float(np.mean(np.concatenate(v["all"]))),
                                      "n_near1m": int(sum(x.size for x in v["near"]))} for arm, v in acc.items()}
        print(N, json.dumps({arm: round(v["mse_near1m_m2"], 4) for arm, v in res["per_N"][str(N)].items()}), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "train", "openloop"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--near_frac", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--weights")
    ap.add_argument("--weights_old", default=None)
    ap.add_argument("--out")
    ap.add_argument("--sources", nargs="+", default=["datagen"])
    ap.add_argument("--src", default="datagen")
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "train": cmd_train, "openloop": cmd_openloop}[a.cmd](a)


if __name__ == "__main__":
    main()
