"""Phase B, gate G-pred (logs/overnight/AGENT_BRIEF_phaseB.md §A): is the per-neighbour estimation error of the frozen
delay predictor PREDICTABLE (heteroscedastic) from the predictor's own input?

Windows = experiments.overnight_p1_predictor.extract_windows (same 20-dim feature map as perceive(); near-pair
oversampling optional). For each window: e = ||f(X) - Y|| [m], f = frozen predictor (folded weights, numpy float64
forward). An error model g(Z) is trained on TRAIN seeds (val seeds for early stopping), evaluated on TEST seeds
110-111. Z = X (20) + f(X) (2) + derived (|f|, stale range, |f - stale|, T_frame) = 26 inputs. Heads: log|e| (MSE)
and the 0.9 quantile of |e| (pinball); the score s = predicted log|e| ranks pairs.

Metrics on the evaluation subset (true current distance < 2 m, the pairs the CBF acts on), per N and pooled:
  r2_abs   R^2 of exp(pred log|e|) as a predictor of |e|   (variance explained, linear scale)
  r2_log   R^2 in log space
  spearman rank correlation s vs |e|
  auc_top10   AUC of s for "|e| in the top decile (per N)"
  lift_top1   fraction of the top-1% errors whose s is in the top 10% of s  (random = 0.10)
  r2_tail     R^2 (log) on the subset with |e| above the per-N 90th percentile
Baselines with the same recipe on restricted inputs: 'age' (the 4 ages + T_frame only), 'range' (stale range only).

    python -m experiments.phaseB_gpred --tag v3 --weights results/overnight_p1/predictor_v3/predictor_weights.npz \
        --src datagenL --out results/phaseB/gpred_v3_datagenL.json [--hidden 128] [--near_frac 0.5] [--per_N]
    python -m experiments.phaseB_gpred --proxy ...   # local proxy: N=100 amend11 geometry, results/ladder_fargoal_N100
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

from control.params import t_frame
from experiments.overnight_p1_predictor import NS, SPLITS, datagen_dir, extract_windows
from experiments.smoke import ROOT

LR, BATCH, MAX_EPOCHS, PATIENCE, INIT_SEED = 1e-3, 512, 200, 15, 0
EVAL_R = 2.0
TAU_Q = 0.9


def np_forward(w, X):
    X = X.astype(np.float64)
    h1 = np.maximum(X @ w["W1"] + w["b1"], 0)
    h2 = np.maximum(h1 @ w["W2"] + w["b2"], 0)
    return h2 @ w["W3"] + w["b3"]


def features(X, F, N):
    stale = np.stack([X[:, 0] * np.cos(X[:, 1]), X[:, 0] * np.sin(X[:, 1])], -1)
    der = np.stack([np.linalg.norm(F, axis=-1), X[:, 0], np.linalg.norm(F - stale, axis=-1),
                    np.full(len(X), t_frame(N))], -1)
    return np.concatenate([X, F, der], -1)


INPUT_SETS = {"full": None, "age": [2, 7, 12, 17, 25], "range": [0]}


def load(dirs_by_N, seeds, n, near_frac, w):
    Z, E, Nv, D = [], [], [], []
    for N, d in dirs_by_N.items():
        for s in seeds:
            h = np.load(os.path.join(d, f"pos_seed{s}.npz"))["pos"].astype(np.float64)
            X, Y = extract_windows(h, N, np.random.default_rng(2_000_003 + 7919 * N + s), n, near_frac)
            F = np_forward(w, X)
            Z.append(features(X, F, N)); E.append(np.linalg.norm(F - Y, axis=-1))
            Nv.append(np.full(len(X), N, np.int32)); D.append(np.linalg.norm(Y, axis=-1))
    return [np.concatenate(v) for v in (Z, E, Nv, D)]


def mlp_init(key, din, hidden, depth, dout=2):
    ks = jax.random.split(key, depth + 1)
    dims = [din] + [hidden] * depth + [dout]
    return [(jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / a), jnp.zeros((b,))) for k, a, b in zip(ks, dims, dims[1:])]


def mlp(p, Z):
    for W, b in p[:-1]:
        Z = jax.nn.relu(Z @ W + b)
    W, b = p[-1]
    return Z @ W + b


def train_error_model(Ztr, Etr, Zva, Eva, hidden, depth):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    ly = np.log(Etr + 1e-4)
    lmu, lsd = ly.mean(), ly.std() + 1e-6
    nz = lambda Z: jnp.asarray((Z - mu) / sd, jnp.float32)
    tgt = lambda E: jnp.asarray((np.log(E + 1e-4) - lmu) / lsd, jnp.float32)
    qy = lambda E: jnp.asarray((E - Etr.mean()) / Etr.std(), jnp.float32)

    def loss(p, Z, T, Q):
        o = mlp(p, Z)
        r = Q - o[:, 1]
        return jnp.mean((o[:, 0] - T) ** 2) + jnp.mean(jnp.maximum(TAU_Q * r, (TAU_Q - 1) * r))

    p = mlp_init(jax.random.PRNGKey(INIT_SEED), Ztr.shape[1], hidden, depth)
    opt = optax.adam(LR)
    st = opt.init(p)
    Xn, Tn, Qn = nz(Ztr), tgt(Etr), qy(Etr)
    Xv, Tv, Qv = nz(Zva), tgt(Eva), qy(Eva)

    @jax.jit
    def epoch(p, st, perm):
        def body(c, idx):
            p, st = c
            l, g = jax.value_and_grad(loss)(p, Xn[idx], Tn[idx], Qn[idx])
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l
        (p, st), ls = jax.lax.scan(body, (p, st), perm)
        return p, st, ls.mean()

    vloss = jax.jit(loss)
    rng = np.random.default_rng(INIT_SEED)
    nb = Xn.shape[0] // BATCH
    best, bp, since, ep = np.inf, p, 0, 0
    for ep in range(MAX_EPOCHS):
        perm = jnp.asarray(rng.permutation(Xn.shape[0])[:nb * BATCH].reshape(nb, BATCH))
        p, st, tl = epoch(p, st, perm)
        vl = float(vloss(p, Xv, Tv, Qv))
        if vl < best - 1e-5:
            best, bp, since = vl, p, 0
        else:
            since += 1
        if ep % 10 == 0:
            print(f"  epoch {ep}: train {float(tl):.4f} val {vl:.4f} best {best:.4f}", flush=True)
        if since >= PATIENCE:
            break

    def predict(Z):
        o = np.asarray(mlp(bp, nz(Z)), np.float64)
        return o[:, 0] * lsd + lmu, o[:, 1] * Etr.std() + Etr.mean()
    return predict, {"epochs": ep + 1, "best_val": best}


def r2(y, yh):
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def auc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score)
    npos = label.sum()
    nneg = label.size - npos
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(E, slog, q90):
    from scipy.stats import spearmanr
    le = np.log(E + 1e-4)
    t10 = E > np.quantile(E, 0.9)
    t1 = E > np.quantile(E, 0.99)
    s_top10 = slog > np.quantile(slog, 0.9)
    return {"n": int(E.size), "mean_e_m": float(E.mean()), "p99_e_m": float(np.quantile(E, 0.99)),
            "r2_abs": r2(E, np.exp(slog)), "r2_log": r2(le, slog), "spearman": float(spearmanr(slog, E)[0]),
            "auc_top10": auc(slog, t10), "lift_top1": float(s_top10[t1].mean()),
            "r2_tail_log": r2(le[t10], slog[t10]),
            "q90_coverage_raw": float(np.mean(E <= q90)),
            "e_ratio_top_vs_bottom_sdecile": float(E[s_top10].mean() / E[slog < np.quantile(slog, 0.1)].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default="datagenL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--near_frac", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=60_000)
    ap.add_argument("--per_N", action="store_true", help="also train one error model per N (diagnostic ceiling)")
    ap.add_argument("--proxy", action="store_true", help="local proxy: N=100 amend11 geometry histories only")
    a = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    t0 = time.time()
    w = {k: v.astype(np.float64) for k, v in np.load(a.weights).items()}
    dirs = ({100: os.path.join(ROOT, "results", "ladder_fargoal_N100", "datagen")} if a.proxy
            else {N: datagen_dir(N, a.src) for N in NS})
    data = {sp: load(dirs, seeds, a.samples if sp == "train" else a.samples // 2, a.near_frac, w)
            for sp, seeds in SPLITS.items()}
    print("loaded", {sp: v[0].shape for sp, v in data.items()}, f"{time.time() - t0:.0f}s", flush=True)
    res = {"tag": a.tag, "proxy": a.proxy, "src": "ladder_fargoal_N100" if a.proxy else a.src,
           "weights": a.weights, "weights_sha256": hashlib.sha256(open(a.weights, "rb").read()).hexdigest(),
           "args": vars(a), "eval_subset": f"true distance < {EVAL_R} m", "matmul_precision": "highest",
           "Ns": sorted(dirs), "models": {}}

    def evaluate(pred, cols, Nsel=None, split="test"):
        Z, E, Nv, D = data[split]
        m = D < EVAL_R
        if Nsel is not None:
            m &= Nv == Nsel
        Zs = Z[m] if cols is None else Z[m][:, cols]
        slog, q90 = pred(Zs)
        out = {"pooled": metrics(E[m], slog, q90)} if Nsel is None else {}
        for N in ([Nsel] if Nsel is not None else sorted(dirs)):
            k = Nv[m] == N
            out[str(N)] = metrics(E[m][k], slog[k], q90[k])
        return out

    def fit(cols, Nsel=None):
        sel = lambda sp: (np.ones(len(data[sp][1]), bool) if Nsel is None else data[sp][2] == Nsel)
        Zc = lambda sp: data[sp][0][sel(sp)] if cols is None else data[sp][0][sel(sp)][:, cols]
        return train_error_model(Zc("train"), data["train"][1][sel("train")], Zc("val"), data["val"][1][sel("val")],
                                 a.hidden, a.depth)

    for name, cols in INPUT_SETS.items():
        print(f"error model '{name}'", flush=True)
        pred, info = fit(cols)
        res["models"][name] = {"train": info, "test": evaluate(pred, cols), "val": evaluate(pred, cols, split="val")}
        print(json.dumps({k: {m: round(v[m], 3) for m in ("r2_abs", "spearman", "auc_top10", "lift_top1")}
                          for k, v in res["models"][name]["test"].items()}), flush=True)
    if a.per_N and not a.proxy:
        res["models"]["full_perN"] = {}
        for N in sorted(dirs):
            pred, info = fit(None, N)
            res["models"]["full_perN"][str(N)] = {"train": info, "test": evaluate(pred, None, N)[str(N)],
                                                  "val": evaluate(pred, None, N, "val")[str(N)]}
    res["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
