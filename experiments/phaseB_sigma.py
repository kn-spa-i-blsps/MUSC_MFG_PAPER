"""Phase B step 2 (prereg amend31): the frozen sigma head + split-conformal calibration, gate G-cal.

Error model g = the G-pred recipe D (pooled N-balanced, near_frac 0.5, 60k windows / seed / N, MLP 2x128, heads log|e|
+ q0.9; experiments.phaseB_gpred) on datagenL, with a 4-way seed split: TRAIN 100-105, VAL 106-107 (early stopping),
CAL 108-109 (conformal), TEST 110-111 (G-cal). CAL and TEST windows are drawn uniformly (near_frac 0, 200k / seed / N)
and restricted to true distance < 2 m = the natural distribution of near pairs.

Normalized split conformal, Mondrian per N: score r = |e| / exp(g_log); q_N(alpha) = the ceil((n+1)(1-alpha))-th
smallest CAL score of that N; sigma_c = q_N(alpha) * exp(g_log) covers |e| with probability >= 1 - alpha (marginal,
exchangeable windows). A pooled-N q(alpha) is reported for comparison. Registered §C calibration fix, always computed:
Mondrian per (N, sigma-quintile): the CAL quintile edges of g_log within N split CAL into 5 bins, one q per bin
(q_bins / q_edges; used by the sigma_bin arm). Also reported: coverage on the pairs where the margin binds,
0.30 + sigma_c > 0.70.

    python -m experiments.phaseB_sigma --out results/phaseB/sigma_v3      # -> sigma_weights.npz, calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import jax
import numpy as np

from experiments.overnight_p1_predictor import NS, datagen_dir
from experiments.phaseB_gpred import EVAL_R, load, metrics, train_error_model

SPLIT = {"train": list(range(100, 106)), "val": [106, 107], "cal": [108, 109], "test": [110, 111]}
ALPHAS = (0.10, 0.05, 0.01)
TOL = {0.10: 0.02, 0.05: 0.02, 0.01: 0.01}   # G-cal marginal tolerance |coverage - (1 - alpha)|
DECILE_SLACK = 0.05                          # G-cal conditional: per sigma-decile coverage >= 1 - alpha - 0.05 (alpha 0.10)


def conformal_q(r, alpha):
    n = r.size
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(r)[k - 1])


def qfun_scalar(q):
    return lambda N, s: np.full(s.shape, q[N])


def qfun_bins(edges, vals):
    return lambda N, s: np.asarray(vals[N])[np.searchsorted(edges[N], s, side="right")]


def coverage_table(E, slog, qf, Nv, age):
    """Marginal coverage per N, per sigma-decile (within N), per age tercile (within N), and where the margin binds."""
    out = {}
    for N in NS:
        k = Nv == N
        e, s = E[k], slog[k]
        sig = qf(N, s) * np.exp(s)
        cov = e <= sig
        dec = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11)), s, side="right") - 1, 0, 9)
        a = age[k]
        ter = np.clip(np.searchsorted(np.quantile(a, [0, 1 / 3, 2 / 3, 1]), a, side="right") - 1, 0, 2)
        out[str(N)] = {"n": int(k.sum()), "coverage": float(cov.mean()), "sigma_c_mean_m": float(sig.mean()),
                       "sigma_c_p90_m": float(np.quantile(sig, 0.9)),
                       "decile_coverage": [float(cov[dec == i].mean()) for i in range(10)],
                       "age_tercile_coverage": [float(cov[ter == i].mean()) for i in range(3)],
                       "binding_frac": float((sig > 0.40).mean()),
                       "binding_coverage": float(cov[sig > 0.40].mean()) if (sig > 0.40).any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/overnight_p1/predictor_v3/predictor_weights.npz")
    ap.add_argument("--src", default="datagenL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=60_000)
    ap.add_argument("--cal_samples", type=int, default=200_000)
    a = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    t0 = time.time()
    w = {k: v.astype(np.float64) for k, v in np.load(a.weights).items()}
    dirs = {N: datagen_dir(N, a.src) for N in NS}
    data = {sp: load(dirs, seeds, a.samples if sp == "train" else a.samples // 2, 0.5, w)
            for sp, seeds in SPLIT.items() if sp in ("train", "val")}
    data.update({sp: load(dirs, SPLIT[sp], a.cal_samples, 0.0, w) for sp in ("cal", "test")})
    print("loaded", {sp: v[0].shape for sp, v in data.items()}, f"{time.time() - t0:.0f}s", flush=True)
    pred, info = train_error_model(data["train"][0], data["train"][1], data["val"][0], data["val"][1], 128, 2)
    os.makedirs(a.out, exist_ok=True)
    wpath = os.path.join(a.out, "sigma_weights.npz")
    np.savez(wpath, **pred.export)

    def near(sp):
        Z, E, Nv, D = data[sp]
        m = D < EVAL_R
        slog, q90 = pred(Z[m])
        return E[m], slog, q90, Nv[m], Z[m][:, 2]   # column 2 = age of the m=0 broadcast
    Ec, sc, _, Nc, _ = near("cal")
    Et, st, q90t, Nt, at = near("test")
    rc = Ec / np.exp(sc)
    res = {"weights_v3": a.weights, "weights_v3_sha256": hashlib.sha256(open(a.weights, "rb").read()).hexdigest(),
           "sigma_weights": wpath, "sigma_weights_sha256": hashlib.sha256(open(wpath, "rb").read()).hexdigest(),
           "split": SPLIT, "src": a.src, "train": info, "matmul_precision": "highest", "eval_subset": f"d < {EVAL_R} m",
           "n_cal": {str(N): int((Nc == N).sum()) for N in NS},
           "test_metrics": {"pooled": metrics(Et, st, q90t), **{str(N): metrics(Et[Nt == N], st[Nt == N], q90t[Nt == N])
                                                                 for N in NS}},
           "q": {}, "q_pooled": {}, "q_edges": {}, "q_bins": {}, "test_coverage": {}, "test_coverage_pooled_q": {},
           "test_coverage_bins": {}, "gcal": {}}
    edges = {N: np.quantile(sc[Nc == N], [0.2, 0.4, 0.6, 0.8]) for N in NS}
    res["q_edges"] = {str(N): [float(x) for x in v] for N, v in edges.items()}
    for al in ALPHAS:
        qN = {N: conformal_q(rc[Nc == N], al) for N in NS}
        qp = conformal_q(rc, al)
        qb = {N: [conformal_q(rc[Nc == N][np.searchsorted(edges[N], sc[Nc == N], side="right") == b], al)
                  for b in range(5)] for N in NS}
        res["q"][str(al)] = {str(N): v for N, v in qN.items()}
        res["q_pooled"][str(al)] = qp
        res["q_bins"][str(al)] = {str(N): v for N, v in qb.items()}
        res["test_coverage"][str(al)] = coverage_table(Et, st, qfun_scalar(qN), Nt, at)
        res["test_coverage_pooled_q"][str(al)] = coverage_table(Et, st, qfun_scalar({N: qp for N in NS}), Nt, at)
        res["test_coverage_bins"][str(al)] = coverage_table(Et, st, qfun_bins(edges, qb), Nt, at)

    def gate(tab):   # G-cal (registered): marginal per N within TOL, every alpha; per decile >= 0.85 at alpha 0.10
        marg = {f"{al}/{N}": abs(tab[str(al)][str(N)]["coverage"] - (1 - al)) <= TOL[al] for al in ALPHAS for N in NS}
        cond = {str(N): min(tab["0.1"][str(N)]["decile_coverage"]) >= 0.90 - DECILE_SLACK for N in NS}
        return {"marginal_ok": marg, "conditional_ok": cond, "pass": all(marg.values()) and all(cond.values()),
                "marginal_pass": all(marg.values())}
    g1, g2 = gate(res["test_coverage"]), gate(res["test_coverage_bins"])
    res["gcal"] = {"per_N": g1, "per_N_sigma_bin": g2,
                   "verdict": "PASS" if g1["pass"] else ("PASS_AFTER_FIX" if g2["pass"] else
                              ("PARTIAL" if g1["marginal_pass"] or g2["marginal_pass"] else "FAIL")),
                   "primary_gwin_arm": "sigma_a0.05" if g1["pass"] else ("sigma_bin_a0.05" if g2["pass"] else
                                                                          "sigma_a0.05")}
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    for al in ALPHAS:
        print(f"alpha {al}: q_N", {N: round(v, 3) for N, v in res["q"][str(al)].items()},
              "coverage", {N: round(v["coverage"], 4) for N, v in res["test_coverage"][str(al)].items()}, flush=True)
        print(f"alpha {al}: binned coverage", {N: round(v["coverage"], 4) for N, v in
                                                res["test_coverage_bins"][str(al)].items()}, flush=True)
    print("G-cal", res["gcal"]["verdict"], "primary G-win arm", res["gcal"]["primary_gwin_arm"],
          json.dumps({"per_N": g1["conditional_ok"], "bins": g2["conditional_ok"]}), flush=True)


if __name__ == "__main__":
    main()
