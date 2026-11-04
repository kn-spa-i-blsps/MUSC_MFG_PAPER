"""Amendment-38 (log-only): A1 settled bound from registered ever_collided, A2 exact settled recompute at N=100,
B non-settler classification on amend33 fresh / noise_only.

    python -m experiments.amend38_settled_recompute report
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import yaml

from experiments import amend33_n100_baseline_cpu as a33
from experiments.amend12_report import agent_facts
from experiments.smoke import ROOT, build_scenario

RUN = "amend38_settled_recompute"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend38_settled_recompute_nonsettlers.md")
DT, FRAC, WINDOWS, W0 = 0.01, 0.90, (3.0, 5.0, 10.0), 5.0
SEEDS = [0, 1, 2, 3, 4]
L11 = "results/ladder_fargoal_amend11_N100"
A33 = "results/amend33_n100_baseline_cpu"
A34 = "results/amend34_layer_delay_cpu"
ARMS = {  # name -> dir  (all: amend11 geometry, N=100, d_safe 0.55, terminal cost 0)
    "real": f"{L11}/real", "cv": f"{L11}/cv", "learned": f"{L11}/learned", "smith": f"{L11}/smith",
    "fresh": f"{A33}/fresh/fresh", "noise_only": f"{A33}/noise_only/noise_only",
    "fresh_nomfg": f"{A33}/fresh_nomfg/fresh", "fresh_nocbf": f"{A33}/fresh_nocbf/fresh",
    "split_noise_only-real (MFG clean, CBF delayed)": f"{A34}/ab/split_noise_only-real",
    "split_real-noise_only (MFG delayed, CBF clean)": f"{A34}/ab/split_real-noise_only",
    "split_real-real (check)": f"{A34}/check_real/split_real-real",
}
CONTRASTS = [("learned", "real"), ("cv", "real"), ("smith", "real"), ("noise_only", "real"),
             ("split_real-noise_only (MFG delayed, CBF clean)", "real"),
             ("split_noise_only-real (MFG clean, CBF delayed)", "real"),
             ("fresh", "fresh_nomfg")]
A1_REPORTS = {"amend22": "results/amend22_N_crossover_pooled/report.json",
              "amend24": "results/amend24_mechanism_control/report.json",
              "amend27": "results/amend27_ladder_layers/report.json"}


def load(d, s):
    p = os.path.join(ROOT, d, f"logs_seed{s}.npz")
    return np.load(p) if os.path.exists(p) else None


def arm_rows(d, sc):
    g = np.asarray(sc.goal)
    out = []
    for s in SEEDS:
        z = load(d, s)
        if z is None:
            continue
        pos, pT = z["pos"].astype(np.float64), z["pos_final"].astype(np.float64)
        _, ps, arrT, coll = agent_facts({"pos": pos, "pos_final": pT}, sc)
        inside = np.linalg.norm(np.concatenate([pos, pT[None]]) - g, axis=-1) <= sc.r_goal
        reg = json.load(open(os.path.join(ROOT, d, f"metrics_seed{s}.json")))
        r = {"seed": s, "completeness_T": float(arrT.mean()), "ever_collided": float(coll.mean()),
             "collision_pair_steps": ps, "registered_pair_steps": reg["collision_pair_steps"],
             "registered_completeness": reg["completeness_final"], "safe_success_T": float((arrT & ~coll).mean())}
        for W in WINDOWS:
            st = inside[-(int(round(W / DT)) + 1):].mean(0) >= FRAC
            r[f"completeness_settled_{W:g}s"] = float(st.mean())
            r[f"safe_settled_{W:g}s"] = float((st & ~coll).mean())
        out.append(r)
    return out


def classify(z, sc):
    """B: per non-settled agent class, 5 s window."""
    g = np.asarray(sc.goal)
    pos = np.concatenate([z["pos"], z["pos_final"][None]]).astype(np.float64)
    n = int(round(W0 / DT)) + 1
    P = pos[-n:]                                                      # (n, N, 2)
    rr = np.linalg.norm(P - g, axis=-1)
    d = rr - sc.r_goal                                                # signed distance outside
    inside = d <= 0
    settled = inside.mean(0) >= FRAC
    rad = (P[:-1] - g) / np.maximum(rr[:-1, :, None], 1e-9)           # unit outward, aligned to u (n-1 steps)
    un, u = z["u_nom"][-(n - 1):], z["u"][-(n - 1):]
    un_r, u_r = (un * rad).sum(-1), (u * rad).sum(-1)                 # < 0 = inward
    out = []
    for i in np.where(~settled)[0]:
        cross = int((np.diff(inside[:, i].astype(int)) != 0).sum())
        dd = d[-1, i] - d[0, i]
        k1 = int(round(1.0 / DT))
        v_app = -(d[-1, i] - d[-1 - k1, i]) / 1.0                     # approach speed over the last 1 s (>0 = in)
        rec = {"agent": int(i), "crossings": cross, "d_T": float(d[-1, i]), "delta_d_5s": float(dd),
               "inside_frac": float(inside[:, i].mean()), "approach_mps_1s": float(v_app),
               "time_to_enter_s": float(d[-1, i] / v_app) if (v_app > 1e-3 and d[-1, i] > 0) else
               (0.0 if d[-1, i] <= 0 else float("inf")),
               "unom_rad_med": float(np.median(un_r[:, i])), "u_rad_med": float(np.median(u_r[:, i]))}
        parked_band = ((d[:, i] > 0) & (d[:, i] <= 0.60)).mean() >= FRAC
        if cross >= 2:
            c = "oscillating"
        elif dd < -0.30:
            c = "in_transit"
        elif parked_band and abs(dd) <= 0.30:
            c = "parked_CBF_blocked" if (rec["unom_rad_med"] < -0.05 and rec["u_rad_med"] > -0.05) else \
                "parked_MFG_stalled" if rec["unom_rad_med"] >= -0.05 else "parked_other"
        else:
            c = "other"
        rec["class"] = c
        out.append(rec)
    return out


def a1():
    out = {}
    for name, f in A1_REPORTS.items():
        r = json.load(open(os.path.join(ROOT, f)))
        out[name] = {"registered_safe_N": r.get("safe_N"), "cells": {}}
        for N, arms in r["rows"].items():
            for a, v in arms.items():
                if v.get("ever_collided") and v.get("safe_success"):
                    out[name]["cells"][f"N{N}/{a}"] = {
                        "safe_success_T": v["safe_success"]["mean"], "completeness_T": v["completeness"]["mean"],
                        "safe_settled_upper_bound": 1 - v["ever_collided"]["mean"]}
    return out


def main():
    cfg = yaml.safe_load(open(a33.BASE))
    sc = build_scenario(cfg, cfg["N"])
    R = {a: arm_rows(d, sc) for a, d in ARMS.items()}
    keys = ["completeness_T", "safe_success_T", "ever_collided", "collision_pair_steps"] + \
        [f"{k}_{W:g}s" for W in WINDOWS for k in ("completeness_settled", "safe_settled")]
    out = {"label": "REGISTERED (amend38, log-only)",
           "aneks_sha256": hashlib.sha256(open(ANEKS, "rb").read()).hexdigest(),
           "A1_bound": a1(), "A2": {}, "A2_safety_identity": {}, "A2_contrasts": {}, "B": {}}
    for a, rows in R.items():
        out["A2"][a] = {"n_seeds": len(rows), **{k: a33.boot_ci([r[k] for r in rows]) for k in keys},
                        "per_seed_safe_settled_5s": [r["safe_settled_5s"] for r in rows]}
        out["A2_safety_identity"][a] = all(r["collision_pair_steps"] == r["registered_pair_steps"] and
                                           abs(r["completeness_T"] - r["registered_completeness"]) < 1e-9 for r in rows)
    for x, y in CONTRASTS:
        out["A2_contrasts"][f"{x} - {y}"] = {k: a33.paired(R[x], R[y], k)
                                              for k in ("safe_success_T", "safe_settled_5s", "safe_settled_3s",
                                                        "safe_settled_10s", "completeness_T", "completeness_settled_5s")}
    for arm in ("fresh", "noise_only"):
        recs = []
        for s in SEEDS:
            for r in classify(load(ARMS[arm], s), sc):
                r["seed"] = s
                recs.append(r)
        cls = {}
        for r in recs:
            cls[r["class"]] = cls.get(r["class"], 0) + 1
        tr = [r for r in recs if r["class"] == "in_transit"]
        n = len(recs)
        tr10 = sum(r["time_to_enter_s"] <= 10 for r in tr)
        parked = sum(v for k, v in cls.items() if k.startswith("parked"))
        reading = "horizon-limited" if n and tr10 / n >= 0.5 else "structural" if n and parked / n >= 0.5 else "mixed"
        out["B"][arm] = {"n_non_settled": n, "per_seed": [sum(r["seed"] == s for r in recs) for s in SEEDS],
                         "classes": cls, "in_transit_enter_le_10s": tr10, "reading": reading,
                         "d_T_quantiles_[0,.25,.5,.75,1]": np.quantile([r["d_T"] for r in recs],
                                                                      [0, .25, .5, .75, 1]).tolist() if n else None,
                         "agents": recs}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1, default=float)

    print("== A2 exact, N=100 stage-1 config (means over seeds)")
    print(f"{'arm':48s} {'n':>2s} {'compl_T':>7s} {'complSet5':>9s} {'safe_T':>7s} {'safeSet3':>8s} {'safeSet5':>8s} "
          f"{'safeSet10':>9s} {'ever':>6s} {'coll_ps':>8s} ident")
    for a, v in out["A2"].items():
        m = lambda k: v[k]["mean"] if v[k] else float("nan")
        print(f"{a:48s} {v['n_seeds']:2d} {m('completeness_T'):7.3f} {m('completeness_settled_5s'):9.3f} "
              f"{m('safe_success_T'):7.3f} {m('safe_settled_3s'):8.3f} {m('safe_settled_5s'):8.3f} "
              f"{m('safe_settled_10s'):9.3f} {m('ever_collided'):6.3f} {m('collision_pair_steps'):8.0f} "
              f"{out['A2_safety_identity'][a]}")
    print("== A2 contrasts (paired, CI95)")
    for c, v in out["A2_contrasts"].items():
        f = lambda k: f"{v[k]['mean']:+.3f} [{v[k]['ci95'][0]:+.3f},{v[k]['ci95'][1]:+.3f}]" if v[k] else "n/a"
        print(f"  {c:60s} safe_T {f('safe_success_T')}  safeSet5 {f('safe_settled_5s')}  "
              f"set3 {f('safe_settled_3s')}  set10 {f('safe_settled_10s')}")
    print("== A1 bounds (safe_settled <= 1 - ever)")
    for name, b in out["A1_bound"].items():
        print(f"  {name} registered safe_N {b['registered_safe_N']}")
        for c, v in b["cells"].items():
            print(f"    {c:40s} safe_T {v['safe_success_T']:.3f} compl_T {v['completeness_T']:.3f} "
                  f"bound {v['safe_settled_upper_bound']:.3f}")
    print("== B non-settlers (amend33, terminal 0)")
    for arm, b in out["B"].items():
        print(f"  {arm}: n={b['n_non_settled']} per_seed={b['per_seed']} classes={b['classes']} "
              f"transit<=10s={b['in_transit_enter_le_10s']} dT_q={b['d_T_quantiles_[0,.25,.5,.75,1]']} "
              f"-> {b['reading']}")
    print("aneks sha256:", out["aneks_sha256"])


if __name__ == "__main__":
    assert sys.argv[1:] == ["report"], __doc__
    main()
