"""Amendment-37: settled-window success metric, LOG-ONLY. Terminal cost ON (amend36) vs OFF (amend33),
fresh / noise_only / fresh_nomfg, seeds 0-4, N=100 amend11 config.

    python -m experiments.amend37_settled_window report
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import yaml

from experiments import amend33_n100_baseline_cpu as a33
from experiments.amend12_report import agent_facts
from experiments.smoke import ROOT, build_scenario

RUN = "amend37_settled_window"
RES = os.path.join(ROOT, "results", RUN)
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend37_settled_window.md")
SEEDS = a33.SEEDS
DT = 0.01
FRAC = 0.90
WINDOWS = (3.0, 5.0, 10.0)
PRIMARY = 5.0
ARMS = ("fresh", "noise_only", "fresh_nomfg")
DIRS = {
    "ON": {"fresh": "results/amend36_terminal_goal/fresh/fresh",
           "noise_only": "results/amend36_terminal_goal/noise_only/noise_only",
           "fresh_nomfg": "results/amend36_terminal_goal/fresh_nomfg/fresh"},
    "OFF": {"fresh": "results/amend33_n100_baseline_cpu/fresh/fresh",
            "noise_only": "results/amend33_n100_baseline_cpu/noise_only/noise_only",
            "fresh_nomfg": "results/amend33_n100_baseline_cpu/fresh_nomfg/fresh"},
}


def rows(d, sc):
    g = np.asarray(sc.goal)
    out = []
    for s in SEEDS:
        z = np.load(os.path.join(ROOT, d, f"logs_seed{s}.npz"))
        pos, pT = z["pos"].astype(np.float64), z["pos_final"].astype(np.float64)
        _, pair_steps, arrived_T, collided = agent_facts({"pos": pos, "pos_final": pT}, sc)
        traj = np.concatenate([pos, pT[None]], 0)                      # (K+1, N, 2), last = state at T
        inside = np.linalg.norm(traj - g, axis=-1) <= sc.r_goal          # (K+1, N)
        r = {"seed": s, "completeness_T": float(arrived_T.mean()), "collision_pair_steps": int(pair_steps),
             "ever_collided": float(collided.mean())}
        for W in WINDOWS:
            n = int(round(W / DT)) + 1                                   # samples with t >= T - W
            settled = inside[-n:].mean(0) >= FRAC
            r[f"completeness_settled_{W:g}s"] = float(settled.mean())
            r[f"safe_settled_{W:g}s"] = float((settled & ~collided).mean())
            r[f"n_not_settled_{W:g}s"] = int((~settled).sum())
        out.append(r)
    return out


def link1(R, W):
    c = [r[f"safe_settled_{W:g}s"] for r in R]
    ok = (len(R) == len(SEEDS) and all(r["collision_pair_steps"] == 0 for r in R)
          and np.mean(c) >= 0.99 and min(c) >= 0.98)
    return "HOLDS" if ok else "NOT MET"


def main():
    cfg = yaml.safe_load(open(a33.BASE))
    sc = build_scenario(cfg, cfg["N"])
    R = {c: {a: rows(DIRS[c][a], sc) for a in ARMS} for c in DIRS}
    keys = ["completeness_T", "collision_pair_steps", "ever_collided"] + [
        f"{k}_{W:g}s" for W in WINDOWS for k in ("completeness_settled", "safe_settled", "n_not_settled")]
    out = {"label": "REGISTERED (amend37, log-only)",
           "aneks_sha256": hashlib.sha256(open(ANEKS, "rb").read()).hexdigest(),
           "seeds": SEEDS, "frac": FRAC, "windows_s": WINDOWS, "primary_s": PRIMARY,
           "arms": {c: {a: {"per_seed": {k: [r[k] for r in R[c][a]] for k in keys},
                            **{k: a33.boot_ci([r[k] for r in R[c][a]]) for k in keys}} for a in ARMS} for c in R},
           "link1": {}, "on_minus_off": {}, "decision": {}, "mfg_minus_cbfonly": {}}
    for W in WINDOWS:
        key = f"safe_settled_{W:g}s"
        w = f"{W:g}s"
        out["link1"][w] = {c: {a: link1(R[c][a], W) for a in ("fresh", "noise_only")} for c in R}
        out["on_minus_off"][w] = {a: a33.paired(R["ON"][a], R["OFF"][a], key) for a in ARMS}
        on_wins = all(out["on_minus_off"][w][a]["ci95"][0] > 0 for a in ("fresh", "noise_only"))
        out["decision"][w] = "ON > OFF: keep terminal cost, re-baseline" if on_wins else \
            "OFF >= ON: revert terminal cost, no re-baseline"
        out["mfg_minus_cbfonly"][w] = {c: a33.paired(R[c]["fresh"], R[c]["fresh_nomfg"], key) for c in R}
    same = lambda d: len({json.dumps(d[f"{W:g}s"], sort_keys=True) for W in WINDOWS}) == 1
    robust = same(out["link1"]) and same(out["decision"])
    out["window_robust"] = robust
    out["final"] = out["decision"][f"{PRIMARY:g}s"] if robust else \
        "WINDOW-DEPENDENT: 5 s verdict stated, decision goes to the PI"
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)

    m = lambda c, a, k: out["arms"][c][a][k]["mean"]
    print(f"{'ctrl':4s} {'arm':12s} {'compl_T':>7s} " + " ".join(f"{'safeSet' + f'{W:g}':>9s}" for W in WINDOWS)
          + f" {'coll_ps':>7s}  per-seed safe_settled_5s")
    for c in R:
        for a in ARMS:
            print(f"{c:4s} {a:12s} {m(c, a, 'completeness_T'):7.3f} "
                  + " ".join(f"{m(c, a, f'safe_settled_{W:g}s'):9.3f}" for W in WINDOWS)
                  + f" {m(c, a, 'collision_pair_steps'):7.0f}  {out['arms'][c][a]['per_seed']['safe_settled_5s']}")
    for W in WINDOWS:
        w = f"{W:g}s"
        print(f"--- W={w}  link1 {out['link1'][w]}")
        for a, v in out["on_minus_off"][w].items():
            print(f"  ON-OFF {a:12s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        for c, v in out["mfg_minus_cbfonly"][w].items():
            print(f"  MFG-CBFonly {c:3s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        print(f"  decision: {out['decision'][w]}")
    print("window_robust:", robust, "| FINAL:", out["final"])
    print("aneks sha256:", out["aneks_sha256"])


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["report"], __doc__
    main()
