"""Amendment 39: re-run the Config G headline cells under the settled metric + 0.30 m goal band, archive raw trajectories.
Prereg: prereg/prereg_projectA_amend39_settled_regen.md (frozen before any run).

    python -m experiments.amend39_settled_regen cells            # copy registered cells (only run_name changes) + MANIFEST
    python -m experiments.amend39_settled_regen arm  <cell.yaml>  # print the ladder --arm for a cell
    python -m experiments.amend39_settled_regen settle [--stage A]  # per rollout: settled_seed*.json + traj_seed*.npz
    python -m experiments.amend39_settled_regen report           # report.json + figure_data_amend39.csv
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

from experiments.amend12_report import agent_facts
from experiments.smoke import ROOT, build_scenario

RUN = "amend39_settled_regen"
RES = os.path.join(ROOT, "results", RUN)
CELLS = os.path.join(ROOT, "experiments", "amend39_cells")
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend39_settled_regen.md")
DT, FRAC = 0.01, 0.90
WINDOWS, BANDS = (3.0, 5.0, 10.0), (0.0, 0.30)
W0, B0 = 5.0, 0.0
SEEDS = list(range(10))
N_BOOT = 10000
U_TAIL_S = 10.0
IDENTITY_KEYS = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
                 "cbf_intervention_rate", "slack_active_rate")
# stage -> [(source cell dir, registered results dir, cell-name glob)]
SOURCES = {
    "A": [("amend22_cells", "amend22_N_crossover_pooled", "*.yaml"),
          ("amend24_cells", "amend24_mechanism_control", "*.yaml")],
    "B": [("amend26_cells", "amend26_tslot_sweep", "*.yaml")],
    "C": [("amend23_cells", "amend23_margin_pooled", "*.yaml"),
          ("amend25_cells", "amend25_gate_sweep", "*.yaml")],
    "D": [("amend27_cells", "amend27_ladder_layers", "N*_split_*-*_d0.70.yaml")],
}


def key(stage, src, name):
    return f"{stage}__{src.replace('_cells', '')}__{name}"


def arm_of(cfg, name):
    if cfg.get("amend27_arm"):
        return cfg["amend27_arm"]
    for tag, arm in (("learned", "learned"), ("noise_only", "noise_only"), ("fresh", "fresh"), ("real", "real")):
        if tag in name:
            return arm
    raise ValueError(f"no arm for {name}")


def arm_dir(arm):
    return arm.replace(":", "_").replace("/", "-")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- cells
def cmd_cells(_):
    os.makedirs(CELLS, exist_ok=True)
    man = {"aneks_sha256": sha(ANEKS), "cells": {}}
    for stage, srcs in SOURCES.items():
        for src, reg, pat in srcs:
            for p in sorted(glob.glob(os.path.join(ROOT, "experiments", src, pat))):
                name = os.path.basename(p)[:-5]
                cfg = yaml.safe_load(open(p))
                new = copy.deepcopy(cfg)
                new["run_name"] = f"{RUN}/{stage}/{src.replace('_cells', '')}/{name}"
                diff = {k for k in set(cfg) | set(new) if cfg.get(k) != new.get(k)}
                assert diff == {"run_name"}, (p, diff)
                out = os.path.join(CELLS, key(stage, src, name) + ".yaml")
                header = f"# AMEND39 copy of experiments/{src}/{name}.yaml (only run_name changed)\n"
                open(out, "w").write(header + yaml.safe_dump(new, sort_keys=False))
                man["cells"][key(stage, src, name)] = {
                    "stage": stage, "source": f"experiments/{src}/{name}.yaml", "source_sha256": sha(p),
                    "copy_sha256": sha(out), "arm": arm_of(cfg, name), "run_name": new["run_name"],
                    "registered_dir": f"results/{cfg['run_name']}/{arm_dir(arm_of(cfg, name))}"}
    json.dump(man, open(os.path.join(CELLS, "MANIFEST.json"), "w"), indent=1)
    n = {s: sum(1 for c in man["cells"].values() if c["stage"] == s) for s in SOURCES}
    print(f"[cells] {len(man['cells'])} cells {n}")


def cmd_arm(a):
    cfg = yaml.safe_load(open(a.cell))
    print(arm_of(cfg, os.path.basename(a.cell)))


# ------------------------------------------------------------------------------------------------------------ settle
def settled_metrics(z, sc):
    """Per-rollout metrics. b=0 reproduces experiments.amend38_settled_recompute.arm_rows exactly."""
    pos = np.asarray(z["pos"], np.float64)
    pT = np.asarray(z["pos_final"], np.float64)
    _, ps, arrT, coll = agent_facts({"pos": pos, "pos_final": pT}, sc)
    P = np.concatenate([pos, pT[None]])
    dist = np.linalg.norm(P - np.asarray(sc.goal), axis=-1)
    r = {"completeness_T": float(arrT.mean()), "ever_collided": float(coll.mean()),
         "safe_success_T": float((arrT & ~coll).mean()), "collision_pair_steps": int(ps)}
    for b in BANDS:
        inside = dist <= sc.r_goal + b
        for W in WINDOWS:
            st = inside[-(int(round(W / DT)) + 1):].mean(0) >= FRAC
            r[f"completeness_settled_{W:g}s_b{b:.2f}"] = float(st.mean())
            r[f"safe_settled_{W:g}s_b{b:.2f}"] = float((st & ~coll).mean())
    return r, coll


def cmd_settle(a):
    man = json.load(open(os.path.join(CELLS, "MANIFEST.json")))
    done = flagged = 0
    for k, c in man["cells"].items():
        if a.stage and c["stage"] != a.stage:
            continue
        cfg = yaml.safe_load(open(os.path.join(CELLS, k + ".yaml")))
        d = os.path.join(ROOT, "results", c["run_name"], arm_dir(c["arm"]))
        sc = None
        for s in SEEDS:
            zf, mf = os.path.join(d, f"logs_seed{s}.npz"), os.path.join(d, f"metrics_seed{s}.json")
            out = os.path.join(d, f"settled_seed{s}.json")
            if not (os.path.exists(zf) and os.path.exists(mf)) or os.path.exists(out):
                continue
            sc = sc or build_scenario(cfg, cfg["N"])
            with np.load(zf) as z:
                r, coll = settled_metrics(z, sc)
                n_tail = int(round(U_TAIL_S / DT))
                tail = {k2: np.asarray(z[k2][-n_tail:], np.float32) for k2 in ("u", "u_nom") if k2 in z.files}
                np.savez_compressed(os.path.join(d, f"traj_seed{s}.npz"), pos=np.asarray(z["pos"], np.float32),
                                    pos_final=np.asarray(z["pos_final"], np.float32), collided=coll,
                                    goal=np.asarray(sc.goal, np.float32), r_goal=np.float32(sc.r_goal), **tail)
            m = json.load(open(mf))
            r["check_vs_ladder"] = (r["collision_pair_steps"] == m["collision_pair_steps"] and
                                    abs(r["completeness_T"] - m["completeness_final"]) < 1e-9)
            flagged += not r["check_vs_ladder"]
            r.update(seed=s, cell=k, stage=c["stage"], arm=c["arm"], matmul_precision=m.get("matmul_precision"),
                     **{f"ladder_{x}": m.get(x) for x in IDENTITY_KEYS})
            json.dump(r, open(out, "w"), indent=1)
            done += 1
    print(f"[settle] {done} new settled files, {flagged} flagged (recomputed != ladder metrics)")


# ------------------------------------------------------------------------------------------------------------ report
def boot_ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return None
    b = x[np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n": int(len(x))}


def paired(ra, rb, k):
    da, db = {r["seed"]: r[k] for r in ra}, {r["seed"]: r[k] for r in rb}
    return boot_ci([da[s] - db[s] for s in sorted(set(da) & set(db))])


def rows_of(c):
    d = os.path.join(ROOT, "results", c["run_name"], arm_dir(c["arm"]))
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "settled_seed*.json")))]


def registered_of(c):
    out = []
    for s in SEEDS:
        p = os.path.join(ROOT, c["registered_dir"], f"metrics_seed{s}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def metric_keys():
    ks = ["completeness_T", "safe_success_T", "ever_collided"]
    for b in BANDS:
        for W in WINDOWS:
            ks += [f"completeness_settled_{W:g}s_b{b:.2f}", f"safe_settled_{W:g}s_b{b:.2f}"]
    return ks


def safe_n(cells_by_n, b):
    """amend22 definition, completeness -> completeness_settled_5s_b (prereg Q1)."""
    best = None
    for N in sorted(cells_by_n):
        s = cells_by_n[N]
        ev, cp = s.get("ever_collided"), s.get(f"completeness_settled_{W0:g}s_b{b:.2f}")
        if ev is None or cp is None or not (ev["ci95"][1] <= 0.10 and cp["mean"] >= 0.90):
            break
        best = N
    return best


def cmd_report(_):
    man = json.load(open(os.path.join(CELLS, "MANIFEST.json")))
    ks = metric_keys()
    R, S, out = {}, {}, {"label": "REGISTERED (amend39)", "aneks_sha256": sha(ANEKS), "cells": {}, "missing": {},
                         "reproduction": {}, "flagged_rollouts": [], "Q": {}}
    for k, c in man["cells"].items():
        R[k] = rows_of(c)
        if len(R[k]) < len(SEEDS):
            out["missing"][k] = len(SEEDS) - len(R[k])
        S[k] = {m: boot_ci([r[m] for r in R[k]]) for m in ks} if R[k] else {}
        out["cells"][k] = {"stage": c["stage"], "arm": c["arm"], "n_seeds": len(R[k]), **S[k]}
        out["flagged_rollouts"] += [f"{k}/seed{r['seed']}" for r in R[k] if not r.get("check_vs_ladder", True)]
        reg = registered_of(c)
        if R[k] and reg:
            by = {m["seed"]: m for m in reg}
            same = [all(r[f"ladder_{x}"] == by[r["seed"]].get(x) for x in IDENTITY_KEYS) for r in R[k] if r["seed"] in by]
            out["reproduction"][k] = {
                "bit_equal_seeds": int(sum(same)), "compared_seeds": len(same),
                "registered_completeness_T": boot_ci([m["completeness_final"] for m in reg]),
                "regen_completeness_T": S[k].get("completeness_T")}

    def cell(stage, src, name):
        return key(stage, src + "_cells", name)

    def stat(stage, src, name):
        return S.get(cell(stage, src, name)) or {}

    Ns = [25, 50, 100, 150, 200]
    arms = {"real_d0.55": ("amend22", "real_d0.55"), "real_d0.70": ("amend22", "real_d0.70"),
            "learned_d0.70": ("amend22", "learned_pooled_d0.70"), "noise_only_d0.70": ("amend24", "noise_only_d0.70")}
    q = out["Q"]
    q["Q1_safe_N"] = {f"b{b:.2f}": {a: safe_n({N: stat("A", src, f"N{N:03d}_{suf}") for N in Ns
                                                if stat("A", src, f"N{N:03d}_{suf}")}, b)
                                    for a, (src, suf) in arms.items()} for b in BANDS}
    q["Q2_learned_minus_real0.70"] = {
        N: {f"b{b:.2f}": paired(R.get(cell("A", "amend22", f"N{N:03d}_learned_pooled_d0.70"), []),
                                R.get(cell("A", "amend22", f"N{N:03d}_real_d0.70"), []), f"safe_settled_{W0:g}s_b{b:.2f}")
            for b in BANDS} for N in Ns}
    q["Q3_noise_only"] = {N: {m: stat("A", "amend24", f"N{N:03d}_noise_only_d0.70").get(m)
                              for m in ("ever_collided", f"safe_settled_{W0:g}s_b0.00", f"safe_settled_{W0:g}s_b0.30")}
                          for N in Ns}
    ts = {"0.25": "ts2.50ms", "0.5": "ts5.00ms", "0.75": "ts7.50ms", "1.0": "ts10.00ms"}
    q["Q4_staleness"] = {
        tf: {"learned_minus_real0.70": {f"b{b:.2f}": paired(R.get(cell("B", "amend26", f"{t}_learned_pooled_d0.70"), []),
                                                             R.get(cell("B", "amend26", f"{t}_real_d0.70"), []),
                                                             f"safe_settled_{W0:g}s_b{b:.2f}") for b in BANDS},
             "learned_fixedN100": {m: stat("B", "amend26", f"{t}_learned_pooled_d0.70").get(m)
                                   for m in ("ever_collided", f"completeness_settled_{W0:g}s_b0.00")},
             "learned_crossover_equal_Tframe": {m: stat("A", "amend22", f"N{n:03d}_learned_pooled_d0.70").get(m)
                                                for m in ("ever_collided", f"completeness_settled_{W0:g}s_b0.00")}}
        for (tf, t), n in zip(ts.items(), (50, 100, 150, 200))}
    q["Q5_margin"] = {}
    for N in (100, 150):
        cand = {}
        for d in ("0.55", "0.70", "0.85", "1.00"):
            s = stat("C", "amend23", f"N{N}_learned_pooled_d{d}")
            if s:
                cand[d] = (s[f"completeness_settled_{W0:g}s_b0.00"]["mean"], s[f"safe_settled_{W0:g}s_b0.00"]["mean"])
        elig = {d: v for d, v in cand.items() if v[0] >= 0.90}
        q["Q5_margin"][N] = {"candidates_[compl_settled,safe_settled]": cand,
                             "selected": max(elig, key=lambda d: elig[d][1]) if elig else None}
    q["Q6_gate"] = {g: paired(R.get(cell("C", "amend25", f"g{g}_learned_pooled_d0.70"), []),
                              R.get(cell("C", "amend25", f"g{g}_real_d0.70"), []), f"safe_settled_{W0:g}s_b0.00")
                    for g in ("1.500", "2.309", "3.500", "5.000", "7.000")}
    q["Q7_layers"] = {N: {a: stat("D", "amend27", f"N{N}_split_{a}_d0.70").get(f"safe_settled_{W0:g}s_b0.00")
                          for a in ("learned-real", "real-learned", "noise_only-real", "real-noise_only")}
                      for N in (100, 150, 200)}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1, default=str)
    with open(os.path.join(RES, "figure_data_amend39.csv"), "w") as fh:
        fh.write("cell,stage,arm,metric,mean,ci_lo,ci_hi,n\n")
        for k, s in S.items():
            for m, v in s.items():
                if v:
                    fh.write(f"{k},{man['cells'][k]['stage']},{man['cells'][k]['arm']},{m},{v['mean']},"
                             f"{v['ci95'][0]},{v['ci95'][1]},{v['n']}\n")
    print(json.dumps({"missing": out["missing"], "flagged": len(out["flagged_rollouts"]), "Q1": q["Q1_safe_N"]},
                     indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("cells").set_defaults(f=cmd_cells)
    p = sp.add_parser("arm"); p.add_argument("cell"); p.set_defaults(f=cmd_arm)
    p = sp.add_parser("settle"); p.add_argument("--stage", default=None); p.set_defaults(f=cmd_settle)
    sp.add_parser("report").set_defaults(f=cmd_report)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
