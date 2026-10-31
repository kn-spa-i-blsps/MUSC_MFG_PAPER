"""Amendment-29 (physical limits) and amendment-30 (generic layouts): the compensation-method ladder under perturbed
physics / geometry, amend21 arena, precision=highest, seeds 0-9. One module, two experiments (--exp amend29|amend30).

Arms: real, cv, smith (each at its better d_safe from the amend27 report at the same N, by mean safe-success; tie ->
0.70), learned v3 d0.70, noise_only d0.70 (delay-free reference); amend30 also fresh d0.70 and fresh_nomfg d0.70
(decoupled routing, N=100 only).

amend29 settings (one knob at a time from the default; N in {100, 150}):
    sigma_scale 0.5 / 2 / 4      (UWB range + bearing noise x scale)
    p_loss      0.1 / 0.2 / 0.4  (TDMA packet loss)
    nu          0.01 / 0.05      (process noise, m^2/s)
amend30 layouts (N in {100, 150}): two_gates, offset_gate, pillar (side 4 m); one solve per (layout, N), cache deleted
after that layout's rollouts.

    python -m experiments.amend29_physical_limits --exp amend29 write-cells
    python -m experiments.amend29_physical_limits --exp amend29 report
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import run_rows, sha256
from experiments.smoke import ROOT, build_scenario

SRC = os.path.join(ROOT, "experiments", "amend21_cells")
A27 = os.path.join(ROOT, "results", "amend27_ladder_layers", "report.json")
SEEDS = list(range(10))
NS = (100, 150)
SETTINGS = {
    "amend29": {**{f"sigma{s:g}": {"sigma_scale": s} for s in (0.5, 2.0, 4.0)},
                **{f"loss{p:g}": {"p_loss": p} for p in (0.1, 0.2, 0.4)},
                **{f"nu{v:g}": {"nu": v} for v in (0.01, 0.05)}},
    "amend30": {L: {"scenario": {"layout": L, **({"pillar_side": 4.0} if L == "pillar" else {})}}
                for L in ("two_gates", "offset_gate", "pillar")},
}
RUN = {"amend29": "amend29_physical_limits", "amend30": "amend30_layouts"}
ANEKS = {"amend29": "prereg_projectA_amend29_physical_limits.md", "amend30": "prereg_projectA_amend30_layouts.md"}
COND = {"real": "real", "cv": "cv", "smith": "smith", "learned": "learned", "noise_only": "noise_only",
        "fresh": "fresh", "fresh_nomfg": "fresh"}


def best_margins():
    """amend27 rule (pre-registered): per N and method, the d_safe with the higher mean safe-success (tie -> 0.70)."""
    a = json.load(open(A27))
    out = {}
    for N in NS:
        r = a["rows"][str(N)]
        for m in ("real", "cv", "smith"):
            s55, s70 = r[f"{m}_d0.55"]["safe_success"]["mean"], r[f"{m}_d0.70"]["safe_success"]["mean"]
            out[(N, m)] = 0.55 if s55 > s70 else 0.70
    return out


def arms(exp, N):
    a = ["real", "cv", "smith", "learned", "noise_only"]
    if exp == "amend30":
        a += ["fresh"] + (["fresh_nomfg"] if N == 100 else [])
    return a


def merge(c, d):
    for k, v in d.items():
        if isinstance(v, dict):
            merge(c.setdefault(k, {}), v)
        else:
            c[k] = v
    return c


def cell_cfg(exp, setting, N, arm, margins):
    c = copy.deepcopy(yaml.safe_load(open(os.path.join(SRC, f"N{N:03d}_real.yaml"))))
    merge(c, copy.deepcopy(SETTINGS[exp][setting]))
    d = margins.get((N, arm), 0.70)
    c["cbf"]["d_safe"] = d
    if arm == "fresh_nomfg":
        c["macro_cost"]["c_w"] = 0.0
        c["macro_cost"]["w_bar"] = 0.0
        c["control"]["c_dens"] = 0.0
    c["run_name"] = f"{RUN[exp]}/{setting}/N{N:03d}/{arm}_d{d:.2f}"
    c["chain_arm"] = COND[arm]
    return c


def cell_dir(exp):
    return os.path.join(ROOT, "experiments", f"{exp}_cells")


def cmd_write_cells(a):
    margins = best_margins()
    for setting in SETTINGS[a.exp]:
        os.makedirs(os.path.join(cell_dir(a.exp), setting), exist_ok=True)
        for N in NS:
            for arm in arms(a.exp, N):
                c = cell_cfg(a.exp, setting, N, arm, margins)
                p = os.path.join(cell_dir(a.exp), setting, f"N{N:03d}_{arm}.yaml")
                with open(p, "w") as f:
                    f.write(f"# {a.exp.upper()} cell {setting} N{N:03d} {arm}: amend21_cells/N{N:03d}_real.yaml + "
                            f"{json.dumps(SETTINGS[a.exp][setting])}, d_safe {c['cbf']['d_safe']} (amend27 rule), "
                            f"run_name, chain_arm\n")
                    yaml.safe_dump(c, f, sort_keys=False)
                print(p)
    json.dump({f"N{N}_{m}": d for (N, m), d in margins.items()},
              open(os.path.join(cell_dir(a.exp), "margins_from_amend27.json"), "w"), indent=1)


def cmd_report(a):
    out = {"aneks_sha256": sha256(os.path.join(ROOT, "prereg", ANEKS[a.exp])), "matmul_precision": "highest",
           "seeds": a.seeds, "ci": "seed-level (cluster) bootstrap, 10k, 95%",
           "margins": json.load(open(os.path.join(cell_dir(a.exp), "margins_from_amend27.json"))), "rows": {},
           "diffs": {}, "recovery": {}}
    ps = lambda r, k: r[k]["per_seed"]
    for setting in SETTINGS[a.exp]:
        for N in NS:
            rows = {}
            for arm in arms(a.exp, N):
                c = yaml.safe_load(open(os.path.join(cell_dir(a.exp), setting, f"N{N:03d}_{arm}.yaml")))
                sc = build_scenario(c, N)
                rows[arm] = run_rows(os.path.join(ROOT, "results", c["run_name"], COND[arm]), sc, a.seeds, 0.30)
            key = f"{setting}/N{N}"
            out["rows"][key] = rows
            out["diffs"][key] = {f"learned_minus_{m}": {k: diff_ci(ps(rows["learned"], k), ps(rows[m], k))
                                                        for k in ("safe_success", "ever_collided", "completeness")}
                                 for m in ("real", "cv", "smith", "noise_only")}
            gap = rows["noise_only"]["safe_success"]["mean"] - rows["real"]["safe_success"]["mean"]
            out["recovery"][key] = {m: ((rows[m]["safe_success"]["mean"] - rows["real"]["safe_success"]["mean"]) / gap
                                        if abs(gap) > 1e-9 else None) for m in ("cv", "smith", "learned")}
    res = os.path.join(ROOT, "results", RUN[a.exp])
    os.makedirs(res, exist_ok=True)
    json.dump(out, open(os.path.join(res, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for key, rows in out["rows"].items():
        for arm, r in rows.items():
            print(f"{key:22s} {arm:12s} safe={fmt(r['safe_success'])} ever={fmt(r['ever_collided'])} "
                  f"compl={fmt(r['completeness'])}")
        print(f"{key:22s} recovered fraction of (noise_only - real) safe-success gap: {out['recovery'][key]}")
        for m, v in out["diffs"][key].items():
            s = v["safe_success"]
            print(f"{key:22s} {m}: safe {s['diff']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    print("aneks sha256:", out["aneks_sha256"])


def cmd_solve(a):
    """Phi/m* for one cell yaml into results/cache (amend30 layouts; no rollout may run next to it)."""
    import time
    from experiments.ladder import setup
    t0 = time.time()
    _, grid, sol, _, path = setup(yaml.safe_load(open(a.cell)))
    print(json.dumps({"cell": a.cell, "cache": os.path.relpath(path, ROOT), "wall_s": time.time() - t0,
                      "horizon_s": grid.horizon, "cesaro_converged": bool(sol["cesaro_converged"]),
                      "iters": int(sol["iters"]), "nan": bool(sol["nan"])}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(SETTINGS), required=True)
    ap.add_argument("cmd", choices=["write-cells", "solve", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--cell")
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "solve": cmd_solve, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
