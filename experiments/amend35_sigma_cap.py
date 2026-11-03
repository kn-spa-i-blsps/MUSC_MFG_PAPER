"""Amendment-35: capped sigma-CBF G-win, the last attempt for Phase B. ONE change vs amend31: d_safe_ij and sigma_c are
capped at d_max = 1.5 m (sigma_c <= (1.5 - 0.30)/k). Frozen amend31 sigma head + conformal calibration, learned v3,
precision=highest, N 150 / 200, seeds 0-9. Comparator = the registered amend31 fixed rows (code path untouched).

    python -m experiments.amend35_sigma_cap write-cells
    python -m experiments.amend35_sigma_cap sanity          # after the N150 sigma_bin_a0.05 no-cap seed-0 rollout
    python -m experiments.amend35_sigma_cap report
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from experiments.amend20_closing_lockin import diff_ci
from experiments.amend22_N_crossover_pooled import IDENTITY_KEYS, run_rows, sha256
from experiments.amend31_sigma_cbf import ALPHAS, FIXED, SIG, SIG_EXTRA, cell_cfg
from experiments.smoke import ROOT, build_scenario

RUN = "amend35_sigma_cap"
RES = os.path.join(ROOT, "results", RUN)
CELL_DIR = os.path.join(ROOT, "experiments", "amend35_cells")
A31 = os.path.join(ROOT, "results", "amend31_sigma_cbf")
ANEKS = os.path.join(ROOT, "prereg", "prereg_projectA_amend35_sigma_cap.md")
D_MAX = 1.5
NS = (150, 200)
SEEDS = list(range(10))
PRIMARY = "sigma_bin_a0.05"          # declared in the aneks before any data
NI_MARGIN = 0.02                     # same operationalisation of "completeness not worse" as amend31 addendum 1
SANITY_KEYS = IDENTITY_KEYS + ("d_safe_mean", "d_safe_p95", "sigma_cov_rows", "sigma_cov_rows_near2m",
                               "sigma_mean_rows_m", "sigma_bind_frac")


def sigma_arms():
    return [f"sigma_a{al:g}" for al in ALPHAS] + [PRIMARY]


def fixed_arms():
    return [f"fixed{d:.2f}" for d in FIXED]


def cmd_write_cells(a):
    cal = json.load(open(os.path.join(SIG, "calibration.json")))
    os.makedirs(os.path.join(CELL_DIR, "sanity"), exist_ok=True)
    for N in NS:
        for arm in sigma_arms():
            c = cell_cfg(N, arm, cal)
            c["cbf"]["margin_cap"] = D_MAX
            c["run_name"] = f"{RUN}/N{N:03d}/{arm}"
            with open(os.path.join(CELL_DIR, f"N{N:03d}_{arm}.yaml"), "w") as f:
                f.write(f"# AMEND35 cell N{N:03d} {arm}: amend31 cell (cell_cfg, same calibration) + cbf.margin_cap {D_MAX}, "
                        f"run_name\n")
                yaml.safe_dump(c, f, sort_keys=False)
    os.makedirs(os.path.join(CELL_DIR, "fixed_rerun"), exist_ok=True)
    for N in NS:                             # aneks: off-A100 the fixed arms are re-run on the same GPU
        for arm in fixed_arms():
            c = cell_cfg(N, arm, cal)
            c["run_name"] = f"{RUN}/N{N:03d}/{arm}"
            with open(os.path.join(CELL_DIR, "fixed_rerun", f"N{N:03d}_{arm}.yaml"), "w") as f:
                f.write(f"# AMEND35 fixed re-run cell N{N:03d} {arm}: amend31 cell unchanged except run_name\n")
                yaml.safe_dump(c, f, sort_keys=False)
    c = cell_cfg(150, PRIMARY, cal)          # sanity: NO cap (margin_cap absent) = the amend31 cell, own run_name
    c["run_name"] = f"{RUN}/sanity/N150_{PRIMARY}_nocap"
    with open(os.path.join(CELL_DIR, "sanity", f"N150_{PRIMARY}_nocap.yaml"), "w") as f:
        f.write("# AMEND35 sanity cell: the amend31 N150 sigma_bin_a0.05 cell unchanged except run_name (no cap)\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("cells written to", CELL_DIR)


def _cmp(new_dir, ref_dir, seed=0):
    n = json.load(open(os.path.join(new_dir, f"metrics_seed{seed}.json")))
    r = json.load(open(os.path.join(ref_dir, f"metrics_seed{seed}.json")))
    keys = {k: {"amend31": r.get(k), "new": n.get(k), "equal": r.get(k) == n.get(k)} for k in SANITY_KEYS}
    out = {"keys": keys, "keys_equal": all(v["equal"] for v in keys.values())}
    zn, zr = (os.path.join(d, f"logs_seed{seed}.npz") for d in (new_dir, ref_dir))
    if os.path.exists(zn) and os.path.exists(zr):
        A, B = np.load(zn), np.load(zr)
        out["pos_bit_equal"] = bool(np.array_equal(A["pos"], B["pos"]) and np.array_equal(A["pos_final"], B["pos_final"]))
        if not out["pos_bit_equal"]:
            diff = np.any(A["pos"] != B["pos"], axis=(1, 2))
            out["first_pos_divergence_step"] = int(np.argmax(diff)) if diff.any() else None
    return out


def cmd_sanity(a):
    """S1 (gating, PI): no-cap new code vs amend31 committed N150 sigma_bin_a0.05 seed 0, bit-equal.
    S2 (fallback, only if S1 fails, aneks): old code (amend31 commit, git worktree) vs new code no-cap, both with
    XLA_FLAGS=--xla_gpu_autotune_level=0 on this pod, bit-equal = the cap is the only code change."""
    new = os.path.join(RES, "sanity", f"N150_{PRIMARY}_nocap", "learned")
    out = {"S1_vs_amend31": _cmp(new, os.path.join(A31, "N150", PRIMARY, "learned"))}
    s1 = out["S1_vs_amend31"]["keys_equal"] and out["S1_vs_amend31"].get("pos_bit_equal", True)
    out["S1_pass"] = s1
    if not s1 and a.s2:
        old = os.path.join(RES, "sanity_s2", "old", "learned"); nw = os.path.join(RES, "sanity_s2", "new", "learned")
        out["S2_old_vs_new_autotune0"] = _cmp(nw, old)
        out["S2_pass"] = out["S2_old_vs_new_autotune0"]["keys_equal"] and out["S2_old_vs_new_autotune0"].get("pos_bit_equal", False)
    out["pass"] = s1 or out.get("S2_pass", False)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "sanity.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
    if not s1:
        print("S1 differing keys:", {k: v for k, v in out["S1_vs_amend31"]["keys"].items() if not v["equal"]},
              "first pos divergence:", out["S1_vs_amend31"].get("first_pos_divergence_step"))
    raise SystemExit((0 if s1 else 2) if not a.s2 else (0 if out["pass"] else 1))   # 2 = S1 failed, run S2


def dsafe_max(rdir, seeds):
    """largest non-empty realised d_safe bin (0.01 m) over the seeds, from the logged histogram."""
    m = 0.0
    for s in seeds:
        h = np.load(os.path.join(rdir, f"logs_seed{s}.npz"))["d_safe_hist"].sum(0)
        nz = np.nonzero(h)[0]
        if nz.size:
            m = max(m, 0.30 + 0.01 * nz.max())
    return m


def cmd_report(a):
    out = {"label": "REGISTERED (amend35, one shot)", "aneks_sha256": sha256(ANEKS), "d_max": D_MAX,
           "primary": PRIMARY, "ni_margin": NI_MARGIN, "seeds": a.seeds, "matmul_precision": "highest",
           "ci": "seed-level (cluster) bootstrap, 10k, 95% (amend31 diff_ci)", "rows": {}, "best_fixed": {},
           "gwin": {}, "dsafe_max_realised_m": {}}
    ps = lambda r, k: r[k]["per_seed"]
    for N in NS:
        sc = build_scenario(yaml.safe_load(open(os.path.join(CELL_DIR, f"N{N:03d}_{PRIMARY}.yaml"))), N)
        rer = os.path.exists(os.path.join(RES, f"N{N:03d}", "fixed0.70", "learned", "metrics_seed0.json"))
        out.setdefault("fixed_source", {})[str(N)] = "re-run on this GPU (amend35)" if rer else "amend31 (A100)"
        fsrc = RES if rer else A31
        rows = {arm: run_rows(os.path.join(fsrc, f"N{N:03d}", arm, "learned"), sc, a.seeds, 0.30) for arm in fixed_arms()}
        rows.update({arm: run_rows(os.path.join(RES, f"N{N:03d}", arm, "learned"), sc, a.seeds, 0.30, extra=SIG_EXTRA)
                     for arm in sigma_arms()})
        out["rows"][str(N)] = rows
        best = max(fixed_arms(), key=lambda k: rows[k]["safe_success"]["mean"])
        out["best_fixed"][str(N)] = best
        out["gwin"][str(N)] = {arm: {m: diff_ci(ps(rows[arm], m), ps(rows[best], m))
                                     for m in ("safe_success", "ever_collided", "completeness")} for arm in sigma_arms()}
        out["dsafe_max_realised_m"][str(N)] = {arm: dsafe_max(os.path.join(RES, f"N{N:03d}", arm, "learned"), a.seeds)
                                              for arm in sigma_arms()}
    win = {N: out["gwin"][str(N)][PRIMARY]["safe_success"]["ci95"][0] > 0 for N in NS}
    ni = {N: out["gwin"][str(N)][PRIMARY]["completeness"]["ci95"][0] >= -NI_MARGIN for N in NS}
    both = {N: win[N] and ni[N] for N in NS}
    out["gwin_verdict"] = {"primary_arm": PRIMARY, "safe_win": win, "completeness_noninferior": ni, "per_N": both,
                           "verdict": "PASS" if all(both.values()) else "NOT_MET"}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1)
    fmt = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]"
    for N in NS:
        for arm, r in out["rows"][str(N)].items():
            ex = (f" d_mean={r['d_safe_mean']['mean']:.3f} d_max={out['dsafe_max_realised_m'][str(N)][arm]:.2f} "
                  f"cov2m={r['sigma_cov_rows_near2m']['mean']:.4f}" if arm.startswith("sigma") else "")
            print(f"N{N:3d} {arm:16s} safe={fmt(r['safe_success'])} ever={fmt(r['ever_collided'])} "
                  f"compl={fmt(r['completeness'])}{ex}")
        for arm, v in out["gwin"][str(N)].items():
            s, c = v["safe_success"], v["completeness"]
            print(f"N{N:3d} {arm} - best fixed ({out['best_fixed'][str(N)]}): safe {s['diff']:+.3f} "
                  f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]  compl {c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
                  f"{'  PRIMARY' if arm == PRIMARY else '  secondary'}")
    print("G-win (amend35):", json.dumps(out["gwin_verdict"]))
    print("aneks sha256:", out["aneks_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write-cells", "sanity", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--s2", action="store_true")
    a = ap.parse_args()
    {"write-cells": cmd_write_cells, "sanity": cmd_sanity, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
