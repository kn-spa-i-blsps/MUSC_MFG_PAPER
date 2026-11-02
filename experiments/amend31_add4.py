"""Amendment 31 addendum 4 (SUPPLEMENTARY; prereg/prereg_projectA_amend31_addendum4.md). Never changes the registered
amend31 G-win verdict. Subcommands (run in this order by logs/overnight/amend31_add4_chain_pod.sh):

  qraw        CPU: q_raw(N) = finite-sample split-conformal 0.95 quantile of the raw v3 error ||x_hat - x|| on the CAL
              windows (seeds 108-109, true distance < 2 m; the stage-1 CAL draw), N in {150, 200}
  freeze      after the registered amend31 report: best fixed per N (from report.json), fixed_match(N) = max(0.70,
              row-weighted mean realised d_safe of sigma_bin_a0.05 over seeds 0-9), conf_const d_safe; f32 aliases
  write-cells cells for priorities A / B / C and the gates (fixed path = amend31 fixed cells with only d_safe changed)
  gate        g1 harness identity vs amend22, g2 perturbation-off == base, g3 perturbation-on diverges (step)
  analyze     all addendum-4 statistics -> results/amend31_add4/analysis.json
  animate     N=200 seed 0 GIFs: sigma_bin_a0.05, best fixed, fixed_match
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os

import numpy as np
import yaml

from experiments.smoke import ROOT, build_scenario

RUN31, RUN = "amend31_sigma_cbf", "amend31_add4"
RES31, RES = os.path.join(ROOT, "results", RUN31), os.path.join(ROOT, "results", RUN)
CD31, CD = os.path.join(ROOT, "experiments", "amend31_cells"), os.path.join(ROOT, "experiments", "amend31_add4_cells")
W3 = "results/overnight_p1/predictor_v3/predictor_weights.npz"
NS_ALL, NS4 = (25, 50, 100, 150, 200), (150, 200)
SEEDS, REP, MEMBERS = list(range(10)), list(range(200, 210)), (1, 2, 3, 4, 5)
FIXED = (0.70, 0.85, 1.00, 1.20)
ARMS31 = [f"fixed{d:.2f}" for d in FIXED] + ["sigma_a0.1", "sigma_a0.05", "sigma_a0.01", "sigma_bin_a0.05"]
SB = "sigma_bin_a0.05"
NBOOT = 10_000


def jload(p):
    return json.load(open(p))


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, default=float)


# ----------------------------------------------------------------------------------------------------------- qraw
def cmd_qraw(a):
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    from experiments.overnight_p1_predictor import datagen_dir
    from experiments.phaseB_gpred import EVAL_R, load
    w = {k: v.astype(np.float64) for k, v in np.load(os.path.join(ROOT, W3)).items()}
    out = {"definition": "q_raw(N) = ceil((n+1)*0.95)-th smallest raw error ||v3(X) - Y|| over CAL windows "
                         "(seeds 108-109, 200k uniform draws / seed / N, true distance < 2 m); d_safe = max(0.70, 0.30 + q_raw)",
           "per_N": {}}
    for N in NS4:
        _, E, _, D = load({N: datagen_dir(N, "datagenL")}, [108, 109], 200_000, 0.0, w)
        e = np.sort(E[D < EVAL_R])
        n, k = e.size, math.ceil((e.size + 1) * 0.95)
        q = float(e[k - 1]) if k <= n else float("inf")
        out["per_N"][str(N)] = {"n_cal": int(n), "k": int(k), "q_raw": q, "d_safe": max(0.70, 0.30 + q)}
        print(N, out["per_N"][str(N)], flush=True)
    jdump(out, os.path.join(RES, "q_raw.json"))


# --------------------------------------------------------------------------------------------------------- freeze
def alias_of(d):
    for f in FIXED:
        if np.float32(d) == np.float32(f):
            return f"fixed{f:.2f}"
    return None


def cmd_freeze(a):
    rep = jload(os.path.join(RES31, "report.json"))
    qr = jload(os.path.join(RES, "q_raw.json"))["per_N"]
    fz = {"note": "frozen after the registered amend31 report, seeds 0-9 only", "per_N": {}}
    for N in NS4:
        best = rep["best_fixed"][str(N)]
        s_sum, s_n = 0.0, 0
        for s in SEEDS:
            z = np.load(os.path.join(RES31, f"N{N:03d}", SB, "learned", f"logs_seed{s}.npz"))
            s_sum += float(np.sum(np.asarray(z["d_safe_sum"], np.float64)))
            s_n += int(np.sum(np.asarray(z["d_safe_n"], np.int64)))
        dm = s_sum / s_n
        arms = {"conf_const_a0.05": qr[str(N)]["d_safe"], "fixed_match": max(0.70, dm)}
        fz["per_N"][str(N)] = {
            "best_fixed": best, "best_fixed_d": float(best[5:]),
            "sigma_bin_dsafe_mean_rows_all_valid": dm, "sigma_bin_dsafe_rows": s_n,
            "arms": {k: {"d_safe": v, "d_safe_f32": float(np.float32(v)), "alias": alias_of(v)} for k, v in arms.items()}}
    jdump(fz, os.path.join(RES, "frozen.json"))
    print(json.dumps(fz, indent=1))


# ---------------------------------------------------------------------------------------------------- write-cells
def _cell(src, run_name, **upd):
    c = copy.deepcopy(yaml.safe_load(open(os.path.join(CD31, src))))
    c["run_name"] = run_name
    for k, v in upd.items():
        if k == "d_safe":
            c["cbf"]["d_safe"] = float(v)
        else:
            c[k] = v
    return c


def _write(p, c, note):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(f"# AMEND31 ADDENDUM 4 cell: {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)


def dir_of(N, arm, setname, fz):
    """results dir (…/<chain_arm>) of an arm in a seed set; aliases resolve to existing amend31 fixed runs (seeds 0-9)
    or to the shared replication run of the same d_safe."""
    pn = fz["per_N"][str(N)]
    if setname == "main":
        if arm in ARMS31:
            return os.path.join(RES31, f"N{N:03d}", arm, "learned")
        al = pn["arms"][arm]["alias"]
        if al:
            return os.path.join(RES31, f"N{N:03d}", al, "learned")
        return os.path.join(RES, "main", f"N{N:03d}", arm, "learned")
    if setname == "rep":
        if arm == SB:
            return os.path.join(RES, "rep", f"N{N:03d}", SB, "learned")
        d = pn["best_fixed_d"] if arm == "best_fixed" else pn["arms"][arm]["d_safe"]
        return os.path.join(RES, "rep", f"N{N:03d}", f"d{np.float32(d):.6f}", "learned")
    raise ValueError(setname)


def cmd_write_cells(a):
    fz = jload(os.path.join(RES, "frozen.json"))
    jobs = []   # (priority, cell path, seeds)
    for N in NS4:
        pn = fz["per_N"][str(N)]
        for arm, v in pn["arms"].items():           # priority A
            if v["alias"] is None:
                p = os.path.join(CD, "A", f"N{N:03d}_{arm}.yaml")
                _write(p, _cell(f"N{N:03d}_fixed0.70.yaml", f"{RUN}/main/N{N:03d}/{arm}", d_safe=v["d_safe"]),
                       f"priority A {arm} N={N} d_safe={v['d_safe']!r} (fixed path)")
                jobs.append(("A", p, SEEDS))
        p = os.path.join(CD, "B", f"N{N:03d}_{SB}.yaml")  # priority B
        _write(p, _cell(f"N{N:03d}_{SB}.yaml", f"{RUN}/rep/N{N:03d}/{SB}"), f"priority B {SB} N={N} seeds 200-209")
        jobs.append(("B", p, REP))
        ds = {np.float32(pn["best_fixed_d"]): pn["best_fixed_d"]}
        for v in pn["arms"].values():
            ds.setdefault(np.float32(v["d_safe"]), v["d_safe"])
        for d32, d in ds.items():
            p = os.path.join(CD, "B", f"N{N:03d}_d{d32:.6f}.yaml")
            _write(p, _cell(f"N{N:03d}_fixed0.70.yaml", f"{RUN}/rep/N{N:03d}/d{d32:.6f}", d_safe=d),
                   f"priority B fixed path d_safe={d!r} N={N} seeds 200-209 (shared by every B arm with this f32 d_safe)")
            jobs.append(("B", p, REP))
        for arm, src in ((SB, f"N{N:03d}_{SB}.yaml"), ("best_fixed", f"N{N:03d}_{pn['best_fixed']}.yaml")):  # C
            for m in MEMBERS:
                p = os.path.join(CD, "C", f"N{N:03d}_{arm}_m{m}.yaml")
                _write(p, _cell(src, f"{RUN}/chaos/N{N:03d}/{arm}_m{m}", init_perturb_agent=m - 1),
                       f"priority C {arm} N={N} member m={m}: x of agent {m - 1} += 1 ulp (f32, toward +inf)")
                jobs.append(("C", p, SEEDS))
    gates = {"g1": _cell("identity/N100_fixed0.70_real_ref.yaml", f"{RUN}/gates/g1_fixed0.70_real"),
             "g2": _cell(f"N150_{SB}.yaml", f"{RUN}/gates/g2_perturb_off"),
             "g3": _cell(f"N150_{SB}.yaml", f"{RUN}/gates/g3_perturb_on", init_perturb_agent=0)}
    for g, c in gates.items():
        _write(os.path.join(CD, "gates", f"{g}.yaml"), c, f"gate {g}")
    with open(os.path.join(CD, "jobs.txt"), "w") as f:
        for pr, p, seeds in jobs:
            f.write(f"{pr} {os.path.relpath(p, ROOT)} {' '.join(map(str, seeds))}\n")
    print(f"{len(jobs)} jobs, {sum(len(s) for _, _, s in jobs)} rollouts", flush=True)


# ----------------------------------------------------------------------------------------------------------- gate
def cmd_gate(a):
    from experiments.amend22_N_crossover_pooled import IDENTITY_KEYS
    out = {}
    a22 = jload(os.path.join(ROOT, "results", "amend22_N_crossover_pooled", "N100", "real_d0.70", "real", "metrics_seed0.json"))
    g1 = jload(os.path.join(RES, "gates", "g1_fixed0.70_real", "real", "metrics_seed0.json"))
    out["g1"] = {"keys": {k: [a22[k], g1[k]] for k in IDENTITY_KEYS},
                 "pass": all(a22[k] == g1[k] for k in IDENTITY_KEYS)}
    base = np.load(os.path.join(RES31, "N150", SB, "learned", "logs_seed0.npz"))
    off = np.load(os.path.join(RES, "gates", "g2_perturb_off", "learned", "logs_seed0.npz"))
    on = np.load(os.path.join(RES, "gates", "g3_perturb_on", "learned", "logs_seed0.npz"))
    neq = [k for k in base.files if not np.array_equal(base[k], off[k], equal_nan=True)]
    out["g2"] = {"arrays": len(base.files), "not_equal": neq, "pass": not neq}
    if neq and "pos" in neq:
        out["g2"]["first_divergence_step_pos"] = int(np.argmax(np.any(base["pos"] != off["pos"], axis=(1, 2))))
    p0, p1 = np.asarray(base["pos"]), np.asarray(on["pos"])
    diff = np.any(p0 != p1, axis=(1, 2))
    out["g3"] = {"changed": bool(diff.any()),
                 "first_divergence_step_pos": int(np.argmax(diff)) if diff.any() else None,
                 "step0_agent0_x": [repr(np.float32(p0[0, 0, 0])), repr(np.float32(p1[0, 0, 0]))],
                 "step0_other_entries_equal": bool(np.array_equal(np.delete(p0[0].ravel(), 0), np.delete(p1[0].ravel(), 0))),
                 "pass": bool(diff.any())}
    jdump(out, os.path.join(RES, "gates.json"))
    print(json.dumps({g: out[g]["pass"] for g in out}), json.dumps(out["g3"]))
    if a.which:
        raise SystemExit(0 if all(out[g]["pass"] for g in a.which.split(",")) else 1)


# -------------------------------------------------------------------------------------------------------- analyze
def per_seed(rdir, sc, seeds):
    from experiments.amend15_density_sweep import safe_success
    rows = []
    for s in seeds:
        z = np.load(os.path.join(rdir, f"logs_seed{s}.npz"))
        m = jload(os.path.join(rdir, f"metrics_seed{s}.json"))
        f = safe_success({"pos": z["pos"].astype(np.float64), "pos_final": z["pos_final"].astype(np.float64)}, sc, 0.30)
        for k in ("makespan_s", "t_first_90pct", "mean_speed_mps", "d_safe_mean", "d_safe_p95", "sigma_bind_frac",
                  "sigma_cov_rows", "sigma_cov_rows_near2m", "sigma_cov_binding", "sigma_cov_n_near2m"):
            f[k] = m.get(k)
        rows.append(f)
    return rows


def boot_mean_ci(x, rng):
    x = np.asarray(x, float)
    b = rng.choice(x, (NBOOT, x.size)).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def compare(a, b, seed=0):
    """a, b: per-seed values on the SAME seed list (paired by index)."""
    from experiments.amend20_closing_lockin import diff_ci
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (NBOOT, d.size))
    pb = d[idx].mean(1)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=d.size)))
    null = (signs * d[None]).mean(1)
    p = float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-15))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"diff": float(d.mean()), "unpaired": diff_ci(a, b)["ci95"],
            "paired_ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))],
            "corr_across_seeds": corr, "sign_flip_p_two_sided": p, "n_sign_patterns": int(signs.shape[0])}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def cmd_analyze(a):
    fz = jload(os.path.join(RES, "frozen.json"))
    rep = jload(os.path.join(RES31, "report.json"))
    cal = jload(os.path.join(ROOT, "results", "phaseB", "sigma_v3", "calibration.json"))
    rng = np.random.default_rng(1)
    out = {"label": "SUPPLEMENTARY except 'registered'", "registered": {
        "gwin_verdict_addendum1": rep.get("gwin_verdict_addendum1"), "gwin_verdict_v1": rep.get("gwin_verdict"),
        "gcal_verdict": rep["gcal"]["verdict"], "best_fixed": rep["best_fixed"]},
        "raw_means_all_N": {}, "comparisons": {}, "chaos": {}, "closed_loop_coverage": {}, "costs": {}, "rules": {}}
    scs = {}
    for N in NS_ALL:
        c = yaml.safe_load(open(os.path.join(CD31, f"N{N:03d}_fixed0.70.yaml")))
        scs[N] = build_scenario(c, N)
        out["raw_means_all_N"][str(N)] = {arm: {k: rep["rows"][str(N)][arm][k]["mean"] for k in
                                                ("safe_success", "ever_collided", "completeness")} for arm in ARMS31}
    ps_cache = {}

    def get(N, arm, setname):
        key = (N, arm, setname)
        if key not in ps_cache:
            seeds = SEEDS if setname == "main" else REP
            ps_cache[key] = per_seed(dir_of(N, arm, setname, fz), scs[N], seeds)
        return ps_cache[key]

    def available(N, arm, setname):
        d = dir_of(N, arm, setname, fz)
        seeds = SEEDS if setname == "main" else REP
        return all(os.path.exists(os.path.join(d, f"metrics_seed{s}.json")) for s in seeds)

    comps = ("best_fixed", "conf_const_a0.05", "fixed_match")
    for setname in ("main", "rep"):
        res = {}
        for N in NS4:
            pn = fz["per_N"][str(N)]
            for comp in comps:
                carm = pn["best_fixed"] if (comp == "best_fixed" and setname == "main") else comp
                if not (available(N, SB, setname) and available(N, carm, setname)):
                    res.setdefault(comp, {})[str(N)] = None
                    continue
                A, B = get(N, SB, setname), get(N, carm, setname)
                res.setdefault(comp, {})[str(N)] = {m: compare([r[m] for r in A], [r[m] for r in B])
                                                    for m in ("safe_success", "completeness")}
                res[comp][str(N)]["means"] = {"sigma_bin": {m: float(np.mean([r[m] for r in A])) for m in ("safe_success", "completeness", "ever_collided")},
                                              comp: {m: float(np.mean([r[m] for r in B])) for m in ("safe_success", "completeness", "ever_collided")}}
            for comp in comps:   # Holm over N = 150, 200 per comparison and metric
                if all(res[comp].get(str(N)) for N in NS4):
                    for m in ("safe_success", "completeness"):
                        adj = holm([res[comp][str(N)][m]["sign_flip_p_two_sided"] for N in NS4])
                        for N, pa in zip(NS4, adj):
                            res[comp][str(N)][m]["holm_p"] = pa
        out["comparisons"][setname] = res
    # combined 20 seeds (supplementary only)
    comb = {}
    for N in NS4:
        pn = fz["per_N"][str(N)]
        for comp in comps:
            carm_main = pn["best_fixed"] if comp == "best_fixed" else comp
            if all(available(N, x, s) for x, s in ((SB, "main"), (carm_main, "main"), (SB, "rep"), (comp, "rep"))):
                A = get(N, SB, "main") + get(N, SB, "rep")
                B = get(N, carm_main, "main") + get(N, comp, "rep")
                comb.setdefault(comp, {})[str(N)] = {m: {k: v for k, v in compare([r[m] for r in A], [r[m] for r in B]).items()
                                                         if k != "sign_flip_p_two_sided"} for m in ("safe_success", "completeness")}
    out["comparisons"]["combined_20_seeds_supplementary"] = comb
    # chaos ensemble
    for N in NS4:
        pn = fz["per_N"][str(N)]
        ch = {}
        for arm, base_arm in ((SB, SB), ("best_fixed", pn["best_fixed"])):
            dirs = [os.path.join(RES31, f"N{N:03d}", base_arm, "learned")] + \
                   [os.path.join(RES, "chaos", f"N{N:03d}", f"{arm}_m{m}", "learned") for m in MEMBERS]
            if not all(os.path.exists(os.path.join(d, f"metrics_seed{s}.json")) for d in dirs for s in SEEDS):
                ch[arm] = None
                continue
            M = np.array([[r["safe_success"] for r in per_seed(d, scs[N], SEEDS)] for d in dirs]).T   # (seed, member)
            within = float(M.var(axis=1, ddof=1).mean())
            between = float(M.mean(axis=1).var(ddof=1))
            ch[arm] = {"values_seed_by_member": M.tolist(), "sd_across_members_per_seed": M.std(axis=1, ddof=1).tolist(),
                       "var_within_seed_chaotic": within, "var_between_seed_means": between,
                       "chaotic_share_of_total": within / (within + between) if within + between > 0 else None,
                       "member_mean_per_seed": M.mean(axis=1).tolist()}
        if ch.get(SB) and ch.get("best_fixed"):
            a_, b_ = np.array(ch[SB]["member_mean_per_seed"]), np.array(ch["best_fixed"]["member_mean_per_seed"])
            ch["diff_sigma_bin_minus_best_on_member_means"] = compare(a_, b_)
        out["chaos"][str(N)] = ch
    # closed-loop coverage + costs
    for N in NS4:
        pn = fz["per_N"][str(N)]
        A = get(N, SB, "main")
        off = cal["test_coverage_bins"]["0.05"][str(N)]
        cl = {}
        for k, offv in (("sigma_cov_rows", None), ("sigma_cov_rows_near2m", off["coverage"]),
                        ("sigma_cov_binding", off["binding_coverage"])):
            v = [r[k] for r in A]
            cl[k] = {"mean": float(np.mean(v)), "ci95_seed_boot": boot_mean_ci(v, rng), "offline_gcal_binned_a0.05": offv,
                     "delta_vs_offline": (float(np.mean(v)) - offv) if offv is not None else None}
        cl["note_offline"] = ("offline = G-cal TEST (datagenL, fixed 0.70 closed loop, pairs < 2 m); offline 'binding' = "
                              "sigma_c > 0.40 m; closed-loop 'binding' = rows with d_safe > 0.70 floor (same condition)")
        out["closed_loop_coverage"][str(N)] = cl
        costs = {}
        for arm in (SB, pn["best_fixed"], "conf_const_a0.05", "fixed_match"):
            if not available(N, arm, "main"):
                continue
            R = get(N, arm, "main")
            costs[arm] = {k: (float(np.mean([r[k] for r in R if r[k] is not None])) if any(r[k] is not None for r in R) else None)
                          for k in ("makespan_s", "t_first_90pct", "mean_speed_mps", "d_safe_mean", "d_safe_p95",
                                    "sigma_bind_frac", "ever_collided", "safe_success", "completeness")}
            costs[arm]["makespan_none_seeds"] = sum(r["makespan_s"] is None for r in R)
            costs[arm]["t90_none_seeds"] = sum(r["t_first_90pct"] is None for r in R)
        out["costs"][str(N)] = costs
    # pre-registered reading rules (addendum 4 step 4.5)
    def lo(x, m):
        return x[m]["paired_ci95"][0] if x else None
    win = True
    for setname in ("main", "rep"):
        for N in NS4:
            for comp in ("conf_const_a0.05", "fixed_match"):
                x = out["comparisons"][setname][comp].get(str(N))
                if not x or not (lo(x, "safe_success") > 0 and lo(x, "completeness") >= -0.02):
                    win = False
    out["rules"]["sigma_beats_constant_per_N_schedule"] = win
    out["rules"]["effect_above_chaos"] = {str(N): (out["chaos"][str(N)].get("diff_sigma_bin_minus_best_on_member_means", {})
                                                   .get("paired_ci95", [None])[0] or -1) > 0
                                          if out["chaos"][str(N)].get("diff_sigma_bin_minus_best_on_member_means") else None
                                          for N in NS4}
    out["rules"]["calibration_transfers_closed_loop"] = all(
        out["closed_loop_coverage"][str(N)][k]["mean"] >= 0.93 for N in NS4 for k in ("sigma_cov_rows_near2m", "sigma_cov_binding"))
    jdump(out, os.path.join(RES, "analysis.json"))
    print(json.dumps(out["rules"]), flush=True)


# -------------------------------------------------------------------------------------------------------- animate
def cmd_animate(a):
    from env.rollout import summarize
    from mfg.equilibrium import load_macro
    from viz import render
    fz = jload(os.path.join(RES, "frozen.json"))
    pn = fz["per_N"]["200"]
    os.makedirs(os.path.join(RES, "anim"), exist_ok=True)
    c = yaml.safe_load(open(os.path.join(CD31, "N200_fixed0.70.yaml")))
    sc = build_scenario(c, 200)
    paths = {}
    for name, d in ((SB, dir_of(200, SB, "main", fz)), ("best_fixed_" + pn["best_fixed"], dir_of(200, pn["best_fixed"], "main", fz)),
                    ("fixed_match", dir_of(200, "fixed_match", "main", fz))):
        z = dict(np.load(os.path.join(d, "logs_seed0.npz")))
        m = jload(os.path.join(d, "metrics_seed0.json"))
        macro = load_macro(os.path.join(ROOT, m["macro_cache"]))
        _, series = summarize(z, sc, c["dt_exec"], c_w=1.0)
        p = os.path.join(RES, "anim", f"N200_seed0_{name}.gif")
        n = render.animate(p, sc, z, series, macro, dt_exec=c["dt_exec"], frame_dt=0.5,
                           title=f"amend31 add4 N=200 seed 0 {name} (orange = CBF active, red x = true d<0.30)")
        paths[name] = os.path.relpath(p, ROOT)
        print(p, n, "frames", flush=True)
    jdump(paths, os.path.join(RES, "anim", "paths.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qraw", "freeze", "write-cells", "gate", "analyze", "animate"])
    ap.add_argument("--which", default="")
    a = ap.parse_args()
    {"qraw": cmd_qraw, "freeze": cmd_freeze, "write-cells": cmd_write_cells, "gate": cmd_gate,
     "analyze": cmd_analyze, "animate": cmd_animate}[a.cmd](a)


if __name__ == "__main__":
    main()
