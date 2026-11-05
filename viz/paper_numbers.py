"""Headline numbers + Table I of paper/ral_v1.tex, generated (never typed by hand).

    python -m viz.paper_numbers            # writes paper/gen/numbers.tex and paper/gen/table_crossover.tex

Source:
  * amend39 (Config G re-run, settled metric) if results/amend39_settled_regen/{report.json,figure_data_amend39.csv}
    exist -> headline = safe settled W=5 s with the 0.30 m band (prereg amend39 addendum 1), b=0 alongside;
  * otherwise the registered @T reports of amend22/24/26 (LEGACY): every macro then carries the pending marker.
Ever-collided does not depend on the arrival metric; in settled mode it is taken from the amend39 re-run.
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "gen")
R39 = os.path.join(ROOT, "results", "amend39_settled_regen")
NS = [25, 50, 100, 150, 200]
TS = ["ts2.50ms", "ts5.00ms", "ts7.50ms", "ts10.00ms"]
HW, HB = "5", "0.30"


def f3(x):
    return f"{x:.3f}"


def sg(x):
    return f"{x:+.3f}"


def ci(d):
    return f"${sg(d[0])}\\ci{{{sg(d[1])}}}{{{sg(d[2])}}}$"


def msg(x):
    return f"${sg(x)}$"


def settled_source():
    if not (os.path.exists(os.path.join(R39, "report.json")) and os.path.exists(os.path.join(R39, "figure_data_amend39.csv"))):
        return None
    G = {}
    for r in csv.DictReader(open(os.path.join(R39, "figure_data_amend39.csv"))):
        G[(r["cell"], r["metric"])] = (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"]))
    rep = json.load(open(os.path.join(R39, "report.json")))
    hm, h0 = f"safe_settled_{HW}s_b{HB}", "safe_settled_5s_b0.00"

    def c(arm, N, m):
        src = "amend24" if arm.startswith("noise") else "amend22"
        return G.get((f"A__{src}__N{N:03d}_{arm}", m))

    def t(arm, ts, m):
        return G.get((f"B__amend26__{ts}_{arm}", m))

    def gap(N, b):
        d = rep["Q"]["Q2_learned_minus_real0.70"][str(N)][f"b{b}"]
        return (d["mean"], d["ci95"][0], d["ci95"][1]) if d else None

    def tgap(i, b):
        tf = ["0.25", "0.5", "0.75", "1.0"][i]
        d = rep["Q"]["Q4_staleness"][tf]["learned_minus_real0.70"][f"b{b}"]
        return (d["mean"], d["ci95"][0], d["ci95"][1]) if d else None
    q1 = rep["Q"]["Q1_safe_N"]
    safe_n = {a: (q1["b0.00"].get(a), q1["b0.30"].get(a)) for a in ("real_d0.55", "real_d0.70", "learned_d0.70")}
    return dict(mode="settled", c=c, t=t, hm=hm, h0=h0, cm=f"completeness_settled_{HW}s_b{HB}", gap=lambda N: gap(N, HB),
                gap0=lambda N: gap(N, "0.00"), tgap=lambda i: tgap(i, HB), safe_n=safe_n, missing=rep.get("missing", {}))


def legacy_source():
    a22 = json.load(open(os.path.join(ROOT, "results/amend22_N_crossover_pooled/report.json")))
    a24 = json.load(open(os.path.join(ROOT, "results/amend24_mechanism_control/report.json")))
    a26 = json.load(open(os.path.join(ROOT, "results/amend26_tslot_sweep/report.json")))
    key = {"safe": "safe_success", "ever": "ever_collided", "compl": "completeness"}

    def mc(v):
        return (v["mean"], v["ci95"][0], v["ci95"][1])

    def c(arm, N, m):
        rows = a24["rows"] if arm.startswith("noise") else a22["rows"]
        a = "noise_only" if arm.startswith("noise") else arm
        r = rows.get(str(N), {}).get(a)
        return mc(r[key[m]]) if r else None

    def t(arm, ts, m):
        r = a26["rows"].get(ts, {}).get(arm)
        return mc(r[key[m]]) if r else None

    def gap(N):
        d = a22["diffs"][str(N)]["learned_pooled_d0.70_minus_real_d0.70_safe_success"]
        return (d["diff"], d["ci95"][0], d["ci95"][1])

    def tgap(i):
        d = a26["diffs"][TS[i]]["learned_minus_real_d0.70_safe_success"]
        return (d["diff"], d["ci95"][0], d["ci95"][1])
    sn = a22["safe_N"]
    safe_n = {a: (sn[k] if not isinstance(sn[k], dict) else sn[k].get("safe_N"), None)
              for a, k in (("real_d0.55", "real_d0.55"), ("real_d0.70", "real_d0.70"), ("learned_d0.70", "learned_pooled_d0.70"))}
    return dict(mode="legacy", c=c, t=t, hm="safe", h0=None, cm="compl", gap=gap, gap0=None, tgap=tgap, safe_n=safe_n,
                missing={})


def main():
    S = settled_source() or legacy_source()
    os.makedirs(OUT, exist_ok=True)
    pend = S["mode"] == "legacy"
    c, t = S["c"], S["t"]
    ev = "ever_collided" if not pend else "ever"
    mac = {}
    mac["HeadMetric"] = "safe success at $T$" if pend else "safe settled (5\\,s window, 0.30\\,m band)"
    mac["HeadShort"] = "safe@$T$" if pend else "safe settled"
    mac["Pending"] = "\\textsuperscript{\\dag}" if pend else ""
    mac["PendingNote"] = ("\\textsuperscript{\\dag}Scored at the final instant $T$ (registered metric); being re-run "
                          "under the settled metric with archived trajectories (prereg.\\ amend39)." if pend else "")
    for N, w in zip(NS[1:], ("Fifty", "Hundred", "OneFifty", "TwoHundred")):
        mac[f"Gap{w}"] = ci(S["gap"](N)) if S["gap"](N) else "n/a"
        lr, rr, nr = c("learned_pooled_d0.70", N, S["hm"]), c("real_d0.70", N, S["hm"]), c("noise_only_d0.70", N, S["hm"])
        mac[f"Rec{w}"] = (f"{100 * (lr[0] - rr[0]) / (nr[0] - rr[0]):.0f}" if lr and rr and nr and nr[0] > rr[0] else "n/a")
    mac["TauLearned"] = " / ".join(f3(t("learned_pooled_d0.70", x, S["hm"])[0]) for x in TS)
    mac["TauGap"] = ", ".join(msg(S["tgap"](i)[0]) for i in range(4))
    mac["TauEverFixed"] = " / ".join(f3(t("learned_pooled_d0.70", x, ev)[0]) for x in TS)
    mac["TauEverSweep"] = " / ".join(f3(c("learned_pooled_d0.70", N, ev)[0]) for N in NS[1:])
    mac["TauComplFixed"] = " / ".join(f3(t("learned_pooled_d0.70", x, S["cm"])[0]) for x in TS)
    mac["ComplLearnedLargeN"] = " / ".join(f3(c("learned_pooled_d0.70", N, S["cm"])[0]) for N in (150, 200))
    mac["NoiseHead"] = " / ".join(f3(c("noise_only_d0.70", N, S["hm"])[0]) for N in NS)
    mac["Headroom"] = " / ".join(msg(c("noise_only_d0.70", N, S["hm"])[0] - c("learned_pooled_d0.70", N, S["hm"])[0])
                                 for N in (100, 150, 200))

    def sn(a):
        b0, b3 = S["safe_n"][a]
        if pend:
            return f"{b0}"
        return f"{b3 if b3 is not None else '$<$25'} (band), {b0 if b0 is not None else '$<$25'} (no band)"
    mac["SafeNReal"] = sn("real_d0.55") if sn("real_d0.55") == sn("real_d0.70") else f"{sn('real_d0.55')} / {sn('real_d0.70')}"
    mac["SafeNLearned"] = sn("learned_d0.70")
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write(f"% GENERATED by viz/paper_numbers.py (source: {S['mode']}) -- do not edit\n")
        fh.write("\\newif\\ifpending\n" + ("\\pendingtrue\n" if pend else "\\pendingfalse\n"))
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # ---- Table I
    rows = []
    rows.append(("ever", [("real, $d$ 0.70", "real_d0.70"), ("learned v3", "learned_pooled_d0.70"),
                          ("delay removed", "noise_only_d0.70")], ev))
    rows.append(("safe", [("real, $d$ 0.55", "real_d0.55"), ("real, $d$ 0.70", "real_d0.70"),
                          ("learned v3", "learned_pooled_d0.70"), ("delay removed", "noise_only_d0.70")], S["hm"]))
    if not pend:
        rows.append(("safe, b=0", [("learned v3", "learned_pooled_d0.70"), ("delay removed", "noise_only_d0.70")], S["h0"]))
    rows.append(("compl", [("learned v3", "learned_pooled_d0.70"), ("delay removed", "noise_only_d0.70")], S["cm"]))
    lab = {"ever": "ever coll.", "safe": "safe@$T$" if pend else "safe settled", "safe, b=0": "no band",
           "compl": "compl.@$T$" if pend else "compl.\\ settled"}
    L = ["\\begin{tabular}{@{}llccccc@{}}", "\\toprule", " & $N$ & 25 & 50 & 100 & 150 & 200\\\\",
         " & $\\Tf$ [s] & 0.125 & 0.25 & 0.5 & 0.75 & 1.0\\\\"]
    for g, arms, m in rows:
        L.append("\\midrule")
        for i, (name, arm) in enumerate(arms):
            vals = [c(arm, N, m) for N in NS]
            cells = " & ".join(f3(v[0]) if v else "--" for v in vals)
            head = f"\\multirow{{{len(arms)}}}{{*}}{{\\rotatebox{{90}}{{\\scriptsize {lab[g]}}}}}" if i == 0 else ""
            L.append(f"{head} & {name} & {cells}\\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    open(os.path.join(OUT, "table_crossover.tex"), "w").write(
        f"% GENERATED by viz/paper_numbers.py (source: {S['mode']}) -- do not edit\n" + "\n".join(L) + "\n")
    print(f"[paper_numbers] source={S['mode']} missing={S['missing']}")
    for k, v in mac.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
