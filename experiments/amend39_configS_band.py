"""Amendment 39, addendum 2 (log-only): settled metric with the 0.30 m goal band on the Config S (amend38 A2) cells.

    python -m experiments.amend39_configS_band

Reuses the amend39 per-rollout metric and the amend38 arm list / contrasts. b = 0 must reproduce amend38 A2 exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import yaml

from experiments import amend33_n100_baseline_cpu as a33
from experiments import amend38_settled_recompute as a38
from experiments.amend39_settled_regen import BANDS, WINDOWS, W0, metric_keys, settled_metrics
from experiments.smoke import ROOT, build_scenario

RES = os.path.join(ROOT, "results", "amend39_configS_band")
ADD = os.path.join(ROOT, "prereg", "prereg_projectA_amend39_addendum2.md")


def rows(d, sc):
    out = []
    for s in a38.SEEDS:
        z = a38.load(d, s)
        if z is None:
            continue
        with z:
            r, _ = settled_metrics(z, sc)
        r["seed"] = s
        out.append(r)
    return out


def main():
    cfg = yaml.safe_load(open(a33.BASE))
    sc = build_scenario(cfg, cfg["N"])
    R = {a: rows(d, sc) for a, d in a38.ARMS.items()}
    ref = json.load(open(os.path.join(ROOT, "results", "amend38_settled_recompute", "report.json")))["A2"]
    ks = metric_keys()
    out = {"label": "REGISTERED (amend39 addendum 2, log-only)",
           "addendum_sha256": hashlib.sha256(open(ADD, "rb").read()).hexdigest(),
           "b0_reproduces_amend38": {}, "arms": {}, "contrasts": {}}
    for a, rr in R.items():
        out["arms"][a] = {"n_seeds": len(rr), **{k: a33.boot_ci([r[k] for r in rr]) for k in ks}}
        out["b0_reproduces_amend38"][a] = all(
            abs(out["arms"][a][f"safe_settled_{W:g}s_b0.00"]["mean"] - ref[a][f"safe_settled_{W:g}s"]["mean"]) < 1e-12
            and abs(out["arms"][a][f"completeness_settled_{W:g}s_b0.00"]["mean"]
                    - ref[a][f"completeness_settled_{W:g}s"]["mean"]) < 1e-12 for W in WINDOWS)
    for x, y in a38.CONTRASTS:
        out["contrasts"][f"{x} - {y}"] = {f"safe_settled_{W0:g}s_b{b:.2f}": a33.paired(R[x], R[y], f"safe_settled_{W0:g}s_b{b:.2f}")
                                          for b in BANDS}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "report.json"), "w"), indent=1, default=str)
    ok = all(out["b0_reproduces_amend38"].values())
    print("b=0 reproduces amend38 A2:", ok)
    for a, v in out["arms"].items():
        f = lambda k: f"{v[k]['mean']:.3f}"  # noqa: E731
        print(f"{a:50s} safe5 b0 {f('safe_settled_5s_b0.00')}  b.30 {f('safe_settled_5s_b0.30')}  "
              f"compl5 b0 {f('completeness_settled_5s_b0.00')}  b.30 {f('completeness_settled_5s_b0.30')}  "
              f"ever {f('ever_collided')}")
    for c, v in out["contrasts"].items():
        print(f"{c:60s} " + "  ".join(f"{k[-5:]} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
                                       for k, d in v.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
