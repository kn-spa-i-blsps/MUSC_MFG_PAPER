"""Log-only diagnosis (no rollouts): why do 13-15 / 100 drones end outside the goal disc without delay?
amend11 config, N=100, seeds 0-4. Reads existing npz logs (amend33 fresh / fresh_nocbf / fresh_nomfg / noise_only,
amend34 A) and the existing macro cache; writes results/diag_goal_ceiling/diag.json.

    python -m experiments.diag_goal_ceiling
"""
from __future__ import annotations

import json
import os

import numpy as np
import yaml

from experiments.smoke import ROOT, build_scenario

BASE = os.path.join(ROOT, "experiments", "ladder_fargoal_amend11.yaml")
CACHE = os.path.join(ROOT, "results", "cache", "macro_N100_cw1_13e24892b8.npz")
A33 = os.path.join(ROOT, "results", "amend33_n100_baseline_cpu")
ARMS = {"fresh": os.path.join(A33, "fresh", "fresh"),
        "noise_only": os.path.join(A33, "noise_only", "noise_only"),
        "A_cbf_fresh": os.path.join(ROOT, "results", "amend34_layer_delay_cpu", "ab", "split_real-noise_only"),
        "fresh_nocbf": os.path.join(A33, "fresh_nocbf", "fresh"),
        "fresh_nomfg": os.path.join(A33, "fresh_nomfg", "fresh")}
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(ROOT, "results", "diag_goal_ceiling")
LAST = 100          # last 1 s (dt 0.01) for time-averaged end-state controls


def bilinear(f, p, step):
    """Same sampler as the controller (control.law.sample_field_bilinear: grid nodes at i*step)."""
    import jax.numpy as jnp
    from control.law import sample_field_bilinear
    return np.asarray(sample_field_bilinear(jnp.asarray(f), jnp.asarray(p), step))


def macro_facts(sc):
    z = np.load(CACHE)
    m = z["m_star"]
    step, dt = float(z["step"]), float(z["dt"])
    nx, nz = m.shape[1:]
    X, Z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step, indexing="ij")
    R = np.hypot(X - sc.goal[0], Z - sc.goal[1])
    ing = R <= sc.r_goal
    T = m.shape[0]
    ts = [int(round(s / dt)) for s in (10, 20, 30, 40, 50, 60)] + [T - 1]
    in_goal = {f"{k * dt:.1f}": float((m[k] * ing).sum() * step ** 2) for k in ts}
    edges = np.arange(0, sc.r_goal + 2.01, 0.5)
    prof = [float((m[T - 1] * ((R >= a) & (R < b))).sum() * step ** 2) for a, b in zip(edges[:-1], edges[1:])]
    pg = z["phi_grad"][T - 1]
    return {"horizon_s": (T - 1) * dt, "mstar_mass_in_disc_by_time_s": in_goal,
            "mstar_final_mass_by_radius": {"edges_m": edges.tolist(), "mass": prof},
            "mstar_final_peak_in_disc": float(m[T - 1][ing].max())}, pg, step, z["m_star"][T - 1]


def seed_facts(d, s, sc, pg_T, step):
    z = np.load(os.path.join(d, f"logs_seed{s}.npz"))
    pos = z["pos"].astype(np.float64)
    pT = z["pos_final"].astype(np.float64)
    g = np.asarray(sc.goal)
    r_t = np.linalg.norm(pos - g, axis=-1)                         # (T, N)
    rT = np.linalg.norm(pT - g, axis=-1)
    inside_t = r_t <= sc.r_goal
    out = rT > sc.r_goal
    ever = inside_t.any(0)
    radial = (pT - g) / np.maximum(rT[:, None], 1e-9)               # outward unit vector
    u_nom = z["u_nom"][-LAST:].mean(0)
    u = z["u"][-LAST:].mean(0)
    route = bilinear(pg_T, pT, step)                                 # -grad Phi field at t = T as used by u_nom
    ur_nom = (u_nom * radial).sum(-1)                                # < 0 = pulls inward
    ur = (u * radial).sum(-1)
    ur_route = (route * radial).sum(-1)
    dd = np.linalg.norm(pT[:, None] - pT[None], axis=-1) + np.eye(len(pT)) * 1e9
    nn = dd.min(1)
    n_within_08 = (dd < 0.80).sum(1)
    inter = z["intervention"][-LAST:].mean(0)
    last_inside = np.where(ever, (inside_t.shape[0] - 1 - np.argmax(inside_t[::-1], 0)) * 0.01, np.nan)
    disc_area = np.pi * sc.r_goal ** 2

    def summ(mask, x):
        return None if not mask.any() else [float(np.min(x[mask])), float(np.median(x[mask])), float(np.max(x[mask]))]

    return {"seed": s, "n_out": int(out.sum()),
            "out_ever_inside": int((out & ever).sum()),
            "out_last_inside_s_min_med_max": summ(out & ever, last_inside),
            "out_excess_r_m_min_med_max": summ(out, rT - sc.r_goal),
            "out_within_0p5m_of_rim": int((out & (rT - sc.r_goal <= 0.5)).sum()),
            "out_upstream": int((out & (pT[:, 0] < sc.wall_x)).sum()),
            "out_speed_last1s_med": summ(out, np.linalg.norm(u, axis=-1)),
            "out_unom_radial_min_med_max": summ(out, ur_nom),
            "out_route_radial_min_med_max": summ(out, ur_route),
            "out_u_radial_min_med_max": summ(out, ur),
            "out_unom_norm_med": summ(out, np.linalg.norm(u_nom, axis=-1)),
            "out_intervention_rate_last1s_med": summ(out, inter),
            "out_nearest_nb_m_med": summ(out, nn),
            "in_unom_radial_med": summ(~out, ur_nom),
            "in_nearest_nb_m_min_med_max": summ(~out, nn),
            "in_count": int((~out).sum()), "in_density_per_m2": float((~out).sum() / disc_area),
            "rim_shell_count_r_in_[r_goal-1,r_goal]": int(((rT > sc.r_goal - 1) & ~out).sum()),
            "core_count_r_lt_2": int((rT < 2.0).sum()),
            "radial_counts_0.5m": np.histogram(rT, np.arange(0, sc.r_goal + 2.01, 0.5))[0].tolist(),
            "frac_left_after_arrival": float((ever & out).sum() / max(ever.sum(), 1))}


def main():
    cfg = yaml.safe_load(open(BASE))
    sc = build_scenario(cfg, cfg["N"])
    mac, pg_T, step, _ = macro_facts(sc)
    print("macro:", json.dumps(mac))
    res = {"label": "DESCRIPTIVE, log-only (no rollouts)", "r_goal_m": sc.r_goal,
           "hex_capacity_density_per_m2": 2 / (np.sqrt(3) * 0.8 ** 2), "macro": mac, "arms": {}}
    for arm, d in ARMS.items():
        rows = [seed_facts(d, s, sc, pg_T, step) for s in SEEDS if os.path.exists(os.path.join(d, f"logs_seed{s}.npz"))]
        res["arms"][arm] = rows
        for r in rows:
            print(arm, json.dumps(r))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
