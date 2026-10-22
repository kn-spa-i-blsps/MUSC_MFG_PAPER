"""Standing visual control (prereg §8): m(t) animation (equilibrium m* and realized density) with
agents, collision events and CBF-intervention markers; arrived-fraction over time; trajectories;
safety / CBF time series; goal-zone density profile and distance-to-goal distribution."""
from __future__ import annotations

import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from control.params import D_COLLISION, H_BANDWIDTH  # noqa: E402

INTERV_EPS = 0.01  # m/s, |u - u_nom| above which an agent counts as CBF-intervened


def kde_grid(pos, xs, zs, h=H_BANDWIDTH):
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    d2 = (X[..., None] - pos[:, 0]) ** 2 + (Z[..., None] - pos[:, 1]) ** 2
    return np.exp(-d2 / (2 * h * h)).sum(-1) / (2 * np.pi * h * h)


def _draw_scene(ax, sc):
    W, H = sc.arena
    for seg in sc.wall_segments():
        ax.plot(seg[:, 0], seg[:, 1], color="k", lw=3)
    ax.add_patch(plt.Circle(sc.goal, sc.r_goal, fill=False, ls="--", color="tab:green", lw=1.5))
    att = getattr(sc, "attractor", sc.goal)
    if tuple(att) != tuple(sc.goal):                       # amend10: V_goal attractor off the disc centre
        ax.plot(*att, marker="x", color="tab:red", ms=8, mew=2)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _collision_agents(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 1e9
    return np.where((d < D_COLLISION).any(1))[0]


def animate(path, sc, logs, series, macro, *, dt_exec, frame_dt=0.25, title=""):
    pos = np.asarray(logs["pos"])
    interv = np.asarray(logs["intervention"])
    slack = np.asarray(logs["slack_max"])
    m_star = np.asarray(macro["m_star"])
    xs = np.linspace(0, sc.arena[0], m_star.shape[1])
    zs = np.linspace(0, sc.arena[1], m_star.shape[2])
    vmax = max(float(np.quantile(m_star, 0.999)), 4.0)
    t_all = series["t"]
    frac = series["arrived_frac"]
    stride = max(1, int(round(frame_dt / dt_exec)))
    frames = []
    for k in list(range(0, pos.shape[0], stride)) + [pos.shape[0] - 1]:
        t = k * dt_exec
        fig = plt.figure(figsize=(12, 6.2), dpi=80)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        ti = min(int(round(t / macro["dt"])), m_star.shape[0] - 1)
        ax0.imshow(m_star[ti].T, origin="lower", extent=(0, sc.arena[0], 0, sc.arena[1]),
                   vmin=0, vmax=vmax, cmap="magma")
        _draw_scene(ax0, sc)
        ax0.set_title(f"equilibrium m*(t)   t={t:5.2f}s")
        p = pos[k]
        ax1.imshow(kde_grid(p, xs, zs).T, origin="lower", extent=(0, sc.arena[0], 0, sc.arena[1]),
                   vmin=0, vmax=vmax, cmap="magma")
        _draw_scene(ax1, sc)
        act = interv[k] > INTERV_EPS
        ax1.scatter(p[~act, 0], p[~act, 1], s=9, c="tab:cyan", edgecolors="none", label="nominal")
        ax1.scatter(p[act, 0], p[act, 1], s=14, c="tab:orange", edgecolors="none", label="CBF active")
        sl = slack[k] > 1e-4
        if sl.any():
            ax1.scatter(p[sl, 0], p[sl, 1], s=40, facecolors="none", edgecolors="yellow", label="slack>0")
        ca = _collision_agents(p)
        if len(ca):
            ax1.scatter(p[ca, 0], p[ca, 1], s=60, marker="x", c="red", label="collision")
        ax1.set_title(f"realized m(t) + agents   arrived={frac[k]:.2f}")
        ax1.legend(loc="lower left", fontsize=7, framealpha=0.6)
        ax2.plot(t_all, frac, color="tab:green")
        ax2.axvline(t, color="k", lw=0.8)
        ax2.axhline(0.9, color="gray", ls=":", lw=0.8)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlim(0, t_all[-1])
        ax2.set_ylabel("arrived frac")
        ax2.set_xlabel("t [s]")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, duration=frame_dt * 400, loop=0)
    return len(frames)


def snapshots(path, sc, logs, series, times, *, dt_exec):
    pos = np.asarray(logs["pos"])
    interv = np.asarray(logs["intervention"])
    xs = np.linspace(0, sc.arena[0], 59)
    zs = np.linspace(0, sc.arena[1], 44)
    fig, axs = plt.subplots(2, (len(times) + 1) // 2, figsize=(4 * ((len(times) + 1) // 2), 6.5), dpi=90)
    for ax, t in zip(axs.flat, times):
        k = min(int(round(t / dt_exec)), pos.shape[0] - 1)
        p = pos[k]
        ax.imshow(kde_grid(p, xs, zs).T, origin="lower", extent=(0, sc.arena[0], 0, sc.arena[1]),
                  vmin=0, vmax=6, cmap="magma")
        _draw_scene(ax, sc)
        act = interv[k] > INTERV_EPS
        ax.scatter(p[~act, 0], p[~act, 1], s=6, c="tab:cyan")
        ax.scatter(p[act, 0], p[act, 1], s=9, c="tab:orange")
        ca = _collision_agents(p)
        if len(ca):
            ax.scatter(p[ca, 0], p[ca, 1], s=40, marker="x", c="red")
        ax.set_title(f"t={k * dt_exec:.1f}s arrived={series['arrived_frac'][k]:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_panel(path, sc, runs, *, dt_exec):
    """runs: {label: (logs, series)}. Trajectories of the first run + time series of all runs."""
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), dpi=90)
    label0 = next(iter(runs))
    logs0, series0 = runs[label0]
    pos = np.asarray(logs0["pos"])
    ax = axs[0, 0]
    for i in range(pos.shape[1]):
        ax.plot(pos[:, i, 0], pos[:, i, 1], lw=0.4, alpha=0.6)
    ax.scatter(pos[0, :, 0], pos[0, :, 1], s=4, c="k")
    ax.scatter(pos[-1, :, 0], pos[-1, :, 1], s=4, c="tab:red")
    _draw_scene(ax, sc)
    ax.set_title(f"trajectories ({label0}); black=start red=end")
    for label, (logs, series) in runs.items():
        t = series["t"][:-1]
        axs[0, 1].plot(series["t"], series["arrived_frac"], label=label)
        axs[0, 2].plot(t, np.asarray(logs["p5_sep_pairs"]), label=f"{label} p5 NN")
        axs[0, 2].plot(t, np.asarray(logs["min_sep"]), lw=0.5, alpha=0.7, label=f"{label} min")
        axs[1, 0].plot(t, (np.asarray(logs["intervention"]) > INTERV_EPS).mean(1), lw=0.7, label=label)
        axs[1, 1].plot(t, (np.asarray(logs["slack_max"]) > 1e-4).mean(1), lw=0.7, label=label)
        r = series["r_goal_dist"][-1]
        axs[1, 2].hist(r, bins=np.linspace(0, max(8.0, r.max()), 40), histtype="step", label=label)
    axs[0, 1].axhline(0.9, color="gray", ls=":")
    axs[0, 1].set_title("arrived fraction (within r_goal)")
    axs[0, 2].axhline(D_COLLISION, color="red", ls=":")
    axs[0, 2].set_title("true separation [m]")
    axs[1, 0].set_title(f"CBF intervention rate (|u-u_nom|>{INTERV_EPS})")
    axs[1, 1].set_title("QP slack-active rate")
    axs[1, 2].axvline(sc.r_goal, color="tab:green", ls="--")
    axs[1, 2].set_title("final distance to goal [m]")
    for ax in list(axs.flat)[1:]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def goal_profile(path, profiles, r_goal):
    """profiles: {label: (r_edges, density)}."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=90)
    for label, (edges, dens) in profiles.items():
        ax.step(edges[:-1], dens, where="post", label=label)
    ax.axvline(r_goal, color="tab:green", ls="--", label="r_goal")
    ax.set_xlabel("distance to goal [m]")
    ax.set_ylabel("agents / m^2 (annulus, inside arena)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
