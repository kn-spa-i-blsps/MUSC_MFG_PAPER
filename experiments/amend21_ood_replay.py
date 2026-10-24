"""amend21 post-hoc OOD diagnostic (PI-ordered stop of the learned arms, 2026-10-24): open-loop replay of perceive() for
real / cv / learned on a REAL-arm closed-loop position history; mean squared x_hat error over neighbours closer than 2 m.

    python -m experiments.amend21_ood_replay <logs_seedX.npz> <N>
"""
import sys, numpy as np, jax, jax.numpy as jnp
from compensation.perception import perceive
from experiments.ladder import load_predictor
P = load_predictor("results/ladder_fargoal_N100/predictor_weights.npz")
path, N = sys.argv[1], int(sys.argv[2]); dt = 0.01
hist = jnp.asarray(np.load(path)["pos"], jnp.float32)
T = hist.shape[0]; ks = [200, 500, 1000, 2000, 4000, 6000]
ks = [k for k in ks if k < T]
out = {c: [] for c in ["real", "cv", "learned"]}
for k in ks:
    for c in out:
        xh = perceive(c, hist, k, dt, N, jax.random.PRNGKey(k), predictor_params=P if c == "learned" else None)
        d = jnp.linalg.norm(hist[k][:, None] - hist[k][None], axis=-1)
        near = (d < 2.0) & (d > 0)
        e = jnp.sum((xh - hist[k][None]) ** 2, -1)
        out[c].append(float(jnp.sum(jnp.where(near, e, 0)) / jnp.maximum(jnp.sum(near), 1)))
print(f"{path} N={N}: open-loop mse on neighbours <2 m at k={ks}:")
for c, v in out.items(): print(f"   {c:8s}", [round(x, 4) for x in v])
