"""Log-only: non-arrivers in the last seconds before T and |grad Phi| at the rim vs time-to-T (no rollouts).
    python -m experiments.diag_goal_ceiling_last > results/diag_goal_ceiling/last_seconds.txt"""
import numpy as np, yaml
from experiments.smoke import build_scenario
from experiments.diag_goal_ceiling import ARMS, CACHE
cfg = yaml.safe_load(open('experiments/ladder_fargoal_amend11.yaml')); sc = build_scenario(cfg, cfg['N']); g = np.array(sc.goal)
z = np.load(CACHE); pg = z['phi_grad']; dt = float(z['dt'])
x = np.arange(pg.shape[1]) * 0.2; zz = np.arange(pg.shape[2]) * 0.2
X, Z = np.meshgrid(x, zz, indexing='ij'); R = np.hypot(X - g[0], Z - g[1]); m = (R > sc.r_goal - 1) & (R < sc.r_goal + 1)
T = len(pg) - 1
print('median |gradPhi| in rim band (r_goal +-1 m) at T-s:',
      [(s, round(float(np.median(np.linalg.norm(pg[T - int(s / dt)][m], axis=-1))), 3)) for s in (10, 5, 2, 1, 0.5, 0.2, 0.1, 0)])
print('r_goal', sc.r_goal, 'T', T * dt)
for arm in ['fresh', 'noise_only', 'A_cbf_fresh']:
    for s in range(5):
        L = np.load(f"{ARMS[arm]}/logs_seed{s}.npz"); pos = np.concatenate([L['pos'], L['pos_final'][None]]).astype(float)
        r = np.linalg.norm(pos - g, axis=-1); n = (r > sc.r_goal).sum(1); K = len(n) - 1
        print(arm, s, 'out at T-10/5/2/1/0.5/0.2/0:', [int(n[K - int(t * 100)]) for t in (10, 5, 2, 1, 0.5, 0.2, 0)],
              'min last5s', int(n[-500:].min()), 'max last5s', int(n[-500:].max()),
              'out at T beyond r+0.25 / r+0.5:', int((r[-1] > sc.r_goal + 0.25).sum()), int((r[-1] > sc.r_goal + 0.5).sum()))
