import numpy as np, yaml, json, jax.numpy as jnp
from experiments.smoke import build_scenario
from control.law import sample_field_bilinear
from experiments.diag_goal_ceiling import ARMS, CACHE
cfg=yaml.safe_load(open('experiments/ladder_fargoal_amend11.yaml')); sc=build_scenario(cfg,cfg['N'])
z=np.load(CACHE); pg=z['phi_grad']; dt=float(z['dt']); step=float(z['step'])
g=np.array(sc.goal)
times=[20,30,40,50,55,60,63,65,67,68,69,70,70.3]
out={}
for arm in ['fresh','noise_only','A_cbf_fresh']:
  for s in range(5):
    L=np.load(f"{ARMS[arm]}/logs_seed{s}.npz"); pos=L['pos'].astype(np.float64); un=L['u_nom']; u=L['u']
    row=[]
    for t in times:
      k=min(int(round(t/0.01)),len(pos)-1); p=pos[k]
      r=np.linalg.norm(p-g,axis=-1); rad=(p-g)/np.maximum(r[:,None],1e-9)
      rt=np.asarray(sample_field_bilinear(jnp.asarray(pg[min(int(round(t/dt)),len(pg)-1)]),jnp.asarray(p),step))
      shell=(r>sc.r_goal-1)&(r<sc.r_goal+1)
      row.append(dict(t=t,n_out=int((r>sc.r_goal).sum()),
        route_norm_shell=float(np.median(np.linalg.norm(rt[shell],axis=-1))),
        route_rad_shell=float(np.median((rt[shell]*rad[shell]).sum(-1))),
        unom_rad_shell=float(np.median((un[k][shell]*rad[shell]).sum(-1))),
        u_rad_shell=float(np.median((u[k][shell]*rad[shell]).sum(-1)))))
    out[f"{arm}_s{s}"]=row
    print(arm,s,' '.join(f"{d['t']}:{d['n_out']}|{d['route_norm_shell']:.2f}|{d['route_rad_shell']:+.2f}|{d['unom_rad_shell']:+.2f}|{d['u_rad_shell']:+.2f}" for d in row))
json.dump(out,open('results/diag_goal_ceiling/timeseries.json','w'),indent=1)
