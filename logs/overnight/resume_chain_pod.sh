#!/usr/bin/env bash
# Resume after the RunPod balance stop (2026-10-25 ~22:20 UTC): pod biiz4t7r00oef8 could not restart (host GPUs taken),
# its volume holds the unreported amend25 rollouts -> NEW pod xby8jnv4zqwvn2 (A100-SXM4-80GB, 250 GB volume), everything
# re-run from scratch under the unchanged pre-registrations: amend25 gate caches (5 solves) -> amend25 -> amend26 ->
# N=25/50/150/200 caches + real-arm datagen 100-111 (deterministic, regenerated) -> predictor v4 chain.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
log() { echo "$(date '+%F %T') $*"; }
log "resume chain start HEAD $(git rev-parse --short HEAD) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p logs/amend25_pod logs/amend26_pod logs/overnight/p1
for g in 2.309401076758503 1.5 3.5 5.0 7.0; do
  python -m experiments.amend25_gate_sweep solve --gate $g > logs/amend25_pod/solve_g$g.log 2>&1
  log "solve gate $g exit=$? $(tail -1 logs/amend25_pod/solve_g$g.log)"
done
PRE_REPORT="python -m experiments.amend25_gate_sweep openloop" bash logs/overnight/run_amend_pod.sh amend25 experiments.amend25_gate_sweep experiments/amend25_cells results/amend25_gate_sweep 150 > logs/amend25_pod/amend25_chain_pod.log 2>&1
log "amend25 chain exit $(cat /workspace/amend25_DONE /workspace/amend25_FAILED 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
PRE_REPORT="python -m experiments.amend26_tslot_sweep openloop" bash logs/overnight/run_amend_pod.sh amend26 experiments.amend26_tslot_sweep experiments/amend26_cells results/amend26_tslot_sweep 120 > logs/amend26_pod/amend26_chain_pod.log 2>&1
log "amend26 chain exit $(cat /workspace/amend26_DONE /workspace/amend26_FAILED 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
for N in 25 50 150 200; do
  NN=$(printf "%03d" $N)
  python -m experiments.amend21_N_crossover solve --N $N > logs/overnight/p1/resolve_N$NN.log 2>&1
  log "re-solve N=$N exit=$? $(tail -1 logs/overnight/p1/resolve_N$NN.log)"
done
python -m experiments.overnight_p1_predictor write-cells > /dev/null
export R=results/overnight_p1 L=logs/overnight/p1
for N in 25 50 100 150 200; do echo $N; done | xargs -P 2 -L 1 bash -c '
  NN=$(printf "%03d" $0)
  miss=""; for s in 100 101 102 103 104 105 106 107 108 109 110 111; do [ -f $R/datagen_N$NN/datagen/pos_seed$s.npz ] || miss="$miss $s"; done
  [ -n "$miss" ] && python -m experiments.ladder --config experiments/overnight_p1_cells/datagen_N$NN.yaml --mode datagen --seeds $miss > $L/redatagen_N$NN.log 2>&1
  echo "$(date "+%F %T") real datagen N=$0 exit=$?"'
log "real datagen files $(ls results/overnight_p1/datagen_N*/datagen/pos_seed*.npz | wc -l)/60"
python -m experiments.overnight_p1_predictor openloop --weights results/overnight_p1/predictor_v3/predictor_weights.npz --out results/overnight_p1/openloop_v3_regen.json > logs/overnight/p1/openloop_v3_regen.log 2>&1
log "v3 open-loop on regenerated datagen exit=$? (must equal openloop_v3.json): $(python -c "
import json; a=json.load(open('results/overnight_p1/openloop_v3.json'))['per_N']; b=json.load(open('results/overnight_p1/openloop_v3_regen.json'))['per_N']
print(all(a[N][k]['mse_near1m_m2']==b[N][k]['mse_near1m_m2'] for N in a for k in ('real','cv','learned_new')))")"
NOWAIT=1 bash logs/overnight/p1/v4_chain_pod.sh > logs/overnight/p1/v4_chain_pod.log 2>&1
log "v4 chain exit $(cat /workspace/v4_DONE 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
