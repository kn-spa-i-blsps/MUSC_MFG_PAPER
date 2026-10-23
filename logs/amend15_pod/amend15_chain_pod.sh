#!/usr/bin/env bash
# amend15 on the RunPod GPU (one backend for every solve and rollout): per cell -> Phi/m* solve (skipped when the
# solve log already reports nan=false, i.e. s=1.05 from the timing step), then 4 arms x seeds 0-4, P_ROLL at a time;
# a rollout whose metrics json already exists (the timing rollout) is skipped. Then the report. Waits on PIDs only.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
CELLS=(1.05 0.90 0.75 0.60 0.45)
P_ROLL=2
log() { echo "$(date '+%F %T') $*"; }
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
for c in "${CELLS[@]}"; do
  if ! grep -q '"nan": false' logs/amend15_solve_s$c.log 2>/dev/null; then
    log "solve start s=$c"
    python -m experiments.amend15_density_sweep solve --cells $c > logs/amend15_solve_s$c.log 2>&1
    log "solve done s=$c exit=$? $(grep -o '"solve_seconds": [0-9.]*' logs/amend15_solve_s$c.log)"
    grep -q '"nan": false' logs/amend15_solve_s$c.log || { log "solve s=$c failed or NaN -> STOP"; exit 1; }
  else
    log "solve s=$c reused ($(grep -o '"solve_seconds": [0-9.]*' logs/amend15_solve_s$c.log))"
  fi
  export c W
  for s in 0 1 2 3 4; do for arm in real cv learned smith; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      m=results/density_sweep_amend15_N100/s$c/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $c $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $c $0 $1"
      python -m experiments.ladder --config experiments/amend15_cells/s$c.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > logs/amend15_eval_s${c}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $c $0 $1 exit=$?"'
  n=$(ls results/density_sweep_amend15_N100/s$c/{real,cv,learned,smith}/metrics_seed*.json 2>/dev/null | wc -l)
  log "cell $c rollouts done: $n/20 metrics"
done
total=$(ls results/density_sweep_amend15_N100/s*/{real,cv,learned,smith}/metrics_seed*.json 2>/dev/null | wc -l)
log "all cells done: $total/100 metrics"
[ "$total" = 100 ] || { log "report skipped"; exit 1; }
python -m experiments.amend15_density_sweep report > logs/amend15_report.log 2>&1; log "report exit=$?"
log "chain end"
