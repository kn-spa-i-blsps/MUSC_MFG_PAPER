#!/usr/bin/env bash
# Overnight Phases 3-4 on the pod, queued after the night chain (amend24): amend25 per-gate Phi/m* solves (gate 2.309 =
# existing N=100 cache) -> amend25 rollouts + open-loop check + report -> amend26 rollouts + open-loop check + report.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
log() { echo "$(date '+%F %T') $*"; }
until [ -e /workspace/amend24_DONE ] || [ -e /workspace/amend24_FAILED ]; do sleep 60; done
log "amend24 finished ($(ls /workspace/amend24_* | xargs -n1 basename)); start phase 3-4 HEAD $(git rev-parse --short HEAD)"
git pull -q --rebase --autostash origin main; log "HEAD after pull $(git rev-parse --short HEAD)"
mkdir -p logs/amend25_pod logs/amend26_pod
for g in 1.5 3.5 5.0 7.0; do
  log "solve gate $g"
  python -m experiments.amend25_gate_sweep solve --gate $g > logs/amend25_pod/solve_g$g.log 2>&1
  log "solve gate $g exit=$? $(tail -1 logs/amend25_pod/solve_g$g.log)"
done
PRE_REPORT="python -m experiments.amend25_gate_sweep openloop" bash logs/overnight/run_amend_pod.sh amend25 experiments.amend25_gate_sweep experiments/amend25_cells results/amend25_gate_sweep 150 > logs/amend25_pod/amend25_chain_pod.log 2>&1
log "amend25 chain exit $(cat /workspace/amend25_DONE /workspace/amend25_FAILED 2>/dev/null)"
PRE_REPORT="python -m experiments.amend26_tslot_sweep openloop" bash logs/overnight/run_amend_pod.sh amend26 experiments.amend26_tslot_sweep experiments/amend26_cells results/amend26_tslot_sweep 120 > logs/amend26_pod/amend26_chain_pod.log 2>&1
log "amend26 chain exit $(cat /workspace/amend26_DONE /workspace/amend26_FAILED 2>/dev/null)"
