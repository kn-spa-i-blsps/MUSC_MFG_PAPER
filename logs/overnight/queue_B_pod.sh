#!/usr/bin/env bash
# Pod queue after the 2026-11-01 balance stop: amend28 -> 32 -> 29 -> 30 -> 31, strictly sequential.
# Each chain validates its volume outputs (skip-existing), checkpoints per N / per setting, and pushes right after its
# report, before the next one starts; its EXIT trap is the backup. A chain whose /workspace/<id>_DONE exists is skipped.
cd /workspace/MUSC_MFG_PAPER
log() { echo "$(date '+%F %T') $*"; }
run() { local id=$1; shift
  [ -f /workspace/${id}_DONE ] && { log "skip $id (DONE marker)"; return; }
  rm -f /workspace/${id}_FAILED; log "start $id"; bash "$@" > logs/overnight/${id}_chain.log 2>&1
  log "end $id: $(cat /workspace/${id}_DONE /workspace/${id}_FAILED 2>/dev/null)"; }
run amend28 logs/overnight/amend28_chain_pod.sh
run amend32 logs/overnight/amend32_chain_pod.sh
run amend29 logs/overnight/sweep_chain_pod.sh amend29
run amend30 logs/overnight/sweep_chain_pod.sh amend30
run amend31 logs/overnight/amend31_chain_pod.sh
log "queue end"
