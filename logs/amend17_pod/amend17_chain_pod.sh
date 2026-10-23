#!/usr/bin/env bash
# amend17 on the RunPod GPU (pod k9nrewd0bhycsu), detached (tmux). Adaptive CBF margin k in {1.0, 1.5, 2.0} x
# {real, learned} x seeds 0-4 on the s=1.05 cache (NO solve; the cache was re-solved once with the identical key by
# logs/amend17_pod/amend17_presolve_sanity.sh before the aneks commit) -> report -> git add -f results + logs, commit,
# push -> DONE marker. On any exit (also failure) the EXIT trap commits and pushes whatever exists and writes DONE (ok)
# or FAILED. Waits on PIDs only (xargs), no pgrep.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
R=results/amend17_adaptive_margin_N100
L=logs/amend17_pod
P_ROLL=2
CACHE105=results/cache/macro_N100_cw1_f770143b36.npz
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }

finish() {
  log "finish status=$STATUS"
  for f in $R/report.json $R/sanity_k0.json $R/*/*/metrics_seed*.json $R/sanity/*/*/metrics_seed*.json \
           $L/*.log $L/*.json $L/*.sh; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "amend17 result ($STATUS): staleness-aware adaptive CBF margin v1, k in {1.0,1.5,2.0} x {real,learned} (RunPod GPU)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i), retry in 60 s"; sleep 60
  done
  git status -sb | head -1
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > logs/amend17_$STATUS
}
trap finish EXIT

rollouts() {   # $1 = cell, $2.. = arms; seeds 0-4; skip if metrics exist
  local cell=$1; shift
  export cell W L
  for s in 0 1 2 3 4; do for arm in "$@"; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      rn=$(python -c "import yaml,sys; print(yaml.safe_load(open(\"experiments/amend17_cells/$cell.yaml\"))[\"run_name\"])")
      m=results/$rn/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $cell $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $cell $0 $1"
      python -m experiments.ladder --config experiments/amend17_cells/$cell.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > $L/eval_${cell}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $cell $0 $1 exit=$? $(tail -1 $L/eval_${cell}_$0_s$1.log)"'
}

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend17_adaptive_margin.md | cut -c1-64) HEAD $(git rev-parse HEAD)"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
[ -f $CACHE105 ] || { log "s=1.05 cache $CACHE105 missing -> STOP (no re-solve in the run)"; exit 1; }
log "s=1.05 cache present: $CACHE105 sha256 $(sha256sum $CACHE105 | cut -c1-16)"

for k in 1.00 1.50 2.00; do rollouts k$k real learned; done
n=$(ls $R/k*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/30 metrics"
grep -h macro_cache $R/k*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "cache: $l"; done

python -m experiments.amend17_adaptive_margin report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"
[ $rc = 0 ] && [ "$n" = 30 ] && STATUS=DONE
log "chain end"
