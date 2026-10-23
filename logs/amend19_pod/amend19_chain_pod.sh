#!/usr/bin/env bash
# amend19 on the RunPod GPU (pod 8thhi6wqzzclov), detached (tmux). Closure-gated adaptive margin k in {0.5, 1.0, 2.0}
# x learned x seeds 0-4 (default XLA autotuning) on the s=1.05 cache re-solved by amend19_presolve_sanity.sh (NO solve
# here) -> report -> git add -f results + logs, commit, push -> DONE marker. The pod is left running (PI order).
# On any exit the EXIT trap commits and pushes whatever exists and writes DONE (ok) or FAILED. Waits on PIDs only (xargs), no pgrep.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
R=results/amend19_closing_margin_N100
L=logs/amend19_pod
P_ROLL=2
CACHE105=results/cache/macro_N100_cw1_f770143b36.npz
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }

finish() {
  log "finish status=$STATUS"
  echo "$STATUS $(date '+%F %T') (pre-commit)" > $L/amend19_$STATUS
  cp logs/amend19_chain_pod.log $L/ 2>/dev/null
  for f in $R/report.json $R/k*/*/metrics_seed*.json $L/*.log $L/*.sh $L/amend19_* $R/sanity_k0.json $R/sanity/*/*/metrics_seed0.json $R/cache_identity/*/*/metrics_seed0.json $L/*.json; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "amend19 result ($STATUS): closure-gated adaptive CBF margin v2a on learned, k in {0.5,1.0,2.0} @ s=1.05 (RunPod GPU)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i), retry in 60 s"; sleep 60
  done
  git status -sb | head -1
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > logs/amend19_$STATUS
}
trap finish EXIT

rollouts() {   # $1 = cell, $2.. = arms; seeds 0-4; skip if metrics exist
  local cell=$1; shift
  export cell W L
  for s in 0 1 2 3 4; do for arm in "$@"; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      rn=$(python -c "import yaml,sys; print(yaml.safe_load(open(\"experiments/amend19_cells/$cell.yaml\"))[\"run_name\"])")
      m=results/$rn/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $cell $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $cell $0 $1"
      python -m experiments.ladder --config experiments/amend19_cells/$cell.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > $L/eval_${cell}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $cell $0 $1 exit=$? $(tail -1 $L/eval_${cell}_$0_s$1.log)"'
}

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend19_closing_margin.md | cut -c1-64) HEAD $(git rev-parse HEAD)"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
[ -f $CACHE105 ] || { log "s=1.05 cache $CACHE105 missing -> STOP (no re-solve in the run)"; exit 1; }
log "s=1.05 cache present: $CACHE105 sha256 $(sha256sum $CACHE105 | cut -c1-16)"

for k in 0.50 1.00 2.00; do rollouts k$k learned; done
n=$(ls $R/k*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/15 metrics"
grep -h macro_cache $R/k*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "cache: $l"; done

python -m experiments.amend19_closing_margin report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"
[ $rc = 0 ] && [ "$n" = 15 ] && STATUS=DONE
log "chain end"
