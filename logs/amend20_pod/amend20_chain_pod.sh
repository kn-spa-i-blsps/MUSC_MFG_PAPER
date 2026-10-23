#!/usr/bin/env bash
# amend20 on the RunPod GPU (pod 8thhi6wqzzclov), detached (tmux). matmul precision 'highest' (cell key, applied in
# experiments/ladder.py, logged per run). closing k in {0.25,0.50,0.75} + learned fixed {0.55,0.70} + real fixed 0.55,
# seeds 0-9, s=1.05 cache reused (NO solve) -> report -> git add -f results + logs, commit, push -> DONE. Pod left running.
# On any exit the EXIT trap commits and pushes whatever exists and writes DONE (ok) or FAILED. Waits on PIDs only (xargs), no pgrep.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
R=results/amend20_closing_lockin_N100
L=logs/amend20_pod
P_ROLL=2
CACHE105=results/cache/macro_N100_cw1_f770143b36.npz
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }

finish() {
  log "finish status=$STATUS"
  echo "$STATUS $(date '+%F %T') (pre-commit)" > $L/amend20_$STATUS
  cp logs/amend20_chain_pod.log $L/ 2>/dev/null
  for f in $R/report.json $R/*/*/metrics_seed*.json $L/*.log $L/*.sh $L/amend20_* $L/*.json; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "amend20 result ($STATUS): closure-gated margin lock-in under precision=highest, k in {0.25,0.50,0.75} + fixed 0.55/0.70 (+ real 0.55), seeds 0-9 (RunPod GPU)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i), retry in 60 s"; sleep 60
  done
  git status -sb | head -1
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > logs/amend20_$STATUS
}
trap finish EXIT

rollouts() {   # $1 = cell, $2.. = arms; seeds 0-9; skip if metrics exist
  local cell=$1; shift
  export cell W L
  for s in 0 1 2 3 4 5 6 7 8 9; do for arm in "$@"; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      rn=$(python -c "import yaml,sys; print(yaml.safe_load(open(\"experiments/amend20_cells/$cell.yaml\"))[\"run_name\"])")
      m=results/$rn/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $cell $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $cell $0 $1"
      python -m experiments.ladder --config experiments/amend20_cells/$cell.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > $L/eval_${cell}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $cell $0 $1 exit=$? $(tail -1 $L/eval_${cell}_$0_s$1.log)"'
}

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend20_closing_lockin.md | cut -c1-64) HEAD $(git rev-parse HEAD)"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
[ -f $CACHE105 ] || { log "s=1.05 cache $CACHE105 missing -> STOP (no re-solve in the run)"; exit 1; }
log "s=1.05 cache present: $CACHE105 sha256 $(sha256sum $CACHE105 | cut -c1-16)"

for k in 0.25 0.50 0.75; do rollouts closing_k$k learned; done
for d in 0.55 0.70; do rollouts fixed$d learned; done
rollouts real_fixed0.55 real
n=$(ls $R/*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/60 metrics"
grep -h matmul_precision $R/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
grep -h macro_cache $R/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "cache: $l"; done

python -m experiments.amend20_closing_lockin report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"
[ $rc = 0 ] && [ "$n" = 60 ] && STATUS=DONE
log "chain end"
