#!/usr/bin/env bash
# Generic overnight amendment chain (RunPod A100 biiz4t7r00oef8, detached): $1 = amendNN, $2 = python module,
# $3 = cells dir, $4 = results dir, $5 = expected metrics count. One process per cell (seeds 0-9, skip existing), 2 cells in
# parallel (3 OOM the 80 GB GPU), arm from the cell name (learned / noise_only / fresh / real) -> report ->
# git add -f results + logs, commit, push -> DONE/FAILED marker in /workspace.
A=$1; MOD=$2; CD=$3; R=$4; NEXP=$5
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
W=results/overnight_p1/predictor_v3/predictor_weights.npz
L=logs/${A}_pod
mkdir -p $L
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  echo "$STATUS $(date '+%F %T') (pre-commit)" > $L/${A}_$STATUS
  for f in $R/report.json $R/N*/*/*/metrics_seed*.json $L/*.log $L/${A}_*; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "$A result ($STATUS): $(head -1 prereg/prereg_projectA_${A}_*.md | sed 's/^# //'), seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/${A}_$STATUS
}
trap finish EXIT
log "chain start $A gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_${A}_*.md | cut -c1-64)"
log "weights $W sha256 $(sha256sum $W | cut -c1-64)"
export W L
ls $CD/*.yaml | xargs -n1 basename | sed 's/.yaml$//' | xargs -P 2 -L 1 bash -c '
  cell=$0; cd_=$(dirname $(ls '"$CD"'/$cell.yaml)); rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$cd_/$cell.yaml\"))[\"run_name\"])")
  case $cell in *learned*) arm=learned;; *noise_only*) arm=noise_only;; *fresh*) arm=fresh;; *) arm=real;; esac
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/$arm/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && { echo "$(date "+%F %T") skip $cell"; exit 0; }
  echo "$(date "+%F %T") start $cell ($arm) seeds$miss"
  python -m experiments.ladder --config $cd_/$cell.yaml --mode eval --arm $arm --weights $W --seeds $miss > $L/eval_$cell.log 2>&1
  echo "$(date "+%F %T") done $cell exit=$? $(grep -c "seed=" $L/eval_$cell.log) seeds"'
n=$(ls $R/N*/*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/$NEXP"
grep -h '"matmul_precision"' $R/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m $MOD report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -40 $L/report.log
[ $rc = 0 ] && [ "$n" = "$NEXP" ] && STATUS=DONE
log "chain end"
