#!/usr/bin/env bash
# amend27 (method ladder + MFG/CBF layer split), detached on the pod after the G-pred chain. Identity gate first (aborts
# on mismatch), then all cells (2 in parallel, arm from the cell's amend27_arm key), report, commit -f, push, marker.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
A=amend27; L=logs/amend27_pod; R=results/amend27_ladder_layers; CD=experiments/amend27_cells
W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $R/report.json $R/identity.json $R/N*/*/*/metrics_seed*.json $L/*.log; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "amend27 result ($STATUS): compensation-method ladder (cv/smith x 0.55/0.70, fresh) + MFG/CBF layer split, seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/${A}_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend27_*.md | cut -c1-64); weights sha256 $(sha256sum $W | cut -c1-64)"
ev() {  # $1 = cell yaml, $2.. = seeds
  y=$1; shift; arm=$(python -c "import yaml; print(yaml.safe_load(open('$y'))['amend27_arm'])")
  python -m experiments.ladder --config $y --mode eval --arm "$arm" --weights $W --seeds "$@" > $L/eval_$(basename $y .yaml).log 2>&1
  echo "$(date '+%F %T') $(basename $y) ($arm) exit=$? $(grep -c 'seed=' $L/eval_$(basename $y .yaml).log) seeds"
}
export -f ev; export L W
for y in $CD/identity/*.yaml; do ev $y 0; done
python -m experiments.amend27_ladder_layers identity > $L/identity.log 2>&1 || { log "IDENTITY FAIL: $(cat $L/identity.log | tail -3)"; exit 1; }
log "identity PASS: $(tail -1 $L/identity.log)"
ls $CD/*.yaml | xargs -P 2 -L 1 bash -c '
  rn=$(python -c "import yaml; c=yaml.safe_load(open(\"$0\")); print(c[\"run_name\"] + \"/\" + c[\"amend27_arm\"].replace(\":\",\"_\").replace(\"/\",\"-\"))")
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0; ev $0 $miss'
n=$(ls $R/N*/*/*/metrics_seed*.json | grep -v identity | wc -l); log "rollouts: $n (expected 352 incl. the 2 identity seeds)"
grep -h '"matmul_precision"' $R/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m experiments.amend27_ladder_layers report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -60 $L/report.log
[ $rc = 0 ] && STATUS=DONE
