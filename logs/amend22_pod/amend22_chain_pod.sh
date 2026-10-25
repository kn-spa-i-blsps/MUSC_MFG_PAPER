#!/usr/bin/env bash
# amend22 on RunPod A100 pod biiz4t7r00oef8, detached (tmux). N crossover with the pooled predictor on the amend21 fixed
# geometry, precision=highest: N in {25,50,100,150,200} x {real d0.55, real d0.70, learned_pooled d0.70} x seeds 0-9.
# Caches = amend21 keys (re-solved on this pod in overnight P1a, identity checked). One process per cell (seeds 0-9,
# skip existing), 3 cells in parallel -> report -> git add -f results + logs, commit, push -> DONE marker.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
W=$(python -c "from experiments.amend22_N_crossover_pooled import WEIGHTS; print(WEIGHTS)")
R=results/amend22_N_crossover_pooled
L=logs/amend22_pod
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  echo "$STATUS $(date '+%F %T') (pre-commit)" > $L/amend22_$STATUS
  for f in $R/report.json $R/N*/*/*/metrics_seed*.json $L/*.log $L/*.sh $L/amend22_*; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "amend22 result ($STATUS): N crossover with the pooled predictor, N in {25..200} x {real d0.55, real d0.70, learned_pooled d0.70}, seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/amend22_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend22_N_crossover_pooled.md | cut -c1-64)"
log "weights $W sha256 $(sha256sum $W | cut -c1-64)"
export W
for N in 200 150 100 50 25; do for arm in learned_pooled_d0.70 real_d0.70 real_d0.55; do echo "N$(printf %03d $N)_$arm"; done; done |
  xargs -P 3 -L 1 bash -c '
    cell=$0; rn=$(python -c "import yaml; print(yaml.safe_load(open(\"experiments/amend22_cells/$cell.yaml\"))[\"run_name\"])")
    case $cell in *learned*) arm=learned;; *) arm=real;; esac
    miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/$arm/metrics_seed$s.json ] || miss="$miss $s"; done
    [ -z "$miss" ] && { echo "$(date "+%F %T") skip $cell"; exit 0; }
    echo "$(date "+%F %T") start $cell seeds$miss"
    python -m experiments.ladder --config experiments/amend22_cells/$cell.yaml --mode eval --arm $arm --weights $W --seeds $miss > logs/amend22_pod/eval_$cell.log 2>&1
    echo "$(date "+%F %T") done $cell exit=$? $(grep -c "seed=" logs/amend22_pod/eval_$cell.log) seeds"'
n=$(ls $R/N*/*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/150"
grep -h '"matmul_precision"' $R/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m experiments.amend22_N_crossover_pooled report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -30 $L/report.log
[ $rc = 0 ] && [ "$n" = 150 ] && STATUS=DONE
log "chain end"
