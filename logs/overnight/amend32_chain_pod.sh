#!/usr/bin/env bash
# amend32 chain (cvls crossover baseline), detached after amend28 and before amend29/30 (whose cells read its margin).
# 10 cells x 10 seeds 2 in parallel, report, commit -f, push, marker /workspace/amend32_DONE|FAILED.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
. logs/overnight/push_lib.sh
M=experiments.amend32_cvls; L=logs/amend32_pod; CD=experiments/amend32_cells; R=results/amend32_cvls
W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $R/report.json $R/*/*/*/metrics_seed*.json $CD $L/*.log; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "amend32 result ($STATUS): window-matched classical baseline cvls on the crossover, seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/amend32_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"
log "volume check: $(validate_run_dirs $R | tail -1)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend32_cvls.md | cut -c1-64)"
python -m $M write-cells > $L/write_cells.log 2>&1 || { log "write-cells FAILED"; exit 1; }
export L W
for NN in 025 050 100 150 200; do
ls $CD/N${NN}_*.yaml | xargs -P 2 -L 1 bash -c '
  y=$0; c=$(basename $y .yaml)
  rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/cvls/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0
  python -m experiments.ladder --config $y --mode eval --arm cvls --weights $W --seeds $miss > $L/eval_$c.log 2>&1
  echo "$(date "+%F %T") $c exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
checkpoint "amend32 checkpoint N=$NN (partial rollouts, not a result)" $R/N$NN/*/*/metrics_seed*.json $CD $L/*.log
done
n=$(ls $R/N*/*/cvls/metrics_seed*.json | wc -l); log "rollouts: $n/100"
grep -h '"matmul_precision"' $R/N*/*/cvls/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -40 $L/report.log
checkpoint "amend32 report (exit $rc, $n/100 rollouts): pushed before the next amendment" $R/report.json $R/*/*/*/metrics_seed*.json $CD $L/*.log
[ $rc = 0 ] && [ "$n" = "100" ] && STATUS=DONE
