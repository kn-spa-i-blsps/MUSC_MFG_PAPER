#!/usr/bin/env bash
# amend31 chain (Phase B steps 2-3), detached after amend30: sigma head + conformal (G-cal), identity gate (aborts on
# failure), 35 cells x 10 seeds 2 in parallel, report, commit -f, push, marker /workspace/amend31_DONE|FAILED.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
M=experiments.amend31_sigma_cbf; L=logs/amend31_pod; CD=experiments/amend31_cells; R=results/amend31_sigma_cbf
S=results/phaseB/sigma_v3; W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $S/sigma_weights.npz $S/calibration.json $R/report.json $R/identity.json $R/*/*/*/metrics_seed*.json $CD $L/*.log; do
    [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "amend31 result ($STATUS): conformal sigma-CBF vs fixed-margin frontier, G-cal + G-win, seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/amend31_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend31_sigma_cbf.md | cut -c1-64); v3 sha256 $(sha256sum $W | cut -c1-64)"
python -m experiments.phaseB_sigma --out $S > $L/sigma_train.log 2>&1 || { log "phaseB_sigma FAILED"; tail -5 $L/sigma_train.log; exit 1; }
log "sigma head: $(grep -E '^alpha|^G-cal' $L/sigma_train.log | tr '\n' ' ')"
python -m $M write-cells > $L/write_cells.log 2>&1 || { log "write-cells FAILED"; exit 1; }
y=$CD/identity/N100_sigma_a0.05_real_k0.yaml
python -m experiments.ladder --config $y --mode eval --arm real --weights $W --seeds 0 > $L/eval_identity.log 2>&1
log "identity rollout exit=$?"
python -m $M identity > $L/identity.log 2>&1 || { log "identity FAILED: $(cat $L/identity.log | tail -2)"; exit 1; }
log "identity PASS: $(tail -1 $L/identity.log)"
export L W
ls $CD/*.yaml | xargs -P 2 -L 1 bash -c '
  y=$0; c=$(basename $y .yaml)
  rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/learned/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0
  python -m experiments.ladder --config $y --mode eval --arm learned --weights $W --seeds $miss > $L/eval_$c.log 2>&1
  echo "$(date "+%F %T") $c exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
expected=$((10 * $(ls $CD/*.yaml | wc -l)))
n=$(ls $R/N*/*/learned/metrics_seed*.json | wc -l); log "rollouts: $n/$expected"
grep -h '"matmul_precision"' $R/N*/*/learned/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -60 $L/report.log
[ $rc = 0 ] && [ "$n" = "$expected" ] && STATUS=DONE
