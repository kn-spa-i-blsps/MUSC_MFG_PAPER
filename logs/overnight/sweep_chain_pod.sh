#!/usr/bin/env bash
# amend29 / amend30 chain ($1 = amend29 | amend30), detached after the previous amendment. Cells are generated on the
# pod from the amend27 report (pre-registered margin rule). Per setting: (amend30) solve the setting's caches first,
# then its rollouts 2 in parallel, then (amend30) delete that layout's caches. Report, commit -f, push, marker.
EXP=$1
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
M="experiments.amend29_physical_limits --exp $EXP"; L=logs/${EXP}_pod; CD=experiments/${EXP}_cells
R=results/$( [ $EXP = amend29 ] && echo amend29_physical_limits || echo amend30_layouts )
W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $R/report.json $R/*/N*/*/*/metrics_seed*.json $CD $L/*.log; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "$EXP result ($STATUS): $(head -1 prereg/prereg_projectA_${EXP}_*.md | sed 's/^# //'), seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/${EXP}_$STATUS
}
trap finish EXIT
log "chain start $EXP gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD) disk $(du -sh /workspace | cut -f1)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_${EXP}_*.md | cut -c1-64)"
python -m $M write-cells > $L/write_cells.log 2>&1 || { log "write-cells FAILED (amend27 report missing?)"; tail -3 $L/write_cells.log; exit 1; }
log "margins $(cat $CD/margins_from_amend27.json | tr -d '\n ')"
export L W
expected=0
for sdir in $CD/*/; do
  setting=$(basename $sdir); before=$(ls results/cache/*.npz | sort)
  if [ $EXP = amend30 ]; then
    for y in $sdir/N100_fresh.yaml $sdir/N150_fresh.yaml $sdir/N100_fresh_nomfg.yaml; do
      python -m $M solve --cell $y > $L/solve_${setting}_$(basename $y .yaml).log 2>&1
      log "solve $setting $(basename $y) exit=$? $(tail -n 1 $L/solve_${setting}_$(basename $y .yaml).log | cut -c1-160)"
    done
  fi
  ls $sdir*.yaml | xargs -P 2 -L 1 bash -c '
    y=$0; c=$(basename $(dirname $y))_$(basename $y .yaml)
    arm=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"chain_arm\"])")
    rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
    miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/$arm/metrics_seed$s.json ] || miss="$miss $s"; done
    [ -z "$miss" ] && exit 0
    python -m experiments.ladder --config $y --mode eval --arm $arm --weights $W --seeds $miss > $L/eval_$c.log 2>&1
    echo "$(date "+%F %T") $c ($arm) exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
  expected=$((expected + 10 * $(ls $sdir*.yaml | wc -l)))
  if [ $EXP = amend30 ]; then
    for f in $(comm -13 <(echo "$before") <(ls results/cache/*.npz | sort)); do rm -f $f; log "deleted layout cache $f"; done
  fi
  log "setting $setting done; disk $(du -sh /workspace | cut -f1)"
done
n=$(ls $R/*/N*/*/*/metrics_seed*.json | wc -l); log "rollouts: $n/$expected"
grep -h '"matmul_precision"' $R/*/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -80 $L/report.log
[ $rc = 0 ] && [ "$n" = "$expected" ] && STATUS=DONE
