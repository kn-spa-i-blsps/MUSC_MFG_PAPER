#!/usr/bin/env bash
# Overnight predictor v4 (EXPLORATORY, review_amend24 fix 1), queued after amend26 on the pod: datagen under the
# closed-loop learned-v3 d_safe 0.70 arm (seeds 100-111, each N; = the evaluation state distribution, DAgger-style),
# train v4a (real+learned sources, near 0.5, h128) and v4b (same, h256, 200k windows/seed/N), open-loop gate on real AND
# learned test histories (seeds 110-111) incl. noise_only and tails. Self-commit/push.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
L=logs/overnight/p1; R=results/overnight_p1; W3=$R/predictor_v3/predictor_weights.npz
log() { echo "$(date '+%F %T') $*"; }
[ -n "$NOWAIT" ] || until [ -e /workspace/amend26_DONE ] || [ -e /workspace/amend26_FAILED ]; do sleep 60; done
git pull -q --rebase --autostash origin main; log "v4 start HEAD $(git rev-parse --short HEAD)"
python -m experiments.overnight_p1_predictor write-cells > /dev/null
export L R W3
for N in 25 50 100 150 200; do echo $N; done | xargs -P 2 -L 1 bash -c '
  NN=$(printf "%03d" $0)
  miss=""; for s in 100 101 102 103 104 105 106 107 108 109 110 111; do [ -f $R/datagenL_N$NN/datagen/pos_seed$s.npz ] || miss="$miss $s"; done
  [ -n "$miss" ] && python -m experiments.ladder --config experiments/overnight_p1_cells/datagenL_N$NN.yaml --mode datagen --datagen_arm learned --weights $W3 --seeds $miss > $L/datagenL_N$NN.log 2>&1
  echo "$(date "+%F %T") datagenL N=$0 exit=$? $(tail -1 $L/datagenL_N$NN.log)"'
log "datagenL files $(ls $R/datagenL_N*/datagen/pos_seed*.npz | wc -l)/60"
for v in "v4a --near_frac 0.5 --hidden 128 --sources datagen datagenL" "v4b --near_frac 0.5 --hidden 256 --samples 200000 --sources datagen datagenL"; do
  set -- $v; t=$1; shift
  python -m experiments.overnight_p1_predictor train --tag $t "$@" > $L/train_$t.log 2>&1; log "train $t exit=$? sha=$(python -c "import json; print(json.load(open('$R/predictor_$t/predictor_report.json'))['sha256'])" 2>/dev/null)"
done
for src in datagen datagenL; do
  for t in v3 v4a v4b; do
    python -m experiments.overnight_p1_predictor openloop --weights $R/predictor_$t/predictor_weights.npz --src $src --out $R/openloop2_${t}_$src.json > $L/openloop2_${t}_$src.log 2>&1
    log "openloop $t on $src exit=$?"
  done
done
for f in $L/*.log $R/predictor_v4*/predictor_report.json $R/predictor_v4*/predictor_weights.npz $R/openloop2_*.json; do [ -e "$f" ] && git add -f "$f"; done
git commit -q -m "overnight P1 exploratory: learned-v3 closed-loop datagen, predictor v4a/v4b, open-loop gate on real + learned histories (tails, noise_only)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3 4 5; do git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }; sleep 60; done
echo DONE > /workspace/v4_DONE
