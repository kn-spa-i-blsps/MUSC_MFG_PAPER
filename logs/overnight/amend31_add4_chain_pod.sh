#!/usr/bin/env bash
# amend31 addendum 4 chain (SUPPLEMENTARY). Waits for the registered amend31 run + report to finish and for the (blocked)
# queue to end; never touches them. Then: q_raw (CAL only) -> freeze -> cells -> gates -> priorities A, B, C under the
# 6 h / 15 USD step-3 limit -> analysis -> animations. Every stage pushed (push_lib checkpoint). Resumable (skip-existing).
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
. logs/overnight/push_lib.sh
M=experiments.amend31_add4; RUNNER=experiments.amend31_add4_run; R=results/amend31_add4; CD=experiments/amend31_add4_cells
L=logs/amend31_add4_pod; W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L
log() { echo "$(date '+%F %T') $*"; }
log "add4 chain start, waiting for amend31 DONE/FAILED and queue end"
until [ -f /workspace/amend31_DONE ] || [ -f /workspace/amend31_FAILED ]; do sleep 60; done
while tmux has-session -t queueB 2>/dev/null; do sleep 30; done
log "amend31 marker: $(cat /workspace/amend31_DONE /workspace/amend31_FAILED 2>/dev/null); queue: $(tail -1 logs/overnight/queue_B.log)"
if [ -f /workspace/amend31_FAILED ] || [ ! -f results/amend31_sigma_cbf/report.json ]; then
  log "amend31 did not complete -> steps 2-4 not started (addendum 4 step 0d)"; checkpoint "amend31 add4: not started (amend31 incomplete)" $L/*.log
  echo "NOT_STARTED $(date '+%F %T')" > /workspace/amend31_add4_DONE; exit 0
fi
log "HEAD $(git rev-parse HEAD) addendum4 sha256 $(sha256sum prereg/prereg_projectA_amend31_addendum4.md | cut -c1-64)"
# ---- step 2
JAX_PLATFORMS=cpu python -m $M qraw > $L/qraw.log 2>&1 || { log "qraw FAILED"; tail -3 $L/qraw.log; exit 1; }
python -m $M freeze > $L/freeze.log 2>&1 || { log "freeze FAILED"; tail -3 $L/freeze.log; exit 1; }
python -m $M write-cells > $L/write_cells.log 2>&1 || { log "write-cells FAILED"; exit 1; }
log "frozen: $(python -c "import json; d=json.load(open('$R/frozen.json'))['per_N']; print({N: (v['best_fixed'], {a: (round(x['d_safe'],4), x['alias']) for a, x in v['arms'].items()}) for N, v in d.items()})")"
log "jobs: $(tail -1 $L/write_cells.log)"
checkpoint "amend31 add4 step 2: q_raw (CAL), frozen comparator + fixed_match, cells" $R/q_raw.json $R/frozen.json $CD $L/*.log
# ---- gates
run1() { python -m $RUNNER --config $1 --mode eval --arm $2 --weights $W --seeds 0 > $L/gate_$(basename $1 .yaml).log 2>&1; }
run1 $CD/gates/g1.yaml real; run1 $CD/gates/g2.yaml learned; run1 $CD/gates/g3.yaml learned
python -m $M gate > $L/gate.log 2>&1; log "gates: $(tail -1 $L/gate.log | cut -c1-300)"
G1=$(python -c "import json; print(int(json.load(open('$R/gates.json'))['g1']['pass']))")
G23=$(python -c "import json; g=json.load(open('$R/gates.json')); print(int(g['g2']['pass'] and g['g3']['pass']))")
checkpoint "amend31 add4 gates: g1=$G1 g2&g3=$G23" $R/gates.json $R/gates/*/*/metrics_seed0.json $L/*.log
# ---- step 3, priorities under the limit (6 h or 15 USD of pod time at 1.79 USD/h from step-3 start, whichever first)
T0=$(date +%s); LIMIT_S=21600; LIMIT_USD=15; RATE=1.79
declare -A EST=([A]=1200 [B]=2400 [C]=6600)   # conservative wall estimates [s], logged before start
log "step 3 start; estimated wall A ${EST[A]}s B ${EST[B]}s C ${EST[C]}s = $(( (EST[A]+EST[B]+EST[C]) / 60 )) min ~ $(python -c "print(round(($((EST[A]+EST[B]+EST[C])))/3600*$RATE,2))") USD"
export W L RUNNER
for P in A B C; do
  if [ "$G1" != 1 ]; then log "priority $P skipped: gate g1 failed"; continue; fi
  if [ $P = C ] && [ "$G23" != 1 ]; then log "priority C skipped: gate g2/g3 failed"; continue; fi
  el=$(( $(date +%s) - T0 ))
  usd=$(python -c "print(($el + ${EST[$P]})/3600*$RATE)")
  if [ $(( el + EST[$P] )) -gt $LIMIT_S ] || python -c "import sys; sys.exit(0 if $usd > $LIMIT_USD else 1)"; then
    log "priority $P NOT started: limit at risk (elapsed ${el}s + est ${EST[$P]}s)"; continue; fi
  log "priority $P start (elapsed ${el}s)"
  grep "^$P " $CD/jobs.txt | cut -d' ' -f2- | xargs -P 2 -L 1 bash -c '
    y=$0; shift 0; seeds="$*"; c=$(basename $(dirname $y))_$(basename $y .yaml)
    rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
    miss=""; for s in $seeds; do [ -f results/$rn/learned/metrics_seed$s.json ] || miss="$miss $s"; done
    [ -z "$miss" ] && exit 0
    python -m $RUNNER --config $y --mode eval --arm learned --weights $W --seeds $miss > $L/eval_$c.log 2>&1
    echo "$(date "+%F %T") $c exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
  log "priority $P done (elapsed $(( $(date +%s) - T0 ))s); files $(find $R/main $R/rep $R/chaos -name 'metrics_seed*.json' 2>/dev/null | wc -l)"
  checkpoint "amend31 add4 priority $P rollouts (supplementary)" $(find $R/main $R/rep $R/chaos -name 'metrics_seed*.json' 2>/dev/null) $L/*.log
done
# ---- step 4
JAX_PLATFORMS=cpu python -m $M analyze > $L/analyze.log 2>&1; log "analyze exit=$? $(tail -1 $L/analyze.log | cut -c1-300)"
JAX_PLATFORMS=cpu python -m $M animate > $L/animate.log 2>&1; log "animate exit=$? $(grep -c frames $L/animate.log) gifs"
checkpoint "amend31 add4 step 4: analysis.json + animations (supplementary)" $R/analysis.json $R/anim $L/*.log
echo "DONE $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/amend31_add4_DONE
log "add4 chain end"
