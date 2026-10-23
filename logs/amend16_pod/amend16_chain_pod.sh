#!/usr/bin/env bash
# amend16 on the RunPod GPU, detached (tmux). Part 1 (real_robust d_safe sweep, s=1.05 cache reused, NO solve) ->
# Part 3 (3 viz rollouts with log_xhat, CBF wiring check, GIFs) -> Part 2 (5 gate solves + real/learned rollouts) ->
# report -> git add -f results + logs, commit, push -> DONE marker. On any exit (also failure) the EXIT trap commits and
# pushes whatever exists and writes DONE (ok) or FAILED. Waits on PIDs only (xargs), no pgrep.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
R=results/amend16_gate_diagnostic_N100
L=logs/amend16_pod
P_ROLL=2
CACHE105=results/cache/macro_N100_cw1_f770143b36.npz
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }

finish() {
  log "finish status=$STATUS"
  cp logs/amend16_chain_pod.log $L/ 2>/dev/null
  for f in $R/report.json $R/cbf_check.json $R/p3_viz/*.gif $R/p*/*/*/metrics_seed*.json $L/*.log $L/*.json $L/*.sh; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "amend16 result ($STATUS): robust-margin frontier, gate-width diagnostic, §8 viz + CBF wiring check (RunPod GPU)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i), retry in 60 s"; sleep 60
  done
  git status -sb | head -1
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > logs/amend16_$STATUS
}
trap finish EXIT

rollouts() {   # $1 = cell, $2.. = arms; seeds 0-4; skip if metrics exist
  local cell=$1; shift
  export cell W L
  for s in 0 1 2 3 4; do for arm in "$@"; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      rn=$(python -c "import yaml,sys; print(yaml.safe_load(open(\"experiments/amend16_cells/$cell.yaml\"))[\"run_name\"])")
      m=results/$rn/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $cell $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $cell $0 $1"
      python -m experiments.ladder --config experiments/amend16_cells/$cell.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > $L/eval_${cell}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $cell $0 $1 exit=$? $(tail -1 $L/eval_${cell}_$0_s$1.log)"'
}

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend16_gate_diagnostic.md | cut -c1-64) HEAD $(git rev-parse HEAD)"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
if [ ! -f $CACHE105 ]; then
  # DEVIATION (logged in INDEX): old pod 6k5omnvf9h2ulr could not get its GPU back, run migrated to a new A40 pod;
  # the amend15 s=1.05 cache lived on the old pod's volume -> one re-solve with the identical key (same config, same
  # GPU type). Identity is checked in the report (p1 d0.55 vs amend15 real s=1.05, per seed).
  log "DEVIATION: s=1.05 cache $CACHE105 missing on this pod -> re-solve (same key)"
  python -m experiments.amend16_gate_diagnostic solve --cells p1_d0.55 > $L/solve_s1.05_resolve.log 2>&1
  log "re-solve exit=$? $(grep -o '"cache": "[^"]*"\|"solve_seconds": [0-9.]*\|"nan": [a-z]*' $L/solve_s1.05_resolve.log | tr '\n' ' ')"
  [ -f $CACHE105 ] || { log "re-solve did not produce $CACHE105 -> STOP"; exit 1; }
fi
log "s=1.05 cache present: $CACHE105 sha256 $(sha256sum $CACHE105 | cut -c1-16)"
python -m experiments.amend16_gate_diagnostic geometry > $L/geometry.json 2>&1

# ---- Part 1
for d in 0.55 0.90 1.30 1.90; do rollouts p1_d$d real; done
n=$(ls $R/p1/d*/real/metrics_seed*.json 2>/dev/null | wc -l); log "part1 done: $n/20 metrics"
grep -h macro_cache $R/p1/d*/real/metrics_seed*.json | sort | uniq -c | while read l; do log "part1 cache: $l"; done

# ---- Part 3 (seed 0 only, log_xhat)
for name in real real_robust_d1.90 learned; do
  arm=real; [ $name = learned ] && arm=learned
  m=$R/p3/$name/$arm/metrics_seed0.json
  [ -f $m ] || python -m experiments.ladder --config experiments/amend16_cells/p3_$name.yaml --mode eval --arm $arm \
    --weights $W --seeds 0 > $L/eval_p3_${name}_s0.log 2>&1
  log "part3 rollout $name exit=$? $(tail -1 $L/eval_p3_${name}_s0.log 2>/dev/null)"
done
python -m experiments.amend16_gate_diagnostic cbf-check > $L/cbf_check.log 2>&1; log "cbf-check exit=$?"
tail -3 $L/cbf_check.log | while read l; do log "CBF: $l"; done
python -m experiments.amend16_gate_diagnostic render > $L/render.log 2>&1; log "render exit=$?"

# ---- Part 2
for g in 1.50 2.30 3.50 5.00 7.00; do
  if ! grep -q '"nan": false' $L/solve_g$g.log 2>/dev/null; then
    log "solve start gate=$g"
    python -m experiments.amend16_gate_diagnostic solve --cells p2_g$g > $L/solve_g$g.log 2>&1
    log "solve done gate=$g exit=$? $(grep -o '"solve_seconds": [0-9.]*' $L/solve_g$g.log)"
    grep -q '"nan": false' $L/solve_g$g.log || { log "solve gate=$g failed or NaN -> skip cell"; continue; }
  fi
  rollouts p2_g$g real learned
  log "part2 gate=$g: $(ls $R/p2/g$g/*/metrics_seed*.json 2>/dev/null | wc -l)/10 metrics"
done

# ---- report
python -m experiments.amend16_gate_diagnostic report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"
[ $rc = 0 ] && STATUS=DONE
log "chain end"
