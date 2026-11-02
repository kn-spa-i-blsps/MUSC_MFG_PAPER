#!/usr/bin/env bash
# amend28 (baseline validation + ablations + macro fidelity), detached after amend27. Solves first (no rollouts next to
# a solve), rollouts 2 in parallel, fidelity per cell, report, commit -f, push, delete the fresh_nomfg caches, marker.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
. logs/overnight/push_lib.sh
A=amend28; L=logs/amend28_pod; R=results/amend28_baseline_validation; CD=experiments/amend28_cells; M=experiments.amend28_baseline_validation
W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $R/report.json $R/fidelity/*.json $R/N*/*/*/metrics_seed*.json $L/*.log; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "amend28 result ($STATUS): delay-free baseline validation, CBF/MFG ablations, MFG macro fidelity, seeds 0-9, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/${A}_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD) disk $(du -sh /workspace | cut -f1)"
log "volume check: $(validate_run_dirs $R | tail -1)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend28_*.md | cut -c1-64)"
for N in 25 50 100 150 200; do
  python -m $M solve-nomfg --N $N > $L/solve_nomfg_N$N.log 2>&1; log "solve nomfg N=$N exit=$? $(tail -n 1 $L/solve_nomfg_N$N.log)"
done
export L W
for NN in 025 050 100 150 200; do
ls $CD/N${NN}_*.yaml | xargs -P 2 -L 1 bash -c '
  y=$0; c=$(basename $y .yaml); arm=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"amend28_arm\"])")
  rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/$arm/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0
  python -m experiments.ladder --config $y --mode eval --arm $arm --weights $W --seeds $miss > $L/eval_$c.log 2>&1
  echo "$(date "+%F %T") $c ($arm) exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
checkpoint "amend28 checkpoint N=$NN (partial rollouts, not a result)" $R/N$NN/*/*/metrics_seed*.json $L/*.log
done
n=$(ls $R/N*/*/*/metrics_seed*.json | wc -l); log "rollouts: $n/300"
grep -h '"matmul_precision"' $R/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
for N in 25 50 100 150 200; do for arm in fresh fresh_nocbf fresh_nomfg noise_only real learned; do echo "$N $arm"; done; done | xargs -P 2 -L 1 bash -c '
  python -m '$M' fidelity --N $0 --arm $1 >> '$L'/fidelity.log 2>&1 || echo "fidelity FAIL N=$0 $1" >> '$L'/fidelity.log'
log "fidelity: $(grep -c floor $L/fidelity.log)/30 $(grep -c FAIL $L/fidelity.log) fail"
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -50 $L/report.log
checkpoint "amend28 report (exit $rc, $n/300 rollouts): pushed before the next amendment" $R/report.json $R/fidelity $R/N*/*/*/metrics_seed*.json $L/*.log
[ $rc = 0 ] && [ "$n" = 300 ] && STATUS=DONE
for f in $(ls results/cache/macro_N*_cw0_*.npz 2>/dev/null); do grep -q '"w_bar": 0.0' $f.json && rm -f $f && log "deleted nomfg cache $f"; done
