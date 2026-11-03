#!/usr/bin/env bash
# amend35 chain (capped sigma-CBF G-win, one shot): solves (cache hit = load) -> sanity S1 (no cap vs amend31 seed 0;
# S2 fallback = amend31 code vs new code, both autotune 0) -> N=150 (push) -> N=200 (push) -> report (push) -> STOP.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
. logs/overnight/push_lib.sh
M=experiments.amend35_sigma_cap; L=logs/amend35_pod; CD=experiments/amend35_cells; R=results/amend35_sigma_cap
A31=results/amend31_sigma_cbf; W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  checkpoint "amend35 chain end ($STATUS)" $R/report.json $R/sanity.json $R/*/*/*/metrics_seed*.json $R/sanity/*/*/metrics_seed*.json $CD $L/*.log
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/amend35_$STATUS
}
trap finish EXIT
git config user.name Gio; git config user.email mikojasiek@gmail.com
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
log "chain start gpu=$GPU HEAD $(git rev-parse HEAD)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend35_sigma_cap.md | cut -c1-64)"
log "sigma weights sha256 $(sha256sum results/phaseB/sigma_v3/sigma_weights.npz | cut -c1-64)"
echo "$GPU" | grep -q "A100-SXM4-80GB" || { log "ABORT: GPU is not A100-SXM4-80GB; the aneks requires re-running the fixed arms (manual)"; exit 1; }
log "amend31 fixed rows: $(validate_run_dirs $A31/N150 $A31/N200 | tail -1)"
nf=0; for N in 150 200; do for d in 0.70 0.85 1.00 1.20; do for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $A31/N$N/fixed$d/learned/logs_seed$s.npz ] && nf=$((nf+1)); done; done; done
log "amend31 fixed npz present: $nf/80"
[ "$nf" = 80 ] || { log "ABORT: amend31 fixed comparator rows missing on the volume"; exit 1; }
for N in 150 200; do
  python -m experiments.amend21_N_crossover solve --N $N > $L/solve_N$N.log 2>&1
  log "solve N=$N exit=$? $(tail -n 1 $L/solve_N$N.log | cut -c1-160)"
done
# ---- sanity S1
y=$CD/sanity/N150_sigma_bin_a0.05_nocap.yaml
python -m experiments.ladder --config $y --mode eval --arm learned --weights $W --seeds 0 > $L/eval_sanity_nocap.log 2>&1
log "S1 rollout exit=$?"
python -m $M sanity > $L/sanity.log 2>&1; rc=$?
log "S1 sanity rc=$rc: $(tail -2 $L/sanity.log | tr '\n' ' ' | cut -c1-400)"
if [ $rc = 2 ]; then
  # ---- S2 fallback: amend31 code (459d624) vs new code no-cap, both XLA autotune 0
  WT=/workspace/a31code
  [ -d $WT ] || git worktree add $WT 459d624 > $L/worktree.log 2>&1
  mkdir -p $WT/results; [ -e $WT/results/cache ] || ln -s /workspace/MUSC_MFG_PAPER/results/cache $WT/results/cache
  [ -e $WT/logs/env.sh ] || cp logs/env.sh $WT/logs/env.sh
  for v in old new; do
    sed "s#^run_name: .*#run_name: amend35_sigma_cap/sanity_s2/$v#" $y > $L/sanity_s2_$v.yaml
  done
  ( cd $WT && XLA_FLAGS=--xla_gpu_autotune_level=0 python -m experiments.ladder --config /workspace/MUSC_MFG_PAPER/$L/sanity_s2_old.yaml \
      --mode eval --arm learned --weights $W --seeds 0 ) > $L/eval_sanity_s2_old.log 2>&1
  log "S2 old-code rollout exit=$? (worktree $(git -C $WT rev-parse --short HEAD))"
  mkdir -p $R/sanity_s2/old && rm -rf $R/sanity_s2/old/learned && cp -r $WT/results/amend35_sigma_cap/sanity_s2/old/learned $R/sanity_s2/old/
  XLA_FLAGS=--xla_gpu_autotune_level=0 python -m experiments.ladder --config $L/sanity_s2_new.yaml --mode eval --arm learned --weights $W --seeds 0 > $L/eval_sanity_s2_new.log 2>&1
  log "S2 new-code rollout exit=$?"
  python -m $M sanity --s2 > $L/sanity.log 2>&1; rc=$?
  log "S2 sanity rc=$rc: $(tail -2 $L/sanity.log | tr '\n' ' ' | cut -c1-400)"
fi
checkpoint "amend35 sanity (rc=$rc), before any capped rollout" $R/sanity.json $R/sanity/*/*/metrics_seed*.json $R/sanity_s2/*/*/metrics_seed*.json $CD $L/*.log
[ $rc = 0 ] || { log "ABORT: sanity failed"; exit 1; }
# ---- capped sigma-CBF, per N, pushed per N
export L W
for N in 150 200; do
  ls $CD/N${N}_*.yaml | xargs -P 2 -L 1 bash -c '
    y=$0; c=$(basename $y .yaml)
    rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
    miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/learned/metrics_seed$s.json ] || miss="$miss $s"; done
    [ -z "$miss" ] && exit 0
    python -m experiments.ladder --config $y --mode eval --arm learned --weights $W --seeds $miss > $L/eval_$c.log 2>&1
    echo "$(date "+%F %T") $c exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
  checkpoint "amend35 N=$N capped sigma rollouts (partial, not a result)" $R/N$N/*/*/metrics_seed*.json $L/*.log
done
n=$(ls $R/N*/*/learned/metrics_seed*.json | wc -l); log "capped rollouts: $n/80"
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -30 $L/report.log
checkpoint "amend35 report (exit $rc, $n/80 rollouts): capped sigma-CBF G-win, pushed immediately" $R/report.json $L/*.log
[ $rc = 0 ] && [ "$n" = 80 ] && STATUS=DONE
