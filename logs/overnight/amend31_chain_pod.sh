#!/usr/bin/env bash
# amend31 chain (Phase B steps 2-3), FIRST in the queue (addendum 1), staged + pushed per stage: sigma head + conformal (G-cal), identity gate (aborts on
# failure), 35 cells x 10 seeds 2 in parallel, report, commit -f, push, marker /workspace/amend31_DONE|FAILED.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
. logs/env.sh
. logs/overnight/push_lib.sh
M=experiments.amend31_sigma_cbf; L=logs/amend31_pod; CD=experiments/amend31_cells; R=results/amend31_sigma_cbf
S=results/phaseB/sigma_v3; W=results/overnight_p1/predictor_v3/predictor_weights.npz
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $S/sigma_weights.npz $S/calibration.json $S/reliability.png $R/report.json $R/identity.json $R/*/*/*/metrics_seed*.json $CD $L/*.log; do
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
log "volume check: $(validate_run_dirs $R | tail -1)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend31_sigma_cbf.md | cut -c1-64); v3 sha256 $(sha256sum $W | cut -c1-64)"
log "addendum3 sha256 $(sha256sum prereg/prereg_projectA_amend31_addendum3.md | cut -c1-64)"
log "addendum1 sha256 $(sha256sum prereg/prereg_projectA_amend31_addendum1.md | cut -c1-64)"
git config user.name Gio; git config user.email mikojasiek@gmail.com
# ---- stage 0 (addendum 1): amend31 now runs FIRST, so regenerate its inputs if the volume lost them, with the unchanged
# code/keys of the G-pred chain: 5 per-N solves (a cache hit just loads), learned-v3 d0.70 datagenL seeds 100-111.
D=results/overnight_p1
for N in 25 50 100 150 200; do
  NN=$(printf "%03d" $N)
  python -m experiments.amend21_N_crossover solve --N $N > $L/stage0_solve_N$NN.log 2>&1
  log "stage0 solve N=$N exit=$? $(tail -n 1 $L/stage0_solve_N$NN.log | cut -c1-160)"
done
python - > $L/stage0_datagenL_check.log 2>&1 <<'EOF'
import glob, numpy as np, os
bad = 0
for p in sorted(glob.glob("results/overnight_p1/datagenL_N*/datagen/pos_seed*.npz")):
    try:
        with np.load(p) as f:
            for k in f.files: np.asarray(f[k]).sum()
    except Exception as e:
        bad += 1; print("invalid -> deleted", p, repr(e)); os.remove(p)
print("datagenL files ok:", len(glob.glob("results/overnight_p1/datagenL_N*/datagen/pos_seed*.npz")), "deleted:", bad)
EOF
log "stage0 $(tail -1 $L/stage0_datagenL_check.log)"
python -m experiments.overnight_p1_predictor write-cells > /dev/null
export D L W
for N in 25 50 100 150 200; do echo $N; done | xargs -P 2 -L 1 bash -c '
  NN=$(printf "%03d" $0)
  miss=""; for s in 100 101 102 103 104 105 106 107 108 109 110 111; do [ -f $D/datagenL_N$NN/datagen/pos_seed$s.npz ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0
  python -m experiments.ladder --config experiments/overnight_p1_cells/datagenL_N$NN.yaml --mode datagen --datagen_arm learned --weights $W --seeds $miss > $L/stage0_datagenL_N$NN.log 2>&1
  echo "$(date "+%F %T") stage0 datagenL N=$0 exit=$?"'
nd=$(ls $D/datagenL_N*/datagen/pos_seed*.npz | wc -l); log "stage0 datagenL files $nd/60"
[ "$nd" = 60 ] || { log "stage0 FAILED: datagenL incomplete"; exit 1; }
checkpoint "amend31 stage 0: inputs ready (caches + datagenL 60/60), logs only" $L/*.log
# ---- stage 1: sigma head (recipe D) + conformal calibration, gate G-cal (+ reliability diagram per N)
if [ -f $S/calibration.json ] && [ -f $S/sigma_weights.npz ] && git log --oneline -1 -- $S/calibration.json | grep -q "stage 1"; then
  log "stage 1 REUSED (addendum 3): committed sigma head sha $(sha256sum $S/sigma_weights.npz | cut -c1-12), no retraining"
else
python -m experiments.phaseB_sigma --out $S > $L/sigma_train.log 2>&1 || { log "phaseB_sigma FAILED"; tail -5 $L/sigma_train.log; exit 1; }
fi
checkpoint "amend31 stage 1: sigma head + G-cal ($(grep -oE '^G-cal [A-Z_]+' $L/sigma_train.log)), before the sigma-CBF rollouts" $S/sigma_weights.npz $S/calibration.json $S/reliability.png $L/*.log
# ---- stage 2: identity (k=0 == fixed 0.70, real arm, bit-equal to amend22); aborts the chain on failure
log "sigma head: $(grep -E '^alpha|^G-cal' $L/sigma_train.log | tr '\n' ' ')"
python -m $M write-cells > $L/write_cells.log 2>&1 || { log "write-cells FAILED"; exit 1; }
y=$CD/identity/N100_sigma_a0.05_real_k0.yaml
python -m experiments.ladder --config $y --mode eval --arm real --weights $W --seeds 0 > $L/eval_identity.log 2>&1
log "identity rollout exit=$?"
# addendum 2: off-A100 the gating reference is fixed 0.70 real seed 0 on the same GPU
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader); REF=amend22
if ! echo "$GPU" | grep -q "A100-SXM4-80GB"; then
  REF=local
  python -m experiments.ladder --config $CD/identity/N100_fixed0.70_real_ref.yaml --mode eval --arm real --weights $W --seeds 0 > $L/eval_identity_ref.log 2>&1
  log "identity same-GPU reference rollout ($GPU) exit=$?"
fi
python -m $M identity --ref $REF > $L/identity.log 2>&1 || { log "identity FAILED: $(cat $L/identity.log | tail -2)"; exit 1; }
log "identity PASS: $(tail -1 $L/identity.log)"
checkpoint "amend31 stage 2: identity PASS" $R/identity.json $CD $L/*.log
# ---- stage 3: sigma-CBF vs fixed frontier, per N, pushed per N; then the report (G-win)
export L W
for NN in 025 050 100 150 200; do
ls $CD/N${NN}_*.yaml | xargs -P 2 -L 1 bash -c '
  y=$0; c=$(basename $y .yaml)
  rn=$(python -c "import yaml; print(yaml.safe_load(open(\"$y\"))[\"run_name\"])")
  miss=""; for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/$rn/learned/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && exit 0
  python -m experiments.ladder --config $y --mode eval --arm learned --weights $W --seeds $miss > $L/eval_$c.log 2>&1
  echo "$(date "+%F %T") $c exit=$? $(grep -c "seed=" $L/eval_$c.log) seeds"'
checkpoint "amend31 checkpoint N=$NN (partial rollouts, not a result)" $R/N$NN/*/*/metrics_seed*.json $R/identity.json $CD $L/*.log
done
expected=$((10 * $(ls $CD/*.yaml | wc -l)))
n=$(ls $R/N*/*/learned/metrics_seed*.json | wc -l); log "rollouts: $n/$expected"
grep -h '"matmul_precision"' $R/N*/*/learned/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done
python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -60 $L/report.log
checkpoint "amend31 report (exit $rc, $n/$expected rollouts): pushed immediately" $S/sigma_weights.npz $S/calibration.json $S/reliability.png $R/report.json $R/identity.json $R/*/*/*/metrics_seed*.json $CD $L/*.log
[ $rc = 0 ] && [ "$n" = "$expected" ] && STATUS=DONE
