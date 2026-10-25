#!/usr/bin/env bash
# Overnight Phase 1a (EXPLORATORY): per-N Phi/m* re-solve on the new pod (amend21 cells, identical cache keys),
# identity check (amend21 real seed 0 at each N re-run, metrics must equal committed amend21), datagen real-arm
# seeds 100-111 at each N, pooled predictor v1 training, open-loop OOD gate. Self-commit/push of reports (not datagen).
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
L=logs/overnight/p1
R=results/overnight_p1
STATUS=FAILED
TAG=${TAG:-v1}; TRAIN_ARGS=${TRAIN_ARGS:-}
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $L/*.log $L/*.sh $L/*.json $R/predictor_*/predictor_report.json $R/predictor_*/predictor_weights.npz $R/openloop_*.json \
           $R/identity_N*/real/metrics_seed0.json; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "overnight P1a ($STATUS, exploratory): per-N re-solve + identity vs amend21, pooled datagen, predictor $TAG, open-loop gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > $L/p1a_${TAG}_$STATUS.marker
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/p1a_$STATUS
}
trap finish EXIT
log "start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())') HEAD $(git rev-parse --short HEAD)"
for N in 25 50 100 150 200; do
  NN=$(printf "%03d" $N)
  if ! ls results/cache/ 2>/dev/null | grep -q "^macro_N${N}_cw1_"; then
    log "solve start N=$N"
    python -m experiments.amend21_N_crossover solve --N $N > $L/solve_N$NN.log 2>&1
    log "solve done N=$N exit=$? $(tail -1 $L/solve_N$NN.log)"
  fi
  ls results/cache | grep -q "^macro_N${N}_cw1_" || { log "solve N=$N failed"; exit 1; }
done
# identity + datagen AFTER all solves (run concurrently with a solve they OOMed the GPU), 2 processes at a time
export L R
for N in 25 50 100 150 200; do echo $N; done | xargs -P 2 -L 1 bash -c '
  NN=$(printf "%03d" $0)
  [ -f $R/identity_N$NN/real/metrics_seed0.json ] || python -m experiments.ladder --config experiments/overnight_p1_cells/identity_N$NN.yaml --mode eval --arm real --seeds 0 > $L/identity_N$NN.log 2>&1
  miss=""; for s in 100 101 102 103 104 105 106 107 108 109 110 111; do [ -f $R/datagen_N$NN/datagen/pos_seed$s.npz ] || miss="$miss $s"; done
  [ -n "$miss" ] && python -m experiments.ladder --config experiments/overnight_p1_cells/datagen_N$NN.yaml --mode datagen --seeds $miss > $L/datagen_N$NN.log 2>&1
  echo "$(date "+%F %T") N=$0 identity+datagen exit=$? $(tail -1 $L/datagen_N$NN.log)"'
python - > $L/identity_vs_amend21.json <<'PY'
import json
K = ["completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m", "cbf_intervention_rate",
     "slack_active_rate", "closedloop_xhat_mse_near"]
out = {}
for N in [25, 50, 100, 150, 200]:
    a = json.load(open(f"results/amend21_N_crossover/N{N:03d}/real/real/metrics_seed0.json"))
    b = json.load(open(f"results/overnight_p1/identity_N{N:03d}/real/metrics_seed0.json"))
    out[N] = {"equal": all(a[k] == b[k] for k in K), **{k: [a[k], b[k]] for k in K}}
out["all_equal"] = all(v["equal"] for v in out.values())
print(json.dumps(out, indent=1))
PY
log "identity vs amend21 all_equal=$(python -c "import json; print(json.load(open('$L/identity_vs_amend21.json'))['all_equal'])")"
n=$(ls $R/datagen_N*/datagen/pos_seed*.npz | wc -l); log "datagen files $n/60"; [ "$n" = 60 ] || exit 1
log "train $TAG start ($TRAIN_ARGS)"
python -m experiments.overnight_p1_predictor train --tag $TAG $TRAIN_ARGS > $L/train_$TAG.log 2>&1 || exit 1
log "train done sha=$(python -c "import json; print(json.load(open('$R/predictor_$TAG/predictor_report.json'))['sha256'])")"
python -m experiments.overnight_p1_predictor openloop --weights $R/predictor_$TAG/predictor_weights.npz \
  --weights_old results/ladder_fargoal_N100/predictor_weights.npz --out $R/openloop_$TAG.json > $L/openloop_$TAG.log 2>&1 || exit 1
cat $L/openloop_$TAG.log
STATUS=DONE
