#!/usr/bin/env bash
# amend21 on the RunPod GPU (A100 80GB pod r863dyrq106k4m; the A40 OOMed in the solve), detached (tmux). N crossover on ONE fixed geometry (geometry_N 230,
# gate 2.309), precision=highest, N in {25,50,100,150,200} x {real, learned fixed 0.70, learned closing 0.75} x seeds 0-9.
# Per N: Phi/m* solve (PI decision: mass = N, one solve per N, shared by the 3 arms) -> 30 rollouts. -> report ->
# git add -f results + logs, commit, push -> DONE. EXIT trap commits/pushes whatever exists (DONE/FAILED). Pod left running.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
W=results/ladder_fargoal_N100/predictor_weights.npz
R=results/amend21_N_crossover
L=logs/amend21_pod
P_ROLL=2
DISK_MIN_GB=20
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }

finish() {
  log "finish status=$STATUS"
  echo "$STATUS $(date '+%F %T') (pre-commit)" > $L/amend21_$STATUS
  cp logs/amend21_chain_pod.log $L/ 2>/dev/null
  for f in $R/report.json $R/N*/*/*/metrics_seed*.json $L/*.log $L/*.sh $L/amend21_* $R/geometry.json $L/*.json; do
    [ -e "$f" ] && git add -f "$f"
  done
  git commit -q -m "amend21 result ($STATUS): N crossover on a fixed geometry, N in {25..200} x {real, learned fixed 0.70, learned closing 0.75}, seeds 0-9, precision=highest (RunPod GPU)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase origin main && git push origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i), retry in 60 s"; sleep 60
  done
  git status -sb | head -1
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > logs/amend21_$STATUS
}
trap finish EXIT

rollouts() {   # $1 = cell, $2.. = arms; seeds 0-9; skip if metrics exist
  local cell=$1; shift
  export cell W L
  for s in 0 1 2 3 4 5 6 7 8 9; do for arm in "$@"; do echo "$arm $s"; done; done |
    xargs -P $P_ROLL -L 1 bash -c '
      rn=$(python -c "import yaml,sys; print(yaml.safe_load(open(\"experiments/amend21_cells/$cell.yaml\"))[\"run_name\"])")
      m=results/$rn/$0/metrics_seed$1.json
      if [ -f $m ]; then echo "$(date "+%F %T") skip $cell $0 $1 (exists)"; exit 0; fi
      echo "$(date "+%F %T") start $cell $0 $1"
      python -m experiments.ladder --config experiments/amend21_cells/$cell.yaml --mode eval --arm $0 --weights $W \
        --seeds $1 > $L/eval_${cell}_$0_s$1.log 2>&1
      echo "$(date "+%F %T") done $cell $0 $1 exit=$? $(tail -1 $L/eval_${cell}_$0_s$1.log)"'
}

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c 'import jax; print(jax.__version__, jax.devices())')"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend21_N_crossover.md | cut -c1-64) HEAD $(git rev-parse HEAD)"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
mkdir -p $R; python -m experiments.amend21_N_crossover geometry > $R/geometry.json 2> $L/geometry.err
log "geometry: $(python -c "import json; g=json.load(open('$R/geometry.json')); p=g['per_N']['200']; print('arena', p['arena'], 'gate', p['gate_width'], 'r_goal', p['r_goal'], 'start_box', p['start_box'], 'horizon', p['horizon_s'], 'identical', all(g['geometry_identical_across_N'].values()), 'dropped', g['dropped_N'], 'minsep', {N: round(v['min_initial_sep_seeds0_9'], 3) for N, v in g['per_N'].items()})")"
for N in 25 50 100 150 200; do
  NN=$(printf "%03d" $N)
  if python -c "import json,sys; sys.exit(0 if $N in json.load(open('$R/geometry.json'))['dropped_N'] else 1)"; then log "N=$N dropped (overlap) -> skip"; continue; fi
  log "solve start N=$N"
  python -m experiments.amend21_N_crossover solve --N $N > $L/solve_N$NN.log 2>&1
  log "solve done N=$N exit=$? $(tail -1 $L/solve_N$NN.log)"
  grep -q '"nan": false' $L/solve_N$NN.log || { log "solve N=$N failed or NaN -> skip N"; continue; }
  rollouts N${NN}_real real
  rollouts N${NN}_learned_fixed0.70 learned
  rollouts N${NN}_learned_closing0.75 learned
  log "N=$N: $(ls $R/N$NN/*/*/metrics_seed*.json 2>/dev/null | wc -l)/30 metrics"
  free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$DISK_MIN_GB" ]; then c=$(grep -o '"cache": "[^"]*"' $L/solve_N$NN.log | cut -d'"' -f4); log "disk ${free}G < ${DISK_MIN_GB}G -> delete cache $c"; rm -f "$c"; fi
done
n=$(ls $R/N*/*/*/metrics_seed*.json 2>/dev/null | wc -l); log "rollouts done: $n/150 metrics"
grep -h matmul_precision $R/N*/*/*/metrics_seed*.json | sort | uniq -c | while read l; do log "precision: $l"; done

python -m experiments.amend21_N_crossover report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"
[ $rc = 0 ] && [ "$n" = 150 ] && STATUS=DONE
log "chain end"
