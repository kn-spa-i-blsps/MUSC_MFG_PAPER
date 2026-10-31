#!/usr/bin/env bash
# Phase B step 1 — gate G-pred (criteria fixed before data: logs/overnight/phaseB_gpred_gate.md, commit b020562), detached.
# The pod volume that held the pooled-geometry histories (xby8jnv4zqwvn2) no longer exists -> regenerate everything with
# the unchanged code/keys: 5 per-N solves (geometry_N=230) -> real-arm datagen + learned-v3 d0.70 datagen (seeds 100-111)
# -> integrity check (v3 open-loop on the regenerated histories vs the committed openloop_v3.json / openloop2_v3_datagenL.json)
# -> G-pred: default recipe D (+ per-N ceiling) and the whole §C ladder (cheap; verdict still D first, ladder by VAL)
# -> secondary D on real-arm histories -> git add -f, commit, push -> /workspace/phaseB_gpred_DONE|FAILED.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519
git config user.name Gio; git config user.email mikojasiek@gmail.com
. logs/env.sh
L=logs/overnight/phaseB; R=results/overnight_p1; P=results/phaseB; W3=$R/predictor_v3/predictor_weights.npz
mkdir -p $L $P
STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
finish() {
  log "finish status=$STATUS"
  for f in $P/gpred_*.json $R/openloop_v3_regen*.json $L/*.log; do [ -e "$f" ] && git add -f "$f"; done
  git commit -q -m "Phase B G-pred gate result ($STATUS): v3 error predictability on regenerated pooled-N histories, D + §C ladder, precision=highest (RunPod A100)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "commit $(git rev-parse --short HEAD)"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && { log "push ok $(git rev-parse --short HEAD)"; break; }
    log "push failed (try $i)"; sleep 60
  done
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/phaseB_gpred_$STATUS
}
trap finish EXIT
log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD)"

for N in 25 50 100 150 200; do
  NN=$(printf "%03d" $N)
  python -m experiments.amend21_N_crossover solve --N $N > $L/solve_N$NN.log 2>&1
  log "solve N=$N exit=$? $(tail -n 1 $L/solve_N$NN.log)"
done
python -m experiments.overnight_p1_predictor write-cells > /dev/null
export R L W3
for src in datagen datagenL; do
  export src
  for N in 25 50 100 150 200; do echo $N; done | xargs -P 2 -L 1 bash -c '
    NN=$(printf "%03d" $0)
    miss=""; for s in 100 101 102 103 104 105 106 107 108 109 110 111; do [ -f $R/${src}_N$NN/datagen/pos_seed$s.npz ] || miss="$miss $s"; done
    extra=""; [ $src = datagenL ] && extra="--datagen_arm learned --weights $W3"
    [ -n "$miss" ] && python -m experiments.ladder --config experiments/overnight_p1_cells/${src}_N$NN.yaml --mode datagen $extra --seeds $miss > $L/${src}_N$NN.log 2>&1
    echo "$(date "+%F %T") $src N=$0 exit=$?"'
  log "$src files $(ls $R/${src}_N*/datagen/pos_seed*.npz | wc -l)/60"
done

# integrity: regenerated histories vs the committed open-loop numbers (real path is bit-deterministic; the learned
# closed loop may differ by GPU autotuning -> reported, not a blocker)
python -m experiments.overnight_p1_predictor openloop --weights $W3 --src datagen --out $R/openloop_v3_regen_datagen.json > $L/openloop_v3_regen_datagen.log 2>&1
python -m experiments.overnight_p1_predictor openloop --weights $W3 --src datagenL --out $R/openloop_v3_regen_datagenL.json > $L/openloop_v3_regen_datagenL.log 2>&1
python - > $L/integrity.log 2>&1 <<'EOF'
import json
for new, old in (("datagen", "openloop_v3.json"), ("datagenL", "openloop2_v3_datagenL.json")):
    try:
        a = json.load(open(f"results/overnight_p1/{old}"))["per_N"]; b = json.load(open(f"results/overnight_p1/openloop_v3_regen_{new}.json"))["per_N"]
        print(new, "exact_equal_per_N:", {N: a[N] == b.get(N) for N in a})
        for N in a:
            for k in a[N]:
                if isinstance(a[N][k], dict) and "mse_near1m_m2" in a[N][k]:
                    print(f"  N={N} {k}: old {a[N][k]['mse_near1m_m2']:.6g} new {b[N][k]['mse_near1m_m2']:.6g}")
    except Exception as e:
        print(new, "compare error", repr(e))
EOF
log "integrity: $(grep exact_equal $L/integrity.log | tr '\n' ' ')"

run() { t=$1; shift; python -m experiments.phaseB_gpred --tag $t --weights $W3 --out $P/gpred_$t.json "$@" > $L/gpred_$t.log 2>&1; log "gpred $t exit=$?"; }
run D_datagenL --src datagenL --per_N
run cap3x256_datagenL --src datagenL --hidden 256 --depth 3
run nf0.8_datagenL --src datagenL --near_frac 0.8
run nf0.0_datagenL --src datagenL --near_frac 0.0
run w200k_datagenL --src datagenL --samples 200000
run D_datagen --src datagen
STATUS=DONE
