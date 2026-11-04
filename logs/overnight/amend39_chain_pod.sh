#!/usr/bin/env bash
# amend39 (Config G headline cells under the settled metric + 0.30 m band; raw trajectories archived).
# Prereg: prereg/prereg_projectA_amend39_settled_regen.md. NOT launched by the drafting agent: start it on an A100-80GB
# pod (volume >= 200 GB) with   nohup bash logs/overnight/amend39_chain_pod.sh > /workspace/amend39_chain.log 2>&1 &
# Order: reproduction gate -> per stage (A, B, C, D): warm phase (seed 0 of every cell, 1 process, so every macro solve
# runs alone) -> seeds 1-9 (2 processes) -> settle (settled_seed*.json + traj_seed*.npz) -> archive manifest ->
# checkpoint (git: json + manifest + logs, never npz). Gate caches are deleted after stage C. Report at the end.
# Set NO_PUSH=1 if the pod has no GitHub key (amend35 path: workstation pulls a bundle). The raw-trajectory archive is
# pulled by the workstation with logs/overnight/amend39_pull_archive.sh; do not delete the pod volume before that.
cd /workspace/MUSC_MFG_PAPER
mkdir -p ~/.ssh && cp -n /workspace/.ssh/id_ed25519* ~/.ssh/ 2>/dev/null; chmod 600 ~/.ssh/id_ed25519 2>/dev/null
. logs/env.sh 2>/dev/null
. logs/overnight/push_lib.sh
A=amend39; L=logs/amend39_pod; R=results/amend39_settled_regen; CD=experiments/amend39_cells; M=experiments.amend39_settled_regen
W=results/overnight_p1/predictor_v3/predictor_weights.npz
STAGES=${STAGES:-"A B C D"}
mkdir -p $L; STATUS=FAILED
log() { echo "$(date '+%F %T') $*"; }
manifest() { (cd $R && find . -name 'traj_seed*.npz' -o -name 'settled_seed*.json' | sort | xargs sha256sum) > $R/ARCHIVE_MANIFEST.txt; }
finish() {
  log "finish status=$STATUS"
  [ -d $R ] && manifest
  checkpoint "amend39 result ($STATUS): Config G headline cells under settled metric + 0.30 m band, seeds 0-9, precision=highest; raw trajectories archived (manifest)" \
    $R/report.json $R/figure_data_amend39.csv $R/identity.json $R/ARCHIVE_MANIFEST.txt $(find $R -name 'metrics_seed*.json' -o -name 'settled_seed*.json' 2>/dev/null) $L/*.log
  echo "$STATUS $(date '+%F %T') $(git rev-parse HEAD)" > /workspace/${A}_$STATUS
}
trap finish EXIT

log "chain start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) HEAD $(git rev-parse HEAD) disk $(df -h /workspace | tail -1)"
log "aneks sha256 $(sha256sum prereg/prereg_projectA_amend39_*.md | cut -c1-64); weights sha256 $(sha256sum $W | cut -c1-64)"
[ "$(sha256sum $W | cut -c1-64)" = 9583cb7ce53bc33a5b739459c51efcf7cc7cb18f761f2d4115f6eee202b2ffb5 ] || { log "v3 weights sha mismatch"; exit 1; }
python -m $M cells > $L/cells.log 2>&1; log "cells: $(tail -1 $L/cells.log)"
git diff --quiet -- $CD || { log "regenerated cells differ from the committed ones -> abort"; exit 1; }

ev() {  # $1 = cell yaml, $2.. = seeds (skip seeds already done)
  y=$1; shift; c=$(basename $y .yaml); arm=$(python -m $M arm $y)
  rn=$(python -c "import yaml; print(yaml.safe_load(open('$y'))['run_name'])"); ad=$(echo "$arm" | tr ':/' '_-')
  miss=""; for s in "$@"; do [ -f results/$rn/$ad/metrics_seed$s.json ] || miss="$miss $s"; done
  [ -z "$miss" ] && return 0
  python -m experiments.ladder --config $y --mode eval --arm "$arm" --weights $W --seeds $miss >> $L/eval_$c.log 2>&1
  echo "$(date '+%F %T') $c ($arm) seeds[$miss ] exit=$?"
}
export -f ev; export L W M

# reproduction gate (prereg): real_d0.70 N=100 seed 0 vs registered amend22
ev $CD/A__amend22__N100_real_d0.70.yaml 0
python - > $L/identity.log 2>&1 <<'PY'
import json
keys = ("completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m",
        "cbf_intervention_rate", "slack_active_rate")
a = json.load(open("results/amend22_N_crossover_pooled/N100/real_d0.70/real/metrics_seed0.json"))
b = json.load(open("results/amend39_settled_regen/A/amend22/N100_real_d0.70/real/metrics_seed0.json"))
out = {k: [a.get(k), b.get(k)] for k in keys}
out["all_equal"] = all(v[0] == v[1] for v in out.values())
json.dump(out, open("results/amend39_settled_regen/identity.json", "w"), indent=1)
print(out)
PY
log "reproduction gate: $(tail -1 $L/identity.log) (non-equal on a non-A100 GPU -> descriptive comparisons only, per prereg)"

for S in $STAGES; do
  before=$(ls results/cache/*.npz 2>/dev/null | sort)
  log "stage $S warm phase (seed 0, serial)"
  for y in $CD/${S}__*.yaml; do ev $y 0; done
  log "stage $S seeds 1-9 (2 in parallel)"
  ls $CD/${S}__*.yaml | xargs -P 2 -I{} bash -c 'ev {} 1 2 3 4 5 6 7 8 9'
  validate_run_dirs $R/$S | tail -1
  ls $CD/${S}__*.yaml | xargs -P 2 -I{} bash -c 'ev {} 0 1 2 3 4 5 6 7 8 9'   # re-run anything validate deleted
  python -m $M settle --stage $S > $L/settle_$S.log 2>&1; log "settle $S: $(tail -1 $L/settle_$S.log)"
  n=$(find $R/$S -name 'settled_seed*.json' | wc -l); log "stage $S settled rollouts: $n"
  manifest
  checkpoint "amend39 checkpoint stage $S ($n settled rollouts; partial, not a result)" \
    $R/identity.json $R/ARCHIVE_MANIFEST.txt $(find $R/$S -name 'metrics_seed*.json' -o -name 'settled_seed*.json') $L/*.log
  if [ "$S" = C ]; then   # gate caches are used only by stage C
    for f in $(comm -13 <(echo "$before") <(ls results/cache/*.npz | sort)); do rm -f $f $f.json; log "deleted stage-C cache $f"; done
  fi
done

python -m $M report > $L/report.log 2>&1; rc=$?
log "report exit=$rc"; tail -40 $L/report.log
total=$(find $R -name 'settled_seed*.json' | wc -l); log "settled rollouts total: $total/690"
[ $rc = 0 ] && [ "$total" = 690 ] && STATUS=DONE
[ $rc = 0 ] && [ "$total" != 690 ] && STATUS=PARTIAL
