#!/usr/bin/env bash
# amend19 pre-run (before the aneks commit), new pod 8thhi6wqzzclov: re-solve the s=1.05 Phi/m* cache (same key
# f770143b36; pod k9nrewd0bhycsu could not get its GPU back), cache identity check (amend16 p1_d0.55 real seed 0 vs the
# committed amend16 metrics), then the k=0 sanity pair (san_k0 vs san_fixed0.55, seed 0, learned + real) with XLA
# autotuning OFF and the element-wise comparison.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
L=logs/amend19_pod
W=results/ladder_fargoal_N100/predictor_weights.npz
log() { echo "$(date "+%F %T") $*"; }
log "start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c "import jax; print(jax.__version__, jax.devices())")"
log "weights sha256 $(sha256sum $W | cut -c1-64)"
python -m experiments.amend16_gate_diagnostic solve --cells p1_d0.55 > $L/solve_s1.05_resolve.log 2>&1
log "solve exit=$? $(tail -1 $L/solve_s1.05_resolve.log)"
sed "s#run_name: .*#run_name: amend19_closing_margin_N100/cache_identity/p1_d0.55#" experiments/amend16_cells/p1_d0.55.yaml > /tmp/idc.yaml
python -m experiments.ladder --config /tmp/idc.yaml --mode eval --arm real --weights $W --seeds 0 > $L/cache_identity_eval.log 2>&1
python - > $L/cache_identity.json <<'PY'
import json
a = json.load(open("results/amend16_gate_diagnostic_N100/p1/d0.55/real/metrics_seed0.json"))
b = json.load(open("results/amend19_closing_margin_N100/cache_identity/p1_d0.55/real/metrics_seed0.json"))
ks = ["completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m", "cbf_intervention_rate",
      "slack_active_rate"]
r = {k: [a[k], b[k]] for k in ks}
print(json.dumps({"cache_identity_amend16_p1_d0.55_real_s0_[amend16,new_cache]": r,
                  "all_equal": all(v[0] == v[1] for v in r.values())}))
PY
log "cache identity: $(cat $L/cache_identity.json)"
for cell in san_fixed0.55 san_k0; do for arm in learned real; do
  XLA_FLAGS=--xla_gpu_autotune_level=0 python -m experiments.ladder --config experiments/amend19_cells/$cell.yaml \
    --mode eval --arm $arm --weights $W --seeds 0 > $L/eval_${cell}_${arm}_s0.log 2>&1
  log "$cell $arm exit=$? $(tail -1 $L/eval_${cell}_${arm}_s0.log)"
done; done
python -m experiments.amend19_closing_margin sanity > $L/sanity.log 2>&1
log "sanity exit=$?"; cat $L/sanity.log
log "end"
