#!/usr/bin/env bash
# amend17 pre-run (before the aneks commit): re-solve the s=1.05 Phi/m* cache on the new pod (same key f770143b36,
# DEVIATION: the amend15/16 cache is on stopped pod 2p0sfca3j5hbny whose host has no free GPU), then the k=0 sanity
# pair (san_k0 vs san_const, seed 0, real + learned) and the element-wise comparison.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
L=logs/amend17_pod
W=results/ladder_fargoal_N100/predictor_weights.npz
log() { echo "$(date "+%F %T") $*"; }
log "start gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) jax=$(python -c "import jax; print(jax.__version__, jax.devices())")"
python -m experiments.amend16_gate_diagnostic solve --cells p1_d0.55 > $L/solve_s1.05_resolve.log 2>&1
log "solve exit=$? $(cat $L/solve_s1.05_resolve.log | tail -1)"
log "cache sha256 $(sha256sum results/cache/macro_N100_cw1_f770143b36.npz | cut -c1-16)"
for cell in san_const san_k0; do for arm in real learned; do
  python -m experiments.ladder --config experiments/amend17_cells/$cell.yaml --mode eval --arm $arm --weights $W \
    --seeds 0 > $L/eval_${cell}_${arm}_s0.log 2>&1
  log "$cell $arm exit=$? $(tail -1 $L/eval_${cell}_${arm}_s0.log)"
done; done
python -m experiments.amend17_adaptive_margin sanity > $L/sanity.log 2>&1
log "sanity exit=$?"; cat $L/sanity.log
log "end"
