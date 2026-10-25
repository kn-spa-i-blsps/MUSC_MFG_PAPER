#!/usr/bin/env bash
# Overnight tail chain (after the 150 GB volume quota was hit at ~21:29 UTC and amend25/26 writes failed):
# validate existing per-seed outputs (corrupt -> deleted, re-run) -> amend25 (skip existing) -> amend26 (skip existing)
# -> swap caches (delete the 4 extra gate caches, re-solve N=25/50/150/200 with the amend21 keys) -> predictor v4 chain.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
log() { echo "$(date '+%F %T') $*"; }
git pull -q --rebase --autostash origin main; log "tail chain start HEAD $(git rev-parse --short HEAD); disk $(du -sh /workspace | cut -f1)"
validate() {   # $1 = results dir
  JAX_PLATFORMS=cpu python - "$1" <<'PY'
import glob, json, os, sys
import numpy as np
bad = 0
ms = sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "*", "metrics_seed*.json")))
for m in ms:
    z = m.replace("metrics_seed", "logs_seed").replace(".json", ".npz")
    try:
        json.load(open(m)); d = np.load(z); [d[k].sum() for k in d.files]
    except Exception as e:
        bad += 1; print("BAD -> removed", m, repr(e)[:80]); os.remove(m)
        if os.path.exists(z): os.remove(z)
print(f"validate {sys.argv[1]}: {len(ms)} checked, {bad} removed")
PY
}
mkdir -p logs/amend25_pod logs/amend26_pod
validate results/amend25_gate_sweep; validate results/amend26_tslot_sweep
PRE_REPORT="python -m experiments.amend25_gate_sweep openloop" bash logs/overnight/run_amend_pod.sh amend25 experiments.amend25_gate_sweep experiments/amend25_cells results/amend25_gate_sweep 150 > logs/amend25_pod/amend25_chain_pod.log 2>&1
log "amend25 chain exit $(cat /workspace/amend25_DONE /workspace/amend25_FAILED 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
PRE_REPORT="python -m experiments.amend26_tslot_sweep openloop" bash logs/overnight/run_amend_pod.sh amend26 experiments.amend26_tslot_sweep experiments/amend26_cells results/amend26_tslot_sweep 120 > logs/amend26_pod/amend26_chain_pod.log 2>&1
log "amend26 chain exit $(cat /workspace/amend26_DONE /workspace/amend26_FAILED 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
rm -f results/cache/macro_N100_cw1_11fc7f3ae5.npz* results/cache/macro_N100_cw1_a6378a47f8.npz* results/cache/macro_N100_cw1_fdbf07e4f1.npz* results/cache/macro_N100_cw1_6fbb7e451f.npz*
log "gate caches removed; disk $(du -sh /workspace | cut -f1)"
for N in 25 50 150 200; do
  NN=$(printf "%03d" $N)
  python -m experiments.amend21_N_crossover solve --N $N > logs/overnight/p1/resolve_N$NN.log 2>&1
  log "re-solve N=$N exit=$? $(tail -1 logs/overnight/p1/resolve_N$NN.log)"
done
NOWAIT=1 bash logs/overnight/p1/v4_chain_pod.sh > logs/overnight/p1/v4_chain_pod.log 2>&1
log "v4 chain exit $(cat /workspace/v4_DONE 2>/dev/null); disk $(du -sh /workspace | cut -f1)"
