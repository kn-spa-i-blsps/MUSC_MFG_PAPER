#!/usr/bin/env bash
# Overnight sequence on the pod: identity gate for the logging-only rollout change (n_near), then amend23, then amend24.
cd /workspace/MUSC_MFG_PAPER
. logs/env.sh
L=logs/overnight/identity_nnear
mkdir -p $L
log() { echo "$(date '+%F %T') $*"; }
log "identity gate start HEAD $(git rev-parse --short HEAD)"
for N in 25 100; do
  NN=$(printf "%03d" $N)
  python - $NN <<'PY'
import sys, yaml
NN = sys.argv[1]
c = yaml.safe_load(open(f"experiments/amend21_cells/N{NN}_real.yaml")); c["run_name"] = f"overnight_identity_nnear/N{NN}"
yaml.safe_dump(c, open(f"logs/overnight/identity_nnear/N{NN}_real.yaml", "w"), sort_keys=False)
PY
  python -m experiments.ladder --config $L/N${NN}_real.yaml --mode eval --arm real --seeds 0 > $L/eval_N$NN.log 2>&1
done
python - > $L/identity.json <<'PY'
import json
K = ["completeness_final", "collision_pair_steps", "min_sep_overall_m", "p5_sep_worst_step_m", "cbf_intervention_rate",
     "slack_active_rate", "closedloop_xhat_mse_near", "closedloop_xhat_mse_all"]
out = {}
for NN in ["025", "100"]:
    a = json.load(open(f"results/amend21_N_crossover/N{NN}/real/real/metrics_seed0.json"))
    b = json.load(open(f"results/overnight_identity_nnear/N{NN}/real/metrics_seed0.json"))
    out[NN] = {"equal": all(a[k] == b[k] for k in K), "pairw": b.get("closedloop_xhat_mse_near_pairw"),
               "near_pair_steps": b.get("near_pair_steps"), **{k: [a[k], b[k]] for k in K}}
out["all_equal"] = all(v["equal"] for v in out.values())
print(json.dumps(out, indent=1))
PY
ok=$(python -c "import json; print(json.load(open('$L/identity.json'))['all_equal'])")
log "identity (n_near logging) vs amend21 real seed 0 N=25,100: all_equal=$ok"
git add -f $L/*.json $L/*.log $L/*.yaml logs/overnight/night_chain_pod.log 2>/dev/null
git commit -q -m "overnight identity gate for n_near logging: all_equal=$ok

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; git pull -q --rebase --autostash origin main && git push -q origin HEAD:main
[ "$ok" = True ] || { log "identity FAILED -> stop"; exit 1; }
bash logs/overnight/run_amend_pod.sh amend23 experiments.amend23_margin_pooled experiments/amend23_cells results/amend23_margin_pooled 80 > logs/amend23_pod/amend23_chain_pod.log 2>&1
log "amend23 chain exit $(cat /workspace/amend23_DONE /workspace/amend23_FAILED 2>/dev/null)"
bash logs/overnight/run_amend_pod.sh amend24 experiments.amend24_mechanism_control experiments/amend24_cells results/amend24_mechanism_control 70 > logs/amend24_pod/amend24_chain_pod.log 2>&1
log "amend24 chain exit $(cat /workspace/amend24_DONE /workspace/amend24_FAILED 2>/dev/null)"
