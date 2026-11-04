#!/bin/bash
# amend36 CPU chain: 2 macro solves (terminal goal, c_w=1 and c_w=0) in parallel, 15 rollouts 2 at a time
# (seed-major), report; push after each stage (push_lib checkpoint).
cd "$(dirname "$0")/../.."
L=logs/amend36; PY=.venv/bin/python; R=results/amend36_terminal_goal
export MUSC_LOCK=$PWD/.git/musc_chain.lock
source logs/overnight/push_lib.sh
echo "start $(date -Is) HEAD $(git rev-parse --short HEAD)" >> $L/chain.log
for arm in fresh fresh_nomfg; do
  $PY -c "import yaml;from experiments.ladder import setup;setup(yaml.safe_load(open('experiments/amend36_cells/$arm.yaml')));print('solve ok')" > $L/solve_$arm.log 2>&1 &
done
wait
grep -q "solve ok" $L/solve_fresh.log && grep -q "solve ok" $L/solve_fresh_nomfg.log || { echo "SOLVE FAILED $(date -Is)" >> $L/chain.log; touch $L/FAILED; exit 1; }
echo "solves done $(date -Is)" >> $L/chain.log
checkpoint "amend36 macro solves done (terminal cost = goal potential, c_w 1 and 0); not a result" $L/*.log
jobs=$L/jobs.txt; : > $jobs
for s in 0 1 2 3 4; do for arm in fresh noise_only fresh_nomfg; do
  c=fresh; [ $arm = noise_only ] && c=noise_only; echo "$arm $c $s" >> $jobs; done; done
cat $jobs | xargs -P2 -L1 bash -c '[ -f '$R'/$1/$2/metrics_seed$3.json ] && exit 0; $0 -m experiments.ladder --config experiments/amend36_cells/$1.yaml --mode eval --arm $2 --seeds $3 >> logs/amend36/$1_s$3.log 2>&1; echo "done $1 s$3 rc=$? $(date -Is)" >> logs/amend36/chain.log' $PY
checkpoint "amend36 rollouts done (15 = 3 delay-free arms x seeds 0-4); not a result until the report" $R/*/*/metrics_seed*.json $L/*.log $L/jobs.txt
$PY -m experiments.amend36_terminal_goal report > $L/report.log 2>&1
echo "report rc=$? $(date -Is)" >> $L/chain.log
checkpoint "amend36 report: MFG terminal cost = goal potential, link 1 (delay-free) verdict in report.json" $R/report.json $L/*.log
touch $L/DONE
