#!/bin/bash
# amend33 CPU chain: c_w=0 solve alone, then 21 rollouts 2 at a time (seed-major), then report + commit.
cd "$(dirname "$0")/../.."
L=logs/amend33; PY=.venv/bin/python
echo "start $(date -Is)" >> $L/chain.log
# 1) decoupled (c_w=0) macro solve: a 0-seed-free way = setup() of the nomfg cell
$PY -c "import yaml;from experiments.ladder import setup;setup(yaml.safe_load(open('experiments/amend33_cells/fresh_nomfg.yaml')));print('solve ok')" >> $L/solve_nomfg.log 2>&1 || echo "SOLVE FAILED" >> $L/chain.log
echo "solve done $(date -Is)" >> $L/chain.log
# 2) jobs, seed-major so every arm has seeds if the night ends early
jobs=$L/jobs.txt; : > $jobs
echo "identity_real real 0" >> $jobs
for s in 0 1 2 3 4; do for arm in fresh fresh_nomfg fresh_nocbf noise_only; do
  c=fresh; [ $arm = noise_only ] && c=noise_only; echo "$arm $c $s" >> $jobs; done; done
cat $jobs | xargs -P2 -L1 bash -c '$0 -m experiments.ladder --config experiments/amend33_cells/$1.yaml --mode eval --arm $2 --seeds $3 >> logs/amend33/$1_s$3.log 2>&1; echo "done $1 s$3 rc=$? $(date -Is)" >> logs/amend33/chain.log' $PY
# 3) report + commit (small files only)
$PY -m experiments.amend33_n100_baseline_cpu report > $L/report.log 2>&1
git add -f results/amend33_n100_baseline_cpu/report.json results/amend33_n100_baseline_cpu/*/*/metrics_seed*.json $L/*.log $L/jobs.txt
git commit -q -m "amend33 result (DONE): CPU baseline validation + layer ablations at N=100 (amend11 config), seeds 0-4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "committed $(date -Is)" >> $L/chain.log
touch $L/DONE
