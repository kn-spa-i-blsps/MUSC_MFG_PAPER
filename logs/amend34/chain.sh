#!/bin/bash
# amend34 CPU chain (EXPLORATORY): one-seed reference checks first, then A/B seeds 0-4, 2 at a time, then report.
cd "$(dirname "$0")/../.."
L=logs/amend34; PY=.venv/bin/python
echo "start $(date -Is) HEAD $(git rev-parse --short HEAD)" >> $L/chain.log
jobs=$L/jobs.txt; : > $jobs
echo "check_real split:real/real 0" >> $jobs
echo "ab split:noise_only/noise_only 0" >> $jobs
for s in 0 1 2 3 4; do for arm in split:real/noise_only split:noise_only/real; do echo "ab $arm $s" >> $jobs; done; done
cat $jobs | xargs -P2 -L1 bash -c 'n=$(echo $1 | tr ":/" "_-"); $0 -m experiments.ladder --config experiments/amend34_cells/$1.yaml --mode eval --arm $2 --seeds $3 >> logs/amend34/${n}_s$3.log 2>&1; echo "done $1 $2 s$3 rc=$? $(date -Is)" >> logs/amend34/chain.log' $PY
$PY -m experiments.amend34_layer_delay_cpu report > $L/report.log 2>&1
echo "report rc=$? $(date -Is)" >> $L/chain.log
touch $L/DONE
