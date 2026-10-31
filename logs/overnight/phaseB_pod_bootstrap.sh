#!/usr/bin/env bash
# Run LOCALLY after creating a fresh A100-80GB pod: $1 = ssh target (e.g. root@IP -p PORT given as "IP PORT").
# Pod key in /workspace/.ssh (pod restarts wipe ~/.ssh), registered on GitHub via gh, repo clone, venv (jax 0.12.0 cuda,
# as on the previous pods), logs/env.sh, then the G-pred chain detached in tmux "gpred".
set -e
IP=$1; PORT=$2; PUB=$(mktemp); S="ssh -o StrictHostKeyChecking=accept-new -p $PORT root@$IP"
$S 'mkdir -p /workspace/.ssh ~/.ssh; [ -f /workspace/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C musc-phaseB -f /workspace/.ssh/id_ed25519; cp /workspace/.ssh/id_ed25519* ~/.ssh/; chmod 600 ~/.ssh/id_ed25519; ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null'
$S 'cat /workspace/.ssh/id_ed25519.pub' > $PUB
gh ssh-key add $PUB --title "musc-phaseB pod (keep for next runs)" || true
$S 'set -e; cd /workspace; [ -d MUSC_MFG_PAPER ] || git clone -q git@github.com:kn-spa-i-blsps/MUSC_MFG_PAPER.git
  [ -d venv ] || python3 -m venv venv; . venv/bin/activate
  pip install -q -U pip; pip install -q "jax[cuda12]==0.12.0" optax scipy jaxopt numpy matplotlib imageio pyyaml pytest
  cd MUSC_MFG_PAPER; pip install -q -e . --no-deps
  printf ". /workspace/venv/bin/activate\nexport JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false\n" > logs/env.sh
  python -c "import jax; print(jax.devices())"
  command -v tmux >/dev/null || (apt-get -qq update && apt-get -qq install -y tmux)
  mkdir -p logs/overnight/phaseB; tmux new -d -s gpred "bash logs/overnight/phaseB_gpred_chain_pod.sh > logs/overnight/phaseB/chain.log 2>&1"
  echo chain launched'
