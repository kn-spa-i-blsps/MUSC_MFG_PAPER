#!/usr/bin/env bash
# Run LOCALLY once a pod is up: $1 = IP, $2 = ssh PORT. Works for the old pod (volume with repo/venv: key restore, pull)
# and a fresh pod (key, gh ssh-key add, clone, venv jax 0.12.0 cuda12). GPU smoke test (devices + highest-precision
# matmul + tiny solve import) aborts before any queue start. Then queue_B_pod.sh detached in tmux "queueB".
set -e
IP=$1; PORT=$2; PUB=$(mktemp); S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT root@$IP"
$S 'mkdir -p /workspace/.ssh ~/.ssh; [ -f /workspace/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C musc-queueB -f /workspace/.ssh/id_ed25519; cp /workspace/.ssh/id_ed25519* ~/.ssh/; chmod 600 ~/.ssh/id_ed25519; ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null'
$S 'cat /workspace/.ssh/id_ed25519.pub' > $PUB
gh ssh-key list | grep -qF "$(cut -d' ' -f2 $PUB)" || gh ssh-key add $PUB --title "musc-queueB pod (keep for next runs)"
$S 'set -e; cd /workspace; [ -d MUSC_MFG_PAPER ] || git clone -q git@github.com:kn-spa-i-blsps/MUSC_MFG_PAPER.git
  [ -d venv ] || { python3 -m venv venv; . venv/bin/activate; pip install -q -U pip; pip install -q "jax[cuda12]==0.12.0" optax scipy jaxopt numpy matplotlib imageio pyyaml pytest; }
  . venv/bin/activate; cd MUSC_MFG_PAPER; git config user.name Gio; git config user.email mikojasiek@gmail.com
  git pull -q --rebase --autostash origin main; pip install -q -e . --no-deps
  printf ". /workspace/venv/bin/activate\nexport JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false\n" > logs/env.sh
  . logs/env.sh; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import jax, jax.numpy as jnp; jax.config.update(\"jax_default_matmul_precision\",\"highest\"); x=jnp.ones((2048,2048)); print(jax.devices(), float((x@x).sum()))"
  python -c "import experiments.ladder, experiments.amend31_sigma_cbf; print(\"imports ok\")"
  command -v tmux >/dev/null || (apt-get -qq update && apt-get -qq install -y tmux)
  tmux has-session -t queueB 2>/dev/null && { echo "queueB already running"; exit 0; }
  tmux new -d -s queueB "bash logs/overnight/queue_B_pod.sh > logs/overnight/queue_B.log 2>&1"; echo "queue launched $(git rev-parse --short HEAD)"'
