#!/usr/bin/env bash
# Run on the WORKSTATION while the amend39 pod is up: pull the raw-trajectory archive and verify it against the pod's
# ARCHIVE_MANIFEST.txt. Usage: bash logs/overnight/amend39_pull_archive.sh <ssh-host> <ssh-port> [dest]
# Only after "VERIFIED" may the pod volume be deleted (prereg amend39, archive section).
set -euo pipefail
HOST=$1; PORT=$2; DEST=${3:-$HOME/MUSC_archive/amend39_settled_regen}
SRC=/workspace/MUSC_MFG_PAPER/results/amend39_settled_regen/
mkdir -p "$DEST"
rsync -av --partial -e "ssh -p $PORT" \
  --include='*/' --include='traj_seed*.npz' --include='settled_seed*.json' --include='metrics_seed*.json' \
  --include='ARCHIVE_MANIFEST.txt' --include='report.json' --include='identity.json' --exclude='*' \
  "$HOST:$SRC" "$DEST/"
cd "$DEST" && shasum -a 256 -c --quiet ARCHIVE_MANIFEST.txt && echo "VERIFIED $(wc -l < ARCHIVE_MANIFEST.txt) files in $DEST"
