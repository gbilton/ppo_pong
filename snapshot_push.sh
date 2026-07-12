#!/usr/bin/env bash
# Refresh the in-repo training-state snapshot and push.
# Run from the machine with git access, BEFORE the training box goes away:
#   ./snapshot_push.sh [training-host]     # default host: fedora
set -euo pipefail
cd "$(dirname "$0")"
HOST="${1:-ppo-vps}"

echo ">> pulling state from $HOST"
rsync -a "$HOST:~/src/ppo_pong/runs" .
rsync -a "$HOST:~/src/ppo_pong/tmp/docker/models" \
         "$HOST:~/src/ppo_pong/tmp/docker/legacy_models" \
         "$HOST:~/src/ppo_pong/tmp/docker/goat_seed" \
         "$HOST:~/src/ppo_pong/tmp/docker/goat_v2_seed" tmp/docker/

echo ">> committing snapshot"
git add runs/latest runs/ppo_vec_accel tmp/docker/models \
        tmp/docker/legacy_models tmp/docker/goat_seed tmp/docker/goat_v2_seed
git commit -m "training-state snapshot $(date +%Y-%m-%d_%H:%M)" || echo "(no changes)"
git push
echo ">> done - a fresh VPS can now continue from this exact state"
