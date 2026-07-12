#!/usr/bin/env bash
# ONE-STEP training-box setup. On a fresh VPS (Ubuntu/Debian/Fedora, CPU-only):
#
#   curl -fsSL https://raw.githubusercontent.com/gbilton/ppo_pong/main/bootstrap_vps.sh | bash
#
# Clones the repo (which carries the latest training-state snapshot - run
# snapshot_push.sh before retiring the old box), installs everything, and
# resumes training + tensorboard + dashboard in tmux.
#
# Access from your laptop:
#   ssh -L 8787:localhost:8787 -L 6006:localhost:6006 <vps>
#   dashboard: http://localhost:8787   tensorboard: http://localhost:6006
#
# Also usable inside an existing checkout: ./bootstrap_vps.sh [setup|start|stop]
set -euo pipefail

REPO_URL="https://github.com/gbilton/ppo_pong"
REPO_DIR="$HOME/src/ppo_pong"

if [ ! -f "$(dirname "$0")/main_vec.py" ] 2>/dev/null || [ "${0}" = "bash" ] || [ "${0}" = "-bash" ]; then
    # piped via curl, or run outside a checkout: clone first
    if ! command -v git >/dev/null; then
        sudo apt-get install -y git 2>/dev/null || sudo dnf install -y git
    fi
    [ -d "$REPO_DIR/.git" ] || git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
else
    cd "$(dirname "$0")"
fi

ACTION="${1:-all}"

setup() {
    if ! command -v tmux >/dev/null; then
        echo ">> installing tmux (needs sudo)"
        sudo apt-get install -y tmux 2>/dev/null || sudo dnf install -y tmux
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    if [ ! -x "$UV" ]; then
        echo ">> installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        UV="$HOME/.local/bin/uv"
    fi
    echo ">> creating venv + installing deps (CPU torch)"
    [ -d .venv ] || "$UV" venv -p 3.11
    "$UV" pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$UV" pip install -r requirements.txt
}

start() {
    if [ -f runs/latest/checkpoint.pt ]; then
        TRAIN_CMD=".venv/bin/python main_vec.py --resume"
        echo ">> resuming $(readlink runs/latest)"
    else
        TRAIN_CMD=".venv/bin/python main_vec.py --num-envs 32"
        echo ">> no checkpoint found - starting a fresh run"
    fi
    # nice -19: on a shared box the other services keep absolute priority;
    # on a dedicated box it changes nothing
    tmux has-session -t ppo 2>/dev/null || tmux new-session -d -s ppo \
        "cd $PWD && nice -n 19 $TRAIN_CMD >> train_vec.log 2>&1"
    tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb \
        "while true; do $PWD/.venv/bin/tensorboard --logdir $PWD/runs --port 6006 --load_fast=false >> $PWD/tb.log 2>&1; sleep 3; done"
    tmux has-session -t ui 2>/dev/null || tmux new-session -d -s ui \
        "cd $PWD && .venv/bin/python dashboard.py >> dashboard.log 2>&1"
    sleep 3 && tmux ls
    echo ">> tunnel from laptop: ssh -L 8787:localhost:8787 -L 6006:localhost:6006 <this-host>"
}

case "$ACTION" in
    all) setup && start ;;
    setup) setup ;;
    start) start ;;
    stop) tmux kill-session -t ppo 2>/dev/null || true
          echo ">> training stopped (checkpointed; tb/ui left running)" ;;
esac
