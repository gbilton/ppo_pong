#!/usr/bin/env bash
# One-shot setup for a fresh training box (Ubuntu/Debian/Fedora, CPU-only).
#
#   1. provision VPS (prioritize CPU cores; 8+ recommended; no GPU needed)
#   2. git clone <repo> ~/src/ppo_pong && cd ~/src/ppo_pong && ./bootstrap_vps.sh
#   3. copy training state from the old box (or skip for a fresh run):
#        rsync -a oldbox:~/src/ppo_pong/runs/ ~/src/ppo_pong/runs/
#        rsync -a oldbox:~/src/ppo_pong/tmp/  ~/src/ppo_pong/tmp/
#   4. ./bootstrap_vps.sh start        # tmux: training(--resume) + tb + dashboard
#
# Access from your laptop:
#   ssh -L 8787:localhost:8787 -L 6006:localhost:6006 <vps>
#   dashboard: http://localhost:8787   tensorboard: http://localhost:6006
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-setup}" = "setup" ]; then
    if ! command -v tmux >/dev/null; then
        echo ">> installing tmux (needs sudo)"
        sudo apt-get install -y tmux 2>/dev/null || sudo dnf install -y tmux
    fi
    if ! command -v "$HOME/.local/bin/uv" >/dev/null && ! command -v uv >/dev/null; then
        echo ">> installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    echo ">> creating venv + installing deps (CPU torch)"
    [ -d .venv ] || "$UV" venv -p 3.11
    "$UV" pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$UV" pip install -r requirements.txt
    echo ">> setup done. copy runs/ and tmp/ from the old box, then: $0 start"

elif [ "$1" = "start" ]; then
    if [ -f runs/latest/checkpoint.pt ]; then
        TRAIN_CMD=".venv/bin/python main_vec.py --resume"
        echo ">> resuming $(readlink runs/latest)"
    else
        TRAIN_CMD=".venv/bin/python main_vec.py --num-envs 32"
        echo ">> no checkpoint found - starting a fresh run"
    fi
    tmux has-session -t ppo 2>/dev/null || tmux new-session -d -s ppo \
        "cd $PWD && $TRAIN_CMD >> train_vec.log 2>&1"
    tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb \
        "while true; do $PWD/.venv/bin/tensorboard --logdir $PWD/runs --port 6006 --load_fast=false >> $PWD/tb.log 2>&1; sleep 3; done"
    tmux has-session -t ui 2>/dev/null || tmux new-session -d -s ui \
        "cd $PWD && .venv/bin/python dashboard.py >> dashboard.log 2>&1"
    sleep 3 && tmux ls
    echo ">> tunnel from laptop: ssh -L 8787:localhost:8787 -L 6006:localhost:6006 <this-host>"

elif [ "$1" = "stop" ]; then
    tmux kill-session -t ppo 2>/dev/null || true   # checkpoints every ~20s; safe
    echo ">> training stopped (tb/ui left running; kill-session -t tb/ui if wanted)"
fi
