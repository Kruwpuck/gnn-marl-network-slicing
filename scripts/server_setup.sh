#!/bin/bash
# Server setup + training runner for Linux cloud server (Lambda Labs, Vast.ai, RunPod, etc.)
# Assumes: fresh Ubuntu 20.04/22.04, CUDA 12.x driver already installed
# Usage: bash scripts/server_setup.sh

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Install uv ─────────────────────────────────────────────────────────
log "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# ── 2. Clone repo ─────────────────────────────────────────────────────────
# TODO: ganti URL ini dengan repo kamu
REPO_URL="https://github.com/YOUR_USERNAME/gnn-marl-network-slicing.git"
REPO_DIR="gnn-marl-network-slicing"

if [ ! -d "$REPO_DIR" ]; then
    log "Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 3. Install dependencies ───────────────────────────────────────────────
log "Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

log "Installing PyTorch + CUDA 12.4..."
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-geometric==2.8.0
uv pip install numpy==1.26.4 scipy==1.13.0 gymnasium==1.1.0 pettingzoo==1.25.0 pyyaml==6.0.2

log "Verifying CUDA..."
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — check driver!')"

# ── 4. Quick test ─────────────────────────────────────────────────────────
log "Running test suite..."
python -m pytest tests/ -q --tb=short

# ── 5. Launch training in tmux ────────────────────────────────────────────
mkdir -p results/logs

log "Starting training in tmux session 'train'..."
tmux new-session -d -s train \
    "source .venv/bin/activate && bash scripts/run_all_sequential.sh 2>&1 | tee training_run.log"

echo ""
echo "======================================================"
echo "  Training started in tmux session 'train'"
echo "  Monitor:   tmux attach -t train"
echo "  Detach:    Ctrl+B then D"
echo "  Live logs: tail -f results/logs/gnn-mappo_gat_seed42.csv"
echo "======================================================"
