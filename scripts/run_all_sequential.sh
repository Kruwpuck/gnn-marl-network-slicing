#!/bin/bash
# Run all 8 training jobs sequentially — safe for any machine (no CPU overload).
# Usage: bash scripts/run_all_sequential.sh
# Override steps: STEPS_PPO=500000 STEPS_DQN=200000 bash scripts/run_all_sequential.sh

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS_PPO=${STEPS_PPO:-1000000}
STEPS_DQN=${STEPS_DQN:-200000}
SEED=${SEED:-42}

mkdir -p results/logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== GNN-MARL Training: Sequential Mode ==="
log "PPO steps: $STEPS_PPO | DQN steps: $STEPS_DQN | seed: $SEED"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# ── PPO variants (fast, run first) ────────────────────────────────────────
log "1/8  gnn-mappo/gat  (${STEPS_PPO} steps)"
python training/train_proposed.py  --algo gnn-mappo --backbone gat  --steps "$STEPS_PPO" --seed "$SEED"

log "2/8  gnn-mappo/sage (${STEPS_PPO} steps)"
python training/train_proposed.py  --algo gnn-mappo --backbone sage --steps "$STEPS_PPO" --seed "$SEED"

log "3/8  ippo           (${STEPS_PPO} steps)"
python training/train_baselines.py --algo ippo                      --steps "$STEPS_PPO" --seed "$SEED"

log "4/8  central-ppo    (${STEPS_PPO} steps)"
python training/train_baselines.py --algo central-ppo               --steps "$STEPS_PPO" --seed "$SEED"

# ── DQN variants (CPU-heavy — use fewer steps) ───────────────────────────
log "5/8  gnn-madqn/gat  (${STEPS_DQN} steps)"
python training/train_proposed.py  --algo gnn-madqn --backbone gat  --steps "$STEPS_DQN" --seed "$SEED"

log "6/8  gnn-madqn/sage (${STEPS_DQN} steps)"
python training/train_proposed.py  --algo gnn-madqn --backbone sage --steps "$STEPS_DQN" --seed "$SEED"

log "7/8  idqn           (${STEPS_DQN} steps)"
python training/train_baselines.py --algo idqn                      --steps "$STEPS_DQN" --seed "$SEED"

log "8/8  central-dqn    (${STEPS_DQN} steps)"
python training/train_baselines.py --algo central-dqn               --steps "$STEPS_DQN" --seed "$SEED"

log "=== All training done. Results in results/logs/ ==="
