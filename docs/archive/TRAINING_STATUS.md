# Training Status — GNN-MARL Network Slicing

**Last updated:** 2026-06-26  
**Branch:** `feat/phase-0-environment`  
**Hardware used:** RTX 4060 Laptop (8GB), CPU bottleneck → stopped early

---

## Implementation Status

| Component | Status |
|---|---|
| Environment (`envs/`) | ✅ Complete |
| GNN Backbones: GAT, SAGE, GCN (`gnn/`) | ✅ Complete |
| Agents: DQN (Dueling), PPO (`agents/`) | ✅ Complete |
| Baseline Agents: MLP-DQN, MLP-PPO (`agents/mlp_agent.py`) | ✅ Complete |
| Training scripts (`training/`) | ✅ Complete + CUDA |
| Evaluation scripts (`evaluation/`) | ✅ Complete |
| Ablation scripts (`ablation/`) | ✅ Complete |
| Test suite (70 tests) | ✅ All green |

---

## Training Progress (stopped 2026-06-26)

Results saved in `results/logs/`. Target: 1,000,000 steps each.

| Algorithm | Steps Done | % | Notes |
|---|---|---|---|
| `ippo` | 771,583 | **77%** | Near done |
| `central-ppo` | 747,007 | **75%** | Near done |
| `gnn-mappo/sage` | 263,167 | **26%** | Needs ~750K more |
| `gnn-mappo/gat` | 144,383 | **14%** | Needs ~856K more |
| `central-dqn` | 11,799 | **1%** | Basically fresh start |
| `idqn` | 9,599 | **1%** | Basically fresh start |
| `gnn-madqn/gat` | 2,799 | **<1%** | Basically fresh start |
| `gnn-madqn/sage` | 5,399 | **<1%** | Basically fresh start |

> **Root cause of slowness:** 8 parallel jobs on 4-core CPU → CPU bottleneck. Env simulation (channel model, traffic) runs on CPU. GPU only used for GNN forward/backward. PPO fast (vectorized rollouts). DQN slow (per-step replay buffer, CPU-heavy).

---

## Estimated Training Time by Hardware

| Hardware | DQN 1M steps | PPO 1M steps | Strategy |
|---|---|---|---|
| RTX 4060 Laptop (4-core) | ~100h | ~1.5h | Not practical for DQN |
| **Colab T4** (4 vCPU, 16GB GPU) | ~8h | ~45min | 1 job at a time |
| **Colab A100** (12 vCPU, 40GB GPU) | ~2h | ~15min | 2-3 parallel |
| Cloud 8-core + V100 | ~4h | ~20min | All 4 parallel |
| Cloud 16-core + A100 | ~1.5h | ~8min | All parallel |

> GPU matters less than CPU count for DQN — env is CPU-bound.

---

## What Still Needs Running

### PPO — resume (near done)
```bash
python training/train_proposed.py  --algo gnn-mappo --backbone gat  --steps 1000000 --seed 42
python training/train_proposed.py  --algo gnn-mappo --backbone sage --steps 1000000 --seed 42
python training/train_baselines.py --algo ippo                      --steps 1000000 --seed 42
python training/train_baselines.py --algo central-ppo               --steps 1000000 --seed 42
```

### DQN — fresh (200K sufficient for convergence comparison)
```bash
python training/train_proposed.py  --algo gnn-madqn --backbone gat  --steps 200000 --seed 42
python training/train_proposed.py  --algo gnn-madqn --backbone sage --steps 200000 --seed 42
python training/train_baselines.py --algo idqn                      --steps 200000 --seed 42
python training/train_baselines.py --algo central-dqn               --steps 200000 --seed 42
```

### Evaluation — after training
```bash
python evaluation/convergence_eval.py  --log-dir results/logs
python evaluation/network_perf_eval.py
python evaluation/zero_shot_eval.py
python ablation/backbone_ablation.py   --ablation backbone   --steps 200000
python ablation/backbone_ablation.py   --ablation reward     --steps 200000
python ablation/backbone_ablation.py   --ablation depth      --steps 200000
python ablation/backbone_ablation.py   --ablation burstiness --steps 200000
```

---

## Notes

- Training scripts write `.pt` checkpoint only at end — current partial runs have NO `.pt` files
- CSV logs contain full episode history; resuming means rerunning from step 0 (scripts don't support mid-run resume)
- See `scripts/colab_training.py` and `scripts/server_setup.sh` for ready-to-run guides
