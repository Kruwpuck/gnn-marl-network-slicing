"""
Attention-vs-interference mechanism analysis (Fase 1 #4.3, docs/rev2-implementation-plan.md).

Two parts, reported together (correlation alone is decoration -- Rev 2 SS RQ6 + Caveat 2):
1. Correlation: GATv2Conv layer-2 attention weight per edge vs raw path-loss edge_attr
   (env-native dB scale, NOT divided by 100 -- see gnn/base_backbone.py to_tensors()).
2. Causal ablation: force attention uniform over each node's neighbors during greedy
   eval (zero the learned `att` parameter -> softmax degenerates to 1/degree), measure
   embb_p5_mbps degradation vs normal attention.

Only *_gat checkpoints have attention (SAGE doesn't).

Usage:
  python scripts/attention_analysis.py
  python scripts/attention_analysis.py --episodes 10 --out results/ATTENTION.md
"""
from __future__ import annotations
import argparse
import glob
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from gnn import BACKBONES
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from scripts.evaluate_checkpoints import EVAL_SEED_BASE
from scripts.rliable_report import parse_run_name


@contextmanager
def uniform_attention(backbone):
    """Zero the learned `att` param of both GATv2Conv layers -> softmax degenerates
    to uniform over each node's neighbors. Restores original weights on exit."""
    convs = [backbone.conv1, backbone.conv2]
    saved = [c.att.data.clone() for c in convs]
    for c in convs:
        c.att.data.zero_()
    try:
        yield
    finally:
        for c, orig in zip(convs, saved):
            c.att.data.copy_(orig)


def load_gat_agent(pt_path: Path, device: torch.device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    algo = ckpt["algo"]
    backbone = BACKBONES[ckpt["backbone"]]()
    if algo.startswith("gnn-madqn"):
        agent = DQNAgent(backbone).to(device)
        kind = "dqn"
    else:
        agent = PPOAgent(backbone).to(device)
        kind = "ppo"
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent, kind


def act(agent, kind: str, graph: dict):
    if kind == "dqn":
        return agent.act(graph, greedy=True)
    actions, _, _ = agent.act(graph, greedy=True)
    return actions


def run_episode(env, agent, kind: str, backbone, seed: int, capture_attention: bool, uniform: bool) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    embb_bps: list[float] = []
    corr_alpha: list[float] = []
    corr_pathloss: list[float] = []

    while not done:
        graph = info["graph"]
        if capture_attention:
            with torch.no_grad():
                _, attn_layers = backbone(graph["x"], graph["edge_index"], graph["edge_attr"], return_attention=True)
            edge_index2, alpha2 = attn_layers[1]  # layer 2: single head, no-concat -- most interpretable
            edge_index2 = edge_index2.cpu().numpy()
            alpha2 = alpha2.detach().cpu().numpy().reshape(-1)
            raw_ei = np.asarray(graph["edge_index"])
            raw_ea = np.asarray(graph["edge_attr"]).reshape(-1)
            # lookup by (src,dst) pair, not position -- GATv2Conv adds self-loops by
            # default so the returned edge_index is longer than the original and may
            # reorder edges; self-loops (src==dst) have no physical path-loss, skip them
            pathloss_lookup = {(int(s), int(d)): float(pl) for s, d, pl in zip(raw_ei[0], raw_ei[1], raw_ea)}
            for k in range(edge_index2.shape[1]):
                s, d = int(edge_index2[0, k]), int(edge_index2[1, k])
                if s == d:
                    continue
                pl = pathloss_lookup.get((s, d))
                if pl is not None:
                    corr_alpha.append(float(alpha2[k]))
                    corr_pathloss.append(pl)

        if uniform:
            with uniform_attention(backbone):
                actions = act(agent, kind, graph)
        else:
            actions = act(agent, kind, graph)

        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        embb_bps.extend(np.asarray(info["embb_rates"], dtype=np.float64).tolist())

    embb_arr = np.asarray(embb_bps) if embb_bps else np.array([0.0])
    return {
        "embb_p5_mbps": float(np.percentile(embb_arr, 5) / 1e6),
        "corr_alpha": corr_alpha,
        "corr_pathloss": corr_pathloss,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/gnn-*_gat_seed*.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default="results/ATTENTION.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(pt) for pt in sorted(glob.glob(args.checkpoints)) if "_floor" not in Path(pt).stem]

    rows = []
    all_alpha: list[float] = []
    all_pathloss: list[float] = []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, kind = load_gat_agent(pt_path, device)
        backbone = agent.backbone

        env = NetworkSlicingEnv(config_path=args.config)
        env.cmdp_enabled = False

        normal_p5, uniform_p5 = [], []
        for ep in range(args.episodes):
            eval_seed = EVAL_SEED_BASE + ep
            m_normal = run_episode(env, agent, kind, backbone, eval_seed, capture_attention=True, uniform=False)
            m_uniform = run_episode(env, agent, kind, backbone, eval_seed, capture_attention=False, uniform=True)
            normal_p5.append(m_normal["embb_p5_mbps"])
            uniform_p5.append(m_uniform["embb_p5_mbps"])
            all_alpha.extend(m_normal["corr_alpha"])
            all_pathloss.extend(m_normal["corr_pathloss"])
        env.close()

        rows.append({
            "algo": algo, "seed": seed,
            "embb_p5_normal": float(np.mean(normal_p5)),
            "embb_p5_uniform_attn": float(np.mean(uniform_p5)),
        })
        print(f"[{pt_path.stem}] normal={np.mean(normal_p5):.6f}  uniform_attn={np.mean(uniform_p5):.6f}")

    if len(all_alpha) > 1:
        pearson_r, pearson_p = pearsonr(all_alpha, all_pathloss)
        spearman_r, spearman_p = spearmanr(all_alpha, all_pathloss)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    out_path = Path(args.out)
    df = pd.DataFrame(rows)
    df.to_csv(out_path.parent / "attention_summary.csv", index=False)

    lines = [
        "# Attention vs Interference Mechanism\n",
        f"n edges pooled across all episodes/seeds: {len(all_alpha)}\n",
        f"Pearson r = {pearson_r:.4f} (p={pearson_p:.4g}), Spearman r = {spearman_r:.4f} "
        f"(p={spearman_p:.4g}) -- attention weight (layer 2, single head) vs raw path-loss "
        f"dB (env-native scale, not the /100 scaled value fed to the model).\n",
        "**Causal ablation** (mandatory): attention forced uniform over neighbors (zero "
        "the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to "
        "1/degree) during greedy eval. Correlation alone would be decoration without this.\n",
        "| algo | seed | embb_p5 normal | embb_p5 uniform-attn | degradation |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        deg = r["embb_p5_normal"] - r["embb_p5_uniform_attn"]
        lines.append(
            f"| `{r['algo']}` | {r['seed']} | {r['embb_p5_normal']:.6f} "
            f"| {r['embb_p5_uniform_attn']:.6f} | {deg:+.6f} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
