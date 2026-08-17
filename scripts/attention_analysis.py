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
from scripts.evaluate_checkpoints import run_episode as eval_episode
from scripts.rliable_report import parse_run_name

# The ablation is measured on these, not on embb_p5_mbps alone: in wave v4 nearly every
# checkpoint's cell-edge throughput is already at the floor (collapse rate 4-5 of 5 seeds,
# results/STABILITY_v4_stoch.md), and a KPI pinned at ~1e-6 cannot show degradation no
# matter what the ablation does. Reporting only embb_p5 would turn "no headroom" into a
# false "attention does not matter".
ABLATION_KPIS = ["embb_p5_mbps", "timely_throughput_mbps", "sla_satisfaction_pct"]


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
        kind = "gnn-dqn"      # eval-side naming, so evaluate_checkpoints can drive it too
    else:
        agent = PPOAgent(backbone).to(device)
        kind = "gnn-ppo"
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent, kind


def act(agent, kind: str, graph: dict, greedy: bool):
    if kind == "gnn-dqn":
        return agent.act(graph, greedy=greedy)
    actions, _, _ = agent.act(graph, greedy=greedy)
    return actions


def per_node_rho(alpha: list[float], pathloss: list[float], dst: list[int]) -> list[float]:
    """Spearman rho between attention and path-loss *within each receiving node's own
    neighbourhood*, one value per node per step.

    The pooled correlation this script also reports mixes edges from nodes with different
    attention scales and different neighbour sets. A node that consistently prefers its
    strongest interferer can be invisible in that pool. Grouping by destination node is
    where the mechanism claim actually lives: the softmax is normalised per receiving node.
    """
    by_dst: dict[int, tuple[list[float], list[float]]] = {}
    for a, pl, d in zip(alpha, pathloss, dst):
        acc = by_dst.setdefault(d, ([], []))
        acc[0].append(a)
        acc[1].append(pl)
    out = []
    for a_vals, pl_vals in by_dst.values():
        if len(a_vals) < 3 or len(set(a_vals)) < 2 or len(set(pl_vals)) < 2:
            continue  # rank correlation undefined on constants or on 2 points
        rho, _ = spearmanr(a_vals, pl_vals)
        if not np.isnan(rho):
            out.append(float(rho))
    return out


def run_episode(env, agent, kind: str, backbone, seed: int, capture_attention: bool,
                uniform: bool, greedy: bool) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    embb_bps: list[float] = []
    corr_alpha: list[float] = []
    corr_pathloss: list[float] = []
    node_rhos: list[float] = []

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
            step_alpha, step_pl, step_dst = [], [], []
            for k in range(edge_index2.shape[1]):
                s, d = int(edge_index2[0, k]), int(edge_index2[1, k])
                if s == d:
                    continue
                pl = pathloss_lookup.get((s, d))
                if pl is not None:
                    step_alpha.append(float(alpha2[k]))
                    step_pl.append(pl)
                    step_dst.append(d)
            corr_alpha.extend(step_alpha)
            corr_pathloss.extend(step_pl)
            node_rhos.extend(per_node_rho(step_alpha, step_pl, step_dst))

        if uniform:
            with uniform_attention(backbone):
                actions = act(agent, kind, graph, greedy)
        else:
            actions = act(agent, kind, graph, greedy)

        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        embb_bps.extend(np.asarray(info["embb_rates"], dtype=np.float64).tolist())

    embb_arr = np.asarray(embb_bps) if embb_bps else np.array([0.0])
    return {
        "embb_p5_mbps": float(np.percentile(embb_arr, 5) / 1e6),
        "corr_alpha": corr_alpha,
        "corr_pathloss": corr_pathloss,
        "node_rhos": node_rhos,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/gnn-*_gat_v4_seed*.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--config", type=str, default=None,
                   help="e.g. configs/generated/floor_none_area_size1000.0_n_gnb20.yaml to "
                        "test the mechanism where node degree is 19 instead of 4")
    p.add_argument("--out", type=str, default="results/ATTENTION_v4.md")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the argmax -- P3 primary readout. "
                        "The ablation measures a KPI, so it inherits the same protocol as "
                        "every other KPI reading.")
    args = p.parse_args()
    greedy = not args.stochastic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(pt) for pt in sorted(glob.glob(args.checkpoints)) if "_floor" not in Path(pt).stem]

    rows = []
    all_alpha: list[float] = []
    all_pathloss: list[float] = []
    all_node_rhos: list[float] = []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, kind = load_gat_agent(pt_path, device)
        backbone = agent.backbone

        env = NetworkSlicingEnv(config_path=args.config)
        env.cmdp_enabled = False

        normal: list[dict] = []
        uniform: list[dict] = []
        for ep in range(args.episodes):
            eval_seed = EVAL_SEED_BASE + ep
            # Attention capture only needs the forward pass, so it runs on its own; the KPI
            # arms go through evaluate_checkpoints.run_episode, the same code path that
            # produced every other KPI in this project.
            torch.manual_seed(eval_seed)
            captured = run_episode(env, agent, kind, backbone, eval_seed,
                                   capture_attention=True, uniform=False, greedy=greedy)
            all_alpha.extend(captured["corr_alpha"])
            all_pathloss.extend(captured["corr_pathloss"])
            all_node_rhos.extend(captured["node_rhos"])

            # Both arms draw the same action noise, otherwise the difference would include
            # sampling luck. Greedy runs are unaffected; seeding keeps the protocols equal.
            torch.manual_seed(eval_seed)
            normal.append(eval_episode(env, agent, kind, seed=eval_seed, greedy=greedy))
            torch.manual_seed(eval_seed)
            with uniform_attention(backbone):
                uniform.append(eval_episode(env, agent, kind, seed=eval_seed, greedy=greedy))
        env.close()

        row = {"algo": algo, "seed": seed}
        for kpi in ABLATION_KPIS:
            row[f"{kpi}_normal"] = float(np.mean([m[kpi] for m in normal]))
            row[f"{kpi}_uniform_attn"] = float(np.mean([m[kpi] for m in uniform]))
        rows.append(row)
        print(f"[{pt_path.stem}] " + "  ".join(
            f"{kpi}: {row[f'{kpi}_normal']:.6f} -> {row[f'{kpi}_uniform_attn']:.6f}"
            for kpi in ABLATION_KPIS))

    if len(all_alpha) > 1:
        pearson_r, pearson_p = pearsonr(all_alpha, all_pathloss)
        spearman_r, spearman_p = spearmanr(all_alpha, all_pathloss)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    rho_arr = np.asarray(all_node_rhos) if all_node_rhos else np.array([np.nan])
    out_path = Path(args.out)
    df = pd.DataFrame(rows)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}_summary.csv", index=False)

    readout = "stochastic (P3 primary)" if args.stochastic else "greedy (reported, never gates)"
    lines = [
        "# Attention vs Interference Mechanism\n",
        f"Readout: `{readout}`. Checkpoints: `{args.checkpoints}`. "
        f"Config: `{args.config or 'configs/experiment_config.yaml (n_gnb=5)'}`.\n",
        f"n edges pooled across all episodes/seeds: {len(all_alpha)}\n",
        f"**Pooled**: Pearson r = {pearson_r:.4f} (p={pearson_p:.4g}), Spearman r = "
        f"{spearman_r:.4f} (p={spearman_p:.4g}) -- attention weight (layer 2, single head) "
        f"vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the "
        f"model).\n",
        f"**Per receiving node** ({len(all_node_rhos)} node-steps with a defined rank "
        f"correlation): mean rho = {np.nanmean(rho_arr):.4f}, median = "
        f"{np.nanmedian(rho_arr):.4f}, fraction rho < 0 = "
        f"{float(np.mean(rho_arr < 0)):.3f}. The pooled number mixes edges from nodes with "
        "different attention scales and different neighbour sets, which can hide a node "
        "that consistently prefers its strongest interferer; the softmax is normalised per "
        "receiving node, so that is the level the mechanism claim lives at. A negative rho "
        "means more attention on lower path-loss, i.e. on the stronger interferer -- the "
        "direction the mechanism story predicts.\n",
        "**Causal ablation** (mandatory): attention forced uniform over neighbors (zero "
        "the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to "
        "1/degree). Both arms draw the same action noise (`torch.manual_seed` per episode) "
        "so the difference is the ablation, not sampling luck. Correlation alone would be "
        "decoration without this.\n",
        "Reported on three KPIs, not on `embb_p5_mbps` alone: at this operating point most "
        "checkpoints already sit at the cell-edge floor, and a KPI pinned near zero cannot "
        "degrade however much the ablation changes. `timely_throughput_mbps` and "
        "`sla_satisfaction_pct` still have headroom, so they are where a real causal effect "
        "would have to show up.\n",
        "| algo | seed | KPI | normal | uniform-attn | degradation |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        for kpi in ABLATION_KPIS:
            normal_v, uniform_v = r[f"{kpi}_normal"], r[f"{kpi}_uniform_attn"]
            lines.append(
                f"| `{r['algo']}` | {r['seed']} | `{kpi}` | {normal_v:.6f} "
                f"| {uniform_v:.6f} | {normal_v - uniform_v:+.6f} |"
            )

    # State whether the ablation moved anything, computed rather than asserted.
    for kpi in ABLATION_KPIS:
        deltas = np.array([r[f"{kpi}_normal"] - r[f"{kpi}_uniform_attn"] for r in rows])
        base = np.array([r[f"{kpi}_normal"] for r in rows])
        rel = np.abs(deltas) / np.maximum(np.abs(base), 1e-12)
        lines.append(
            f"\n`{kpi}`: largest absolute change {np.max(np.abs(deltas)):.6f} "
            f"({np.max(rel) * 100:.3f}% of the un-ablated value), median change "
            f"{np.median(deltas):+.6f}, checkpoints moved by >1% of their own value: "
            f"{int(np.sum(rel > 0.01))}/{len(rows)}."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
