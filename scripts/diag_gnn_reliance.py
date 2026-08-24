"""
Fase 0 diagnostics D2a, D2b, D3 (docs/revisi/PLAN-01-DIAGNOSTICS.md).

Does the policy head actually use the GNN output, or has it learned to ignore it?
PLAN-01 makes D2 the gate for PLAN-04: implementing an auxiliary loss without evidence of
representation collapse adds complexity for a problem that may not exist.

  D2a  zero the neighbour messages entirely, measure the KPI change
  D2b  keep the messages, destroy which neighbour each one came from
  D3   pairwise cosine similarity of the final node embeddings (over-smoothing)

D2c (gradient-norm ratio) is deliberately not here. The PLAN-01 gate table keys the
PLAN-04 decision off D2a and D2b, both pure checkpoint evaluation. If those two disagree
and D2c is genuinely needed, it must be measured with a short instrumented training run
reading `p.grad` after `agent.learn()` -- not a rollout with a re-written loss path. For a
diagnostic that decides a gate, fidelity to the real training path beats cheapness.

Four things carried over from scripts/attention_analysis.py, because the problem is the
same one:

1. Three KPIs, not embb_p5_mbps alone. At this operating point most checkpoints already
   sit at the cell-edge floor (collapse rate 4-5 of 5 seeds, results/STABILITY_v4_stoch.md)
   and a KPI pinned near 1e-6 cannot degrade however much the ablation changes. Reporting
   it alone would turn "no headroom" into a false "the GNN does not matter".
2. Both arms draw the same action noise (torch.manual_seed per episode), so the difference
   is the ablation rather than sampling luck.
3. The primary readout differs by family -- sampled for PPO (P3), argmax for DQN
   (determination of 2026-08-16). Families are never pooled (Gate C3).
4. Every mutation is a context manager that restores what it touched.

Usage:
  python scripts/diag_gnn_reliance.py
  python scripts/diag_gnn_reliance.py --checkpoints "results/logs/gnn-*_v4_seed42.pt" --episodes 3
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from gnn import BACKBONES
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, run_episode
from scripts.diag_equivariance import warm_up
from scripts.rliable_report import parse_run_name

ABLATION_KPIS = ["embb_p5_mbps", "timely_throughput_mbps", "sla_satisfaction_pct"]


@contextmanager
def graph_transform(env: NetworkSlicingEnv, fn):
    """Rewrite the graph the env hands to the policy, without touching envs/.

    `_get_graph_dict` is the single place the graph is built (called from both reset() and
    step()), so wrapping it covers every path the policy can see. PLAN-01 forbids editing
    training code in this phase; this keeps the mutation on the diagnostic's side of the
    line and puts it back on exit.
    """
    orig = env._get_graph_dict
    env._get_graph_dict = lambda: fn(orig())
    try:
        yield
    finally:
        env._get_graph_dict = orig


@contextmanager
def no_pyg_self_loops(backbone):
    """GATv2Conv adds its own self-loops with fill_value='mean'. Fed an edge set that is
    only self-loops it first removes them, then re-adds them over an empty edge_attr, whose
    mean is NaN. Turning the flag off and supplying explicit self-loops with a zero
    attribute is what makes the D2a arm well-defined for the GAT backbone."""
    convs = [c for c in (getattr(backbone, "conv1", None), getattr(backbone, "conv2", None))
             if c is not None and hasattr(c, "add_self_loops")]
    saved = [c.add_self_loops for c in convs]
    for c in convs:
        c.add_self_loops = False
    try:
        yield
    finally:
        for c, orig in zip(convs, saved):
            c.add_self_loops = orig


def strip_neighbours(graph: dict, attention: bool) -> dict:
    """D2a: no neighbour messages reach any node.

    GAT needs explicit self-loops -- attention softmaxes over a node's incoming edges, and
    with none at all the layer has nothing to normalise. SAGE keeps a separate root weight,
    so an empty edge set already means "self only"; adding self-loops there would count the
    node's own features twice and understate the ablation.
    """
    n = np.asarray(graph["x"]).shape[0]
    if attention:
        idx = np.arange(n, dtype=np.int64)
        return {"x": graph["x"],
                "edge_index": np.stack([idx, idx]),
                "edge_attr": np.zeros((n, 1), dtype=np.float32)}
    return {"x": graph["x"],
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, 1), dtype=np.float32)}


def shuffle_edge_attr(graph: dict, rng: np.random.Generator) -> dict:
    """D2b: keep every message, destroy which neighbour it describes.

    The inter-gNB graph is fully connected (envs/channel_model.py build_interference_graph
    emits every ordered pair), so every receiving node has the identical neighbour set and
    permuting source labels within a destination group changes nothing. The only structure
    left to destroy is the edge-to-attribute pairing. This preserves the multiset of inputs
    and breaks their physical meaning -- which is what "magnitude preserved, structure
    destroyed" can mean on a complete graph, and the limit is worth stating out loud: on
    this topology D2b tests sensitivity to edge *information*, not to *topology*.
    """
    ei = np.asarray(graph["edge_index"])
    ea = np.asarray(graph["edge_attr"]).copy()
    for d in np.unique(ei[1]):
        idx = np.where(ei[1] == d)[0]
        ea[idx] = ea[rng.permutation(idx)]
    return {"x": graph["x"], "edge_index": ei, "edge_attr": ea}


def mean_offdiag_cosine(mat: np.ndarray) -> float:
    """Mean pairwise cosine similarity between rows, diagonal excluded."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.maximum(norms, 1e-12)
    sim = unit @ unit.T
    n = sim.shape[0]
    return float(sim[~np.eye(n, dtype=bool)].mean())


def load_gnn_agent(pt_path: Path, device: torch.device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    algo = ckpt["algo"]
    backbone = BACKBONES[ckpt["backbone"]]()
    if algo.startswith("gnn-madqn"):
        agent, kind = DQNAgent(backbone).to(device), "gnn-dqn"
    else:
        agent, kind = PPOAgent(backbone).to(device), "gnn-ppo"
    agent.load_state_dict(ckpt["state_dict"])
    # Same fix evaluate_checkpoints.load_agent applies: epsilon is not in state_dict, so a
    # freshly built DQN sits at 1.0 and every non-greedy reading would be uniform random.
    if hasattr(agent, "epsilon"):
        agent.epsilon = agent.epsilon_min
    agent.eval()
    return agent, algo, kind, ckpt["backbone"]


def kpi_arms(env, agent, kind, backbone, backbone_name, episodes, greedy, seed_base):
    """Normal / D2a / D2b KPI means for one checkpoint."""
    attention = hasattr(backbone, "conv1") and hasattr(backbone.conv1, "add_self_loops")
    normal, ablated, shuffled = [], [], []
    for ep in range(episodes):
        eval_seed = seed_base + ep
        torch.manual_seed(eval_seed)
        normal.append(run_episode(env, agent, kind, seed=eval_seed, greedy=greedy))

        torch.manual_seed(eval_seed)
        with no_pyg_self_loops(backbone), \
                graph_transform(env, lambda g: strip_neighbours(g, attention)):
            ablated.append(run_episode(env, agent, kind, seed=eval_seed, greedy=greedy))

        if backbone_name == "gat":
            rng = np.random.default_rng(eval_seed)
            torch.manual_seed(eval_seed)
            with graph_transform(env, lambda g: shuffle_edge_attr(g, rng)):
                shuffled.append(run_episode(env, agent, kind, seed=eval_seed, greedy=greedy))

    out = {}
    for kpi in ABLATION_KPIS:
        out[f"{kpi}_normal"] = float(np.mean([m[kpi] for m in normal]))
        out[f"{kpi}_d2a"] = float(np.mean([m[kpi] for m in ablated]))
        out[f"{kpi}_d2b"] = float(np.mean([m[kpi] for m in shuffled])) if shuffled else float("nan")
    return out


def d3_for_agent(env, agent, kind, backbone, warmup) -> dict:
    """Cosine similarity of the final node embeddings, next to the same statistic on the
    raw observation. The embedding number alone decides nothing: if the inputs are already
    near-identical, near-identical outputs are not the GNN's doing."""
    obs, graph = warm_up(env, agent, kind, warmup, EVAL_SEED_BASE)
    with torch.no_grad():
        h = backbone(graph["x"], graph["edge_index"], graph["edge_attr"]).cpu().numpy()
    return {"cos_embedding": mean_offdiag_cosine(h),
            "cos_obs_reference": mean_offdiag_cosine(np.asarray(obs, dtype=np.float64))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/gnn-*_v4_seed4*.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--warmup", type=int, default=50, help="steps before the D3 state is read")
    p.add_argument("--out", type=str, default="results/DIAG_GNN_RELIANCE.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(pt) for pt in glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    rows = []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, ckpt_algo, kind, backbone_name = load_gnn_agent(pt_path, device)
        backbone = agent.backbone
        # Primary readout by family: sampled for PPO (P3), argmax for DQN (2026-08-16).
        greedy = "dqn" in algo
        env = NetworkSlicingEnv()
        env.cmdp_enabled = False
        row = {"algo": algo, "seed": seed, "backbone": backbone_name,
               "readout": "argmax" if greedy else "sampled"}
        row.update(kpi_arms(env, agent, kind, backbone, backbone_name,
                            args.episodes, greedy, EVAL_SEED_BASE))
        row.update(d3_for_agent(env, agent, kind, backbone, args.warmup))
        env.close()
        rows.append(row)
        print(f"[{pt_path.stem}] readout={row['readout']} "
              f"timely {row['timely_throughput_mbps_normal']:.3f} -> "
              f"D2a {row['timely_throughput_mbps_d2a']:.3f} / "
              f"D2b {row['timely_throughput_mbps_d2b']:.3f}  "
              f"cos_emb={row['cos_embedding']:.4f} (obs {row['cos_obs_reference']:.4f})")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}.csv", index=False)

    lines = [
        "# D2 GNN reliance + D3 over-smoothing\n",
        f"Checkpoints: `{args.checkpoints}`. Episodes per arm: {args.episodes}, seeds from "
        f"`EVAL_SEED_BASE = {EVAL_SEED_BASE}`.\n",
        "**Readout is per family, never pooled** (Gate C3): sampled for PPO (P3, frozen "
        "2026-08-08), argmax for DQN (determination of 2026-08-16). Each row states its "
        "own. All three arms of a row draw the same action noise "
        "(`torch.manual_seed(eval_seed)` before each), so a difference is the ablation and "
        "not sampling luck.\n",
        "**Three KPIs, not `embb_p5_mbps` alone.** At this operating point most checkpoints "
        "already sit at the cell-edge floor and a KPI pinned near 1e-6 cannot degrade "
        "however much the ablation changes; reporting it alone would turn *no headroom* "
        "into a false *the GNN does not matter*. `timely_throughput_mbps` and "
        "`sla_satisfaction_pct` still have headroom, so that is where a real effect has to "
        "show up.\n",
        "## D2a -- neighbour messages zeroed\n",
        "GAT gets explicit self-loops with a zero edge attribute and PyG's own self-loop "
        "insertion turned off: fed an edge set that is only self-loops, `GATv2Conv` removes "
        "them and re-adds them with `fill_value='mean'` over an empty `edge_attr`, whose "
        "mean is NaN. SAGE gets an empty edge set instead -- `SAGEConv` keeps a separate "
        "root weight, so empty already means *self only*, and adding self-loops there would "
        "count a node's own features twice and understate the ablation.\n",
        "## D2b -- edge attributes shuffled within each receiving node\n",
        "**A limit of this topology, not of the test.** `build_interference_graph` "
        "(`envs/channel_model.py`) emits every ordered pair of gNB, so the graph is "
        "complete: every receiving node has the identical neighbour set, and permuting "
        "source labels within a destination group is a no-op. What is destroyed here is the "
        "edge-to-attribute pairing. On a complete graph D2b therefore tests sensitivity to "
        "edge *information*, not to *topology*.\n",
        "`sage` rows are **N/A**, not zero: `SAGEConv` never reads `edge_attr`, so there is "
        "nothing for this arm to perturb. Writing 0 would read as *the model ignored it*.\n",
        "| algo | seed | backbone | readout | KPI | normal | D2a zeroed | D2b shuffled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for kpi in ABLATION_KPIS:
            d2b = r[f"{kpi}_d2b"]
            d2b_s = "N/A" if np.isnan(d2b) else f"{d2b:.6f}"
            lines.append(
                f"| `{r['algo']}` | {r['seed']} | {r['backbone']} | {r['readout']} | "
                f"`{kpi}` | {r[f'{kpi}_normal']:.6f} | {r[f'{kpi}_d2a']:.6f} | {d2b_s} |"
            )

    for kpi in ABLATION_KPIS:
        base = np.array([r[f"{kpi}_normal"] for r in rows])
        for arm in ("d2a", "d2b"):
            delta = np.array([r[f"{kpi}_normal"] - r[f"{kpi}_{arm}"] for r in rows])
            ok = ~np.isnan(delta)
            if not ok.any():
                continue
            rel = np.abs(delta[ok]) / np.maximum(np.abs(base[ok]), 1e-12)
            lines.append(
                f"\n`{kpi}` / {arm.upper()}: largest absolute change "
                f"{np.max(np.abs(delta[ok])):.6f} ({np.max(rel) * 100:.3f}% of the "
                f"un-ablated value), median {np.median(delta[ok]):+.6f}, checkpoints moved "
                f"by >1% of their own value: {int(np.sum(rel > 0.01))}/{int(ok.sum())}."
            )

    lines += [
        "\n## D3 -- over-smoothing\n",
        "Mean pairwise cosine similarity of the final node embeddings, with the same "
        "statistic on the raw observation alongside. The embedding number alone decides "
        "nothing: if the inputs are already near-identical, near-identical outputs are not "
        "the GNN's doing. State read after "
        f"{args.warmup} policy steps, because `reset()` zeroes 7 of the 8 observation "
        "columns.\n",
        "**The inter-gNB graph is complete, so its diameter is 1** -- one layer already "
        "reaches every node and the two configured layers aggregate the identical "
        "neighbour set twice. That raises the over-smoothing risk rather than lowering it, "
        "the opposite of what PLAN-01 D3 and PLAN-03 §4 assume when they call the diameter "
        "*small*.\n",
        "| algo | seed | backbone | cos(embedding) | cos(obs) reference |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['algo']}` | {r['seed']} | {r['backbone']} | "
                     f"{r['cos_embedding']:.4f} | {r['cos_obs_reference']:.4f} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
