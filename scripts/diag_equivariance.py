"""
Fase 0 diagnostics D1 + D4 (docs/revisi/PLAN-01-DIAGNOSTICS.md).

D1 -- permutation equivariance. Relabel the 5 gNB in all 120 ways, un-permute the
output, and measure how far the policy moved. A permutation-equivariant policy gives
exactly the same per-node decision under every relabelling; a policy that reads node
identity does not. This is the one diagnostic in Fase 0 that is categorical rather than
marginal: variance is zero or it is not.

D4 -- trainable parameter count per algorithm, split backbone vs head for the GNN
variants. Without it, any performance difference can be explained by capacity instead of
by graph inductive bias, in either direction.

Both live in one script because neither needs a rollout: they are properties of the
loaded model, read off a single state.

Three things this script does that a naive version would get wrong:

1. **The state is taken after a warm-up, not at reset().** At reset every one of
   _last_sinr_embb, _prev_alloc, _prev_alloc_lag2, _viol_ewma, _last_backlog_bits and
   _queue_embb is zeroed (envs/network_slicing_env.py reset()), so 7 of the 8 observation
   columns are identical across nodes and only ch_gain varies. Permuting a nearly-uniform
   state is nearly a no-op, and every model would pass D1 for a reason that has nothing to
   do with equivariance.

2. **central-* is reported separately.** Those two broadcast one scalar to every gNB
   (scripts/evaluate_checkpoints.py select_actions), and a constant vector is invariant
   under un-permutation by construction. Zero action variance there would mean "the scalar
   was stable", not "the model is equivariant", so the pre-broadcast scalar is what gets
   reported for them.

3. **Pre-argmax scores are reported next to the action variance.** Float reduction order
   changes when nodes are permuted, so an argmax over near-tied logits can flip on the last
   bit and manufacture non-zero variance out of an equivariant model. The score deviation
   separates "not equivariant" from "tied".

D1 tests a mapping, not a KPI, so the P3 primary-readout rule does not apply: the main
pass is argmax. A sampled pass under a fixed per-permutation seed runs as a second check.

Usage:
  python scripts/diag_equivariance.py
  python scripts/diag_equivariance.py --checkpoints "results/logs/*_v4_seed42.pt" --warmup 50
"""
from __future__ import annotations
import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.network_slicing_env import NetworkSlicingEnv
from scripts.evaluate_checkpoints import EVAL_SEED_BASE, load_agent, select_actions
from scripts.rliable_report import parse_run_name

# These two consume the flattened global observation and broadcast one action to every
# gNB, so the un-permuted action vector is constant by construction.
CENTRAL_KINDS = {"mlp-dqn-central", "mlp-ppo-central"}


def permute_state(obs: np.ndarray, graph: dict, perm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Relabel nodes so that new index i carries old node perm[i].

    edge_attr is not touched: each edge keeps its own path loss, only its endpoint labels
    move. That is what makes this a relabelling of the same physical topology rather than a
    different topology.
    """
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    ei = np.asarray(graph["edge_index"])
    return obs[perm], {
        "x": np.asarray(graph["x"])[perm],
        "edge_index": inv[ei],
        "edge_attr": graph["edge_attr"],
    }


def unpermute(values: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Undo permute_state on a per-node output: row i of `values` belongs to old node
    perm[i]."""
    out = np.empty_like(values)
    out[perm] = values
    return out


def raw_scores(agent, kind: str, obs: np.ndarray, graph: dict) -> np.ndarray:
    """Pre-argmax logits (PPO) or Q-values (DQN), read through the same modules the action
    path uses. Shape (N, n_actions) for per-agent models, (n_actions,) for central ones."""
    with torch.no_grad():
        if kind == "gnn-dqn":
            return agent.q_values(graph).cpu().numpy()
        if kind == "gnn-ppo":
            return agent.actor(agent._embed(graph)).cpu().numpy()
        if kind == "mlp-knn-ppo":
            feats = torch.as_tensor(agent.features(graph), dtype=torch.float32).to(agent._device)
            return agent.actor(feats).cpu().numpy()
        flat = torch.as_tensor(obs.flatten(), dtype=torch.float32).to(agent._device)
        batch = torch.as_tensor(obs, dtype=torch.float32).to(agent._device)
        if kind == "mlp-dqn-central":
            return agent.q_values(flat).cpu().numpy()
        if kind == "mlp-ppo-central":
            return agent.actor(flat).cpu().numpy()
        if kind == "mlp-dqn":
            return agent.q_values(batch).cpu().numpy()
        return agent.actor(batch).cpu().numpy()  # mlp-ppo


def warm_up(env: NetworkSlicingEnv, agent, kind: str, steps: int, seed: int):
    """Run the policy forward so the observation carries the state that reset() zeroes.
    Returns the (obs, graph) the permutation test is run on."""
    obs, info = env.reset(seed=seed)
    for _ in range(steps):
        actions = select_actions(agent, kind, obs, info, env, greedy=True)
        obs, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
    return obs, info["graph"]


def d1_for_agent(env, agent, kind: str, warmup: int, greedy: bool) -> dict:
    obs, graph = warm_up(env, agent, kind, warmup, EVAL_SEED_BASE)
    n = obs.shape[0]

    actions, scores = [], []
    for perm_t in itertools.permutations(range(n)):
        perm = np.array(perm_t)
        obs_p, graph_p = permute_state(obs, graph, perm)
        # Same action noise for every permutation, so a difference is the relabelling
        # rather than sampling luck. Greedy is unaffected; this keeps the two passes equal.
        torch.manual_seed(EVAL_SEED_BASE)
        np.random.seed(EVAL_SEED_BASE)
        a_p = np.asarray(select_actions(agent, kind, obs_p, {"graph": graph_p}, env, greedy))
        s_p = raw_scores(agent, kind, obs_p, graph_p)
        if kind in CENTRAL_KINDS:
            # The broadcast vector is constant, so un-permuting it hides everything. What
            # actually varies is the single pre-broadcast decision.
            actions.append(a_p[:1])
            scores.append(s_p)
        else:
            actions.append(unpermute(a_p, perm))
            scores.append(unpermute(s_p, perm))

    act_arr = np.stack(actions)          # (120, N) or (120, 1) for central
    score_arr = np.stack(scores)         # (120, N, A) or (120, A)
    ref = score_arr[0]
    return {
        "n_perms": act_arr.shape[0],
        "action_var_max": float(np.var(act_arr, axis=0).max()),
        "action_n_distinct": int(len(np.unique(act_arr, axis=0))),
        "score_dev_max": float(np.abs(score_arr - ref).max()),
        "score_scale": float(np.abs(ref).max()),
    }


def d4_for_agent(agent) -> dict:
    total = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    backbone = 0
    if hasattr(agent, "backbone"):
        backbone = sum(p.numel() for p in agent.backbone.parameters() if p.requires_grad)
    return {"params_total": total, "params_backbone": backbone, "params_head": total - backbone}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/*_v4_seed42.pt")
    p.add_argument("--warmup", type=int, default=50,
                   help="policy steps before the state is frozen for the permutation test; "
                        "reset() zeroes 7 of the 8 observation columns, see module docstring")
    p.add_argument("--out", type=str, default="results/DIAG_EQUIVARIANCE.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(pt) for pt in glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    rows = []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, ckpt_algo, kind = load_agent(pt_path, device)
        env = NetworkSlicingEnv()
        env.cmdp_enabled = False
        row = {"algo": algo, "seed": seed, "kind": kind}
        for label, greedy in (("argmax", True), ("sampled", False)):
            d1 = d1_for_agent(env, agent, kind, args.warmup, greedy)
            for k, v in d1.items():
                row[f"{label}_{k}"] = v
        env.close()
        row.update(d4_for_agent(agent))
        rows.append(row)
        print(f"[{pt_path.stem}] argmax action_var_max={row['argmax_action_var_max']:.3e} "
              f"score_dev_max={row['argmax_score_dev_max']:.3e} params={row['params_total']:,}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}.csv", index=False)

    lines = [
        "# D1 permutation equivariance + D4 parameter count\n",
        f"Checkpoints: `{args.checkpoints}`. Warm-up: {args.warmup} policy steps from "
        f"`env.reset(seed={EVAL_SEED_BASE})` before the state is frozen.\n",
        "**Why the warm-up.** `reset()` zeroes `_last_sinr_embb`, `_prev_alloc`, "
        "`_prev_alloc_lag2`, `_viol_ewma`, `_last_backlog_bits` and `_queue_embb`, so 7 of "
        "the 8 observation columns are identical across gNB at t=0 and only `ch_gain` "
        "varies. Permuting a nearly-uniform state is nearly a no-op: every model would pass "
        "D1 for a reason unrelated to equivariance. The state is taken after the policy has "
        "driven the environment for the stated number of steps.\n",
        "**Readout.** D1 tests a mapping, not a KPI, so P3 does not apply -- the primary "
        "pass is argmax. The sampled pass is a second check under a fixed per-permutation "
        "seed, not a KPI reading. Do not compare these numbers against anything in "
        "`results/RLIABLE_*.md`.\n",
        "**Reading the two columns.** `action_var_max` is the largest per-gNB variance of "
        "the un-permuted action across all 120 relabellings. `score_dev_max` is the largest "
        "absolute deviation of the un-permuted pre-argmax logits/Q-values from the identity "
        "permutation, with `score_scale` for context. Float reduction order changes when "
        "nodes are permuted, so an argmax over near-tied scores can flip on the last bit: a "
        "non-zero `action_var_max` with a `score_dev_max` orders of magnitude below "
        "`score_scale` is a tie being broken differently, not a model reading node "
        "identity.\n",
        "**`central-dqn` / `central-ppo` rows are not comparable to the rest.** Those two "
        "consume the flattened global observation and broadcast one action to every gNB "
        "(`scripts/evaluate_checkpoints.py` `select_actions`), so the un-permuted action "
        "vector is constant by construction and would read as perfectly equivariant. Their "
        "rows report the single pre-broadcast decision instead.\n",
        "## D1\n",
        "| algo | kind | argmax action_var_max | argmax distinct action vectors | "
        "argmax score_dev_max | score_scale | sampled action_var_max |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        note = " *(pre-broadcast scalar)*" if r["kind"] in CENTRAL_KINDS else ""
        lines.append(
            f"| `{r['algo']}`{note} | {r['kind']} | {r['argmax_action_var_max']:.3e} | "
            f"{r['argmax_action_n_distinct']} | {r['argmax_score_dev_max']:.3e} | "
            f"{r['argmax_score_scale']:.4f} | {r['sampled_action_var_max']:.3e} |"
        )

    lines += [
        "\n## D4 trainable parameters\n",
        "PLAN-01 D4 requires this table in the paper whichever way it falls: if the counts "
        "are lopsided, any performance difference can be attributed to capacity rather than "
        "to graph inductive bias, in either direction.\n",
        "| algo | total | backbone (GNN) | head |",
        "|---|---|---|---|",
    ]
    for r in rows:
        bb = f"{r['params_backbone']:,}" if r["params_backbone"] else "--"
        lines.append(f"| `{r['algo']}` | {r['params_total']:,} | {bb} | {r['params_head']:,} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
