"""
Fase 0 diagnostic D6 -- where does the node representation collapse?

D3 (scripts/diag_gnn_reliance.py) reported cosine 1.0000 on the final `gat` embeddings
against 0.7955-0.9844 on the raw observation, and PLAN-03 section 5 (residual/JK) was
ordered first on the strength of that gap. D6 tests the premise underneath it: if the eight
observation columns already fail to tell the five gNB apart, then residual/JK is attacking a
symptom in the wrong place and PLAN-03 sections 2 and 7 come first instead.

Why the cosine numbers cannot answer that on their own:

  * All eight observation columns are non-negative (envs/network_slicing_env.py:537-539:
    -pl/100, clipped SINR, log1p queues, an EWMA, allocation fractions). Every node vector
    sits in the positive orthant, so cosine is lifted BY CONSTRUCTION rather than by the
    nodes being alike. `conv2` has no activation (gnn/gat_backbone.py:52), so the embeddings
    are free to be signed. 0.910 against 1.0000 compares two different scales.
  * Mean-centering does not rescue it. Subtracting the across-node mean forces sum_i v_i = 0,
    hence sum_{i!=j} <v_i, v_j> = -sum_i ||v_i||^2 -- the mean off-diagonal inner product is
    forced negative, with a floor near -1/(n-1) = -0.25 at n=5. A value near 1 cannot occur,
    so the instrument cannot answer the question being asked.

So two scale-free statistics are used instead, computed identically at every stage:

  rel_spread = ||X - mean_over_nodes(X)||_F / ||X||_F     0 = nodes identical
  eff_rank   = exp(entropy of the normalised singular values of the centred X)

Raw cosine is reported alongside, so the numbers stay continuous with the committed D3 table
rather than replacing it.

Stages are captured with hooks on the real forward pass, not by re-implementing it: a
forward hook on conv1 gives the pre-activation, a forward PRE-hook on conv2 gives the
post-activation that conv1 actually handed on, and the return value is the embedding. The
two backbones differ (ELU + edge_attr for gat, ReLU and no edge_attr for sage), and a
re-implementation here would be free to drift from gnn/.

Usage:
  python scripts/diag_input_separability.py
  python scripts/diag_input_separability.py --checkpoints "results/logs/gnn-*_v4_seed42.pt"
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
from scripts.evaluate_checkpoints import EVAL_SEED_BASE
from scripts.diag_equivariance import warm_up
from scripts.diag_gnn_reliance import load_gnn_agent, mean_offdiag_cosine
from scripts.rliable_report import parse_run_name

# Order fixed by envs/network_slicing_env.py:537-539. Named here so the per-column table
# says which observation went flat, not just "column 5".
OBS_COLUMNS = ["ch_gain", "sinr_embb", "sinr_urllc", "q_embb",
               "urllc_backlog", "viol_ewma", "prev_alloc_lag2", "prev_alloc"]

STAGES = ["input", "conv1_pre", "conv1_act", "conv2"]

# Declared before the run, with the sensitivity pair, exactly as the D5 collapse threshold
# was. A column counts as varying if its across-node spread is at least 1% of its own scale;
# a checkpoint's input counts as degenerate below SPREAD_MAIN.
VARY_REL = 0.01
SPREAD_MAIN = 0.05
SPREAD_SENSITIVITY = (0.02, 0.10)


def rel_spread(mat: np.ndarray) -> float:
    """Across-node spread as a fraction of the matrix's own magnitude. Scale-free, so the
    same number is comparable between an 8-column observation and a 64-column embedding."""
    total = np.linalg.norm(mat)
    if total < 1e-12:
        return 0.0
    return float(np.linalg.norm(mat - mat.mean(axis=0, keepdims=True)) / total)


def eff_rank(mat: np.ndarray) -> float:
    """exp(Shannon entropy of the normalised singular values) of the centred matrix.

    Returns 0.0 when the nodes are exactly identical -- the centred matrix is then all
    zeros and there is no varying direction at all. That is a real reading, not missing
    data, and it cannot be confused with a genuine value: the smallest of those is 1.0.
    """
    centred = mat - mat.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    total = s.sum()
    if total < 1e-12:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@contextmanager
def stage_capture(backbone):
    """Read conv1's pre-activation and post-activation off the real forward pass. The
    pre-hook on conv2 is what makes the post-activation faithful: it is literally the tensor
    conv1's activation handed to conv2, whichever activation that is."""
    seen: dict[str, np.ndarray] = {}
    handles = [
        backbone.conv1.register_forward_hook(
            lambda m, inp, out: seen.__setitem__("conv1_pre", out.detach().cpu().numpy())),
        backbone.conv2.register_forward_pre_hook(
            lambda m, inp: seen.__setitem__("conv1_act", inp[0].detach().cpu().numpy())),
    ]
    try:
        yield seen
    finally:
        for h in handles:
            h.remove()


def stats_for_checkpoint(env, agent, kind, backbone, warmup):
    obs, graph = warm_up(env, agent, kind, warmup, EVAL_SEED_BASE)
    x = np.asarray(graph["x"], dtype=np.float64)
    with torch.no_grad(), stage_capture(backbone) as seen:
        h = backbone(graph["x"], graph["edge_index"], graph["edge_attr"]).cpu().numpy()
    return {"input": x,
            "conv1_pre": seen["conv1_pre"].astype(np.float64),
            "conv1_act": seen["conv1_act"].astype(np.float64),
            "conv2": h.astype(np.float64)}, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, default="results/logs/gnn-*_v4_seed*.pt")
    p.add_argument("--warmup", type=int, default=50,
                   help="steps before the state is read; reset() zeroes 7 of the 8 columns")
    p.add_argument("--out", type=str, default="results/DIAG_INPUT_SEPARABILITY.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(pt) for pt in glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    rows, skipped = [], []
    for pt_path in paths:
        algo, _, seed = parse_run_name(pt_path.stem)
        agent, _, kind, backbone_name = load_gnn_agent(pt_path, device)
        backbone = agent.backbone
        if not (hasattr(backbone, "conv1") and hasattr(backbone, "conv2")):
            skipped.append((pt_path.stem, backbone_name))
            continue

        env = NetworkSlicingEnv()
        env.cmdp_enabled = False
        mats, x = stats_for_checkpoint(env, agent, kind, backbone, args.warmup)
        env.close()

        col_std = x.std(axis=0)
        col_rel = col_std / np.maximum(np.abs(x.mean(axis=0)), 1e-12)
        base = {"algo": algo, "seed": seed, "backbone": backbone_name}
        for stage in STAGES:
            row = dict(base, stage=stage,
                       rel_spread=rel_spread(mats[stage]),
                       eff_rank=eff_rank(mats[stage]),
                       cos_raw=mean_offdiag_cosine(mats[stage]))
            if stage == "input":
                row["n_cols_nonconst"] = int((col_std > 1e-6).sum())
                row["n_cols_varying"] = int(((col_std > 1e-6) & (col_rel > VARY_REL)).sum())
                for name, v in zip(OBS_COLUMNS, col_rel):
                    row[f"relstd_{name}"] = float(v)
            rows.append(row)

        inp = rows[-len(STAGES)]
        print(f"[{pt_path.stem}] input spread={inp['rel_spread']:.4f} "
              f"rank={inp['eff_rank']:.2f} cols={inp['n_cols_varying']}/8  ->  "
              f"conv2 spread={rows[-1]['rel_spread']:.4f} rank={rows[-1]['eff_rank']:.2f}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}.csv", index=False)

    inputs = df[df.stage == "input"]
    n_ckpt = len(inputs)

    def stage_line(backbone_name: str, stage: str) -> str:
        sub = df[(df.backbone == backbone_name) & (df.stage == stage)]
        return (f"| `{backbone_name}` | `{stage}` | {sub.rel_spread.median():.4f} | "
                f"{sub.rel_spread.min():.4f} | {sub.rel_spread.max():.4f} | "
                f"{sub.eff_rank.median():.2f} | {sub.cos_raw.median():.4f} | n={len(sub)} |")

    lines = [
        "# D6 -- where the node representation collapses\n",
        f"Checkpoints: `{args.checkpoints}` ({n_ckpt} matched). State read after "
        f"{args.warmup} policy steps, because `reset()` zeroes 7 of the 8 observation "
        "columns and every model would otherwise look identical at t=0.\n",
        "**Why not cosine.** The eight observation columns are all non-negative "
        "(`envs/network_slicing_env.py:537-539`), so every node vector sits in the positive "
        "orthant and cosine is lifted by construction; `conv2` has no activation "
        "(`gnn/gat_backbone.py:52`), so the embeddings are free to be signed. Comparing "
        "0.910 against 1.0000 compares two different scales. Mean-centring does not fix it "
        "either: it forces `sum_i v_i = 0`, hence a negative mean off-diagonal inner "
        "product with a floor near `-1/(n-1) = -0.25` at n=5, so a value near 1 cannot "
        "occur. Two scale-free statistics are used instead. Raw cosine is kept alongside so "
        "the numbers stay continuous with the committed D3 table.\n",
        "- `rel_spread` = `||X - mean_over_nodes(X)||_F / ||X||_F`. **0 means the nodes are "
        "identical.**",
        "- `eff_rank` = `exp(entropy of the normalised singular values)` of the centred "
        "matrix. **0.0 means exactly identical nodes** -- the centred matrix is all zeros "
        "and there is no varying direction. It cannot be confused with a real value; the "
        "smallest of those is 1.0.\n",
        "**Stages come from the real forward pass**, captured with a forward hook on "
        "`conv1` (pre-activation), a forward *pre*-hook on `conv2` (the post-activation "
        "`conv1` actually handed on), and the return value. The two backbones differ -- ELU "
        "and `edge_attr` for `gat`, ReLU and none for `sage` -- and re-implementing the "
        "forward here would be free to drift from `gnn/`.\n",
        "| backbone | stage | rel_spread median | min | max | eff_rank median | cos_raw median | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for backbone_name in sorted(df.backbone.unique()):
        for stage in STAGES:
            lines.append(stage_line(backbone_name, stage))

    lines += [
        "\n## Which observation columns tell the gNB apart\n",
        "Across-node standard deviation of each column, relative to that column's own mean "
        f"(median over {n_ckpt} checkpoints). A column counts as *varying* at "
        f"`{VARY_REL}` -- 1% of its own scale.\n",
        "| column | relative std, median | varying in |",
        "|---|---|---|",
    ]
    for name in OBS_COLUMNS:
        col = inputs[f"relstd_{name}"]
        lines.append(f"| `{name}` | {col.median():.4f} | "
                     f"{int((col > VARY_REL).sum())}/{n_ckpt} checkpoints |")
    lines.append(
        f"\nColumns varying per checkpoint: median {inputs.n_cols_varying.median():.0f} of 8, "
        f"range {inputs.n_cols_varying.min()}-{inputs.n_cols_varying.max()}. Non-constant "
        f"(std > 1e-6): median {inputs.n_cols_nonconst.median():.0f} of 8.\n")

    below = int((inputs.rel_spread < SPREAD_MAIN).sum())
    sens = " ".join(f"At `{thr}`: {int((inputs.rel_spread < thr).sum())}/{n_ckpt}."
                    for thr in SPREAD_SENSITIVITY)
    lines += [
        "## Verdict\n",
        "Decision rule, written before the run: the input is degenerate if "
        f"`rel_spread < {SPREAD_MAIN}` **and** at most one column varies -- in which case "
        "PLAN-03 sections 2 and 7 come first and section 5 is deferred. If the input spread "
        "is alive and collapses at `conv1`/`conv2`, the GNN is what collapses it and the "
        "current order (section 5 first) stands.\n",
        f"Checkpoints with a degenerate input at `{SPREAD_MAIN}`: **{below}/{n_ckpt}**. {sens}\n",
        "The threshold is reported at three values because a verdict that moves between "
        "them is a verdict driven by the threshold, and that would itself be the finding.\n",
        "### Which layer does it\n",
        "Ratio of `rel_spread` between consecutive stages, per checkpoint, median. A ratio "
        "far below 1 means that stage is where the separation was lost.\n",
        "| backbone | input -> conv1_pre | conv1_pre -> conv1_act | conv1_act -> conv2 |",
        "|---|---|---|---|",
    ]
    wide = df.pivot_table(index=["algo", "seed", "backbone"], columns="stage",
                          values="rel_spread")
    for backbone_name in sorted(df.backbone.unique()):
        sub = wide.xs(backbone_name, level="backbone")
        ratios = [(sub[b] / sub[a].where(sub[a] > 1e-12)).median()
                  for a, b in zip(STAGES, STAGES[1:])]
        lines.append(f"| `{backbone_name}` | " + " | ".join(f"{r:.4f}" for r in ratios) + " |")

    dead = [name for name in OBS_COLUMNS
            if int((inputs[f"relstd_{name}"] > VARY_REL).sum()) <= n_ckpt // 10]
    if dead:
        lines.append(
            "\n### Observation columns the policy itself flattened\n\n"
            + ", ".join(f"`{d}`" for d in dead)
            + f" vary in at most {n_ckpt // 10} of {n_ckpt} checkpoints. These are the "
            "previous allocation and its lag (`envs/network_slicing_env.py:538`), so the "
            "reading is that every gNB picked the SAME action -- the same lockstep D5 "
            "measured as `mode_share` 1.0000, now visible in the observation itself. The "
            "policy's own degenerate output feeds back as node features that carry no node "
            "identity, which is a loop no document anticipated: it is both a symptom of the "
            "collapse and an input to it.\n")

    if skipped:
        lines.append("**Skipped** (no `conv1`/`conv2` to read): "
                     + ", ".join(f"`{s}` ({b})" for s, b in skipped) + ".\n")
    else:
        lines.append("No checkpoint was skipped: every backbone matched exposed "
                     "`conv1`/`conv2`.\n")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
