"""
Fase 0 diagnostic D2c -- gradient-norm ratio (docs/revisi/PLAN-01-DIAGNOSTICS.md).

D2a and D2b test the *consequence* of representation collapse: the policy's output barely
depends on neighbour messages. D2c tests the stated *mechanism* -- "gradien dari policy
head didominasi fitur lokal instan; jalur GNN tidak terlatih efektif" -- by measuring how
much gradient actually reaches the GNN.

Why it still matters after the PLAN-04 gate already opened: PLAN-03 section 5
(residual/JK) and PLAN-04 (auxiliary loss) both cleared their gates, and PLAN-04's
Larangan 4 forbids running two anti-collapse techniques at once. They target different
causes -- over-smoothing versus an untrained GNN path. A healthy ratio means the auxiliary
loss attacks a problem that is not there; a ratio far below 1 means it is aimed correctly.

Fidelity is the whole point of this script
------------------------------------------
The PLAN-01 correction block fixes the method: a short instrumented training run reading
p.grad after agent.learn(), NOT a rollout with a re-written loss path. This obeys that
literally -- the loss path is untouched and the real training loop runs.

PPOAgent.learn and DQNAgent.learn both go zero_grad -> backward -> clip_grad_norm_ ->
step. optimizer.step() does not clear .grad, and zero_grad() only runs at the start of the
*next* learn(), so reading p.grad right after the original learn() returns gives exactly
the gradient that was applied on that update.

The agent is constructed *inside* the training function (training/train_proposed.py), so
the instance cannot be wrapped from outside -- the class method is wrapped instead, the
same technique graph_transform uses in scripts/diag_gnn_reliance.py. Zero edits under
training/, agents/, gnn/ or envs/ (PLAN-01 Larangan 1).

Not overwriting the v4 artifacts
--------------------------------
train_gnn_* writes in three places when it finishes: _save_ckpt to
results/checkpoints/{run_name}_last.pt (unconditionally, even with ckpt_interval=0),
_save_model to Path(log_path).with_suffix(".pt"), and MetricsLogger appends to the metrics
CSV. Called with the real run_name/log_path it would corrupt the checkpoint, the final
model and the metrics CSV -- the files every gate number comes from.

So each checkpoint is copied to a scratch run_name first, and log_path points outside
results/logs/. Every write lands on the copy. Verify with the md5 comparison in the plan's
verification step before trusting any number this produces.

Usage:
  python scripts/diag_grad_ratio.py --smoke
  python scripts/diag_grad_ratio.py --ppo-updates 15 --dqn-updates 200
"""
from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.hparams import hparams
from training.train_proposed import train_gnn_dqn, train_gnn_ppo

CKPT_DIR = Path("results/checkpoints")

# One row per update, appended by the wrapper below.
_RECORDS: list[dict] = []


def _group_norms(agent) -> dict:
    """L2 norm and parameter count for the backbone and for everything else.

    Params whose .grad is None count toward the size but contribute 0 to the norm: a
    parameter that received no gradient is evidence, not something to drop from the
    denominator.
    """
    backbone_ids = set()
    if hasattr(agent, "backbone"):
        backbone_ids = {id(p) for p in agent.backbone.parameters()}

    sq = {"backbone": 0.0, "head": 0.0}
    n = {"backbone": 0, "head": 0}
    for p in agent.parameters():
        if not p.requires_grad:
            continue
        key = "backbone" if id(p) in backbone_ids else "head"
        n[key] += p.numel()
        if p.grad is not None:
            sq[key] += float(p.grad.detach().pow(2).sum())
    return {
        "grad_backbone_l2": sq["backbone"] ** 0.5,
        "grad_head_l2": sq["head"] ** 0.5,
        "n_backbone": n["backbone"],
        "n_head": n["head"],
    }


@contextmanager
def instrument(cls, algo: str, seed: int, backbone_name: str):
    """Wrap cls.learn so every update appends its gradient norms to _RECORDS.

    The original method is called first and its return value passed through untouched, so
    the training loop behaves exactly as it would unwrapped.
    """
    original = cls.learn
    counter = {"i": 0}

    def wrapped(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        rec = _group_norms(self)
        rec.update(algo=algo, seed=seed, backbone=backbone_name,
                   update_idx=counter["i"])
        counter["i"] += 1
        _RECORDS.append(rec)
        return out

    cls.learn = wrapped
    try:
        yield
    finally:
        cls.learn = original


def run_one(algo: str, backbone_name: str, seed: int, n_updates: int,
            scratch: Path) -> int:
    """Resume one v4 checkpoint onto a scratch copy and run it forward far enough to
    collect n_updates gradient readings. Returns the number of records collected."""
    run_name = f"{algo}_{backbone_name}_v4_seed{seed}"
    src = CKPT_DIR / f"{run_name}_last.pt"
    if not src.exists():
        print(f"[skip] {src} tidak ada")
        return 0

    state = torch.load(src, map_location="cpu", weights_only=False)
    start = int(state["step"])

    if algo == "gnn-mappo":
        # One PPO update per full rollout buffer.
        extra = n_updates * int(hparams("ppo")["rollout_steps"])
        trainer, cls = train_gnn_ppo, PPOAgent
    else:
        # DQN learns every step once the buffer passes replay_start -- and _maybe_resume
        # does NOT restore the ReplayBuffer, so those steps have to be paid again.
        extra = int(hparams("dqn")["replay_start"]) + n_updates
        trainer, cls = train_gnn_dqn, DQNAgent

    scratch_run = f"{run_name}_d2c"
    scratch_ckpt = CKPT_DIR / f"{scratch_run}_last.pt"
    shutil.copy2(src, scratch_ckpt)
    log_path = scratch / f"{scratch_run}.csv"

    before = len(_RECORDS)
    try:
        with instrument(cls, algo, seed, backbone_name):
            trainer(backbone_name, start + extra, seed, str(log_path),
                    0, True, None, scratch_run)
    finally:
        # Scratch checkpoint and model live in the gitignored checkpoint dir; remove them
        # so a later run cannot resume from a d2c-modified state by accident.
        for p in (scratch_ckpt, CKPT_DIR / f"{scratch_run}_best.pt",
                  log_path.with_suffix(".pt")):
            p.unlink(missing_ok=True)
    return len(_RECORDS) - before


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--ppo-updates", type=int, default=15)
    p.add_argument("--dqn-updates", type=int, default=200)
    p.add_argument("--smoke", action="store_true",
                   help="one PPO checkpoint, 3 updates -- checks p.grad is populated "
                        "before committing to the full run")
    p.add_argument("--out", type=str, default="results/DIAG_GRAD_RATIO.md")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(a, b) for a in ("gnn-mappo", "gnn-madqn") for b in ("gat", "sage")]
    if args.smoke:
        jobs, seeds = [("gnn-mappo", "gat")], [42]
        args.ppo_updates = 3

    scratch = Path(tempfile.mkdtemp(prefix="d2c_"))
    try:
        for algo, backbone_name in jobs:
            n_up = args.ppo_updates if algo == "gnn-mappo" else args.dqn_updates
            for seed in seeds:
                got = run_one(algo, backbone_name, seed, n_up, scratch)
                print(f"[{algo}_{backbone_name} seed{seed}] {got} update terekam")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not _RECORDS:
        raise SystemExit("nol update terekam -- instrumentasi tidak kena")

    df = pd.DataFrame(_RECORDS)
    df["ratio_l2"] = df.grad_backbone_l2 / df.grad_head_l2.replace(0.0, np.nan)
    # Per-parameter RMS: a raw L2 ratio below 1 can mean "fewer parameters" rather than
    # "smaller gradients", and the two groups are not the same size.
    df["rms_backbone"] = df.grad_backbone_l2 / np.sqrt(df.n_backbone)
    df["rms_head"] = df.grad_head_l2 / np.sqrt(df.n_head)
    df["ratio_rms"] = df.rms_backbone / df.rms_head.replace(0.0, np.nan)

    out_path = Path(args.out)
    df.to_csv(out_path.parent / f"{out_path.stem.lower()}.csv", index=False)

    lines = [
        "# D2c -- gradient-norm ratio into the GNN\n",
        f"Seeds: {seeds}. Updates recorded per checkpoint: {args.ppo_updates} (PPO), "
        f"{args.dqn_updates} (DQN). Total updates: {len(df)}.\n",
        "**What this measures, and why it is not a repeat of D2a/D2b.** D2a and D2b test "
        "the consequence -- the policy's output barely depends on neighbour messages. This "
        "tests the mechanism PLAN-01 D2 actually states: whether gradient reaches the GNN "
        "at all. It is what separates PLAN-03 section 5 (residual/JK, aimed at "
        "over-smoothing) from PLAN-04 (auxiliary loss, aimed at an untrained GNN path); "
        "PLAN-04 Larangan 4 forbids running both at once.\n",
        "**How it was measured.** The real training loop runs, resumed from the v4 "
        "checkpoint onto a scratch copy. `PPOAgent.learn` / `DQNAgent.learn` are wrapped at "
        "class level; the original runs first and `p.grad` is read after it returns. "
        "`optimizer.step()` does not clear `.grad` and `zero_grad()` only fires at the "
        "start of the next update, so these are the gradients that were actually applied. "
        "The loss path is untouched -- PLAN-01's correction block requires exactly this "
        "and rules out a rollout with a re-written loss.\n",
        "**Two ratios, because one would be misread.** `ratio_l2` is what PLAN-01 D2c asks "
        "for. `ratio_rms` divides each group's norm by the square root of its parameter "
        "count first. The groups are not the same size -- `gnn-*_sage` has 9,344 backbone "
        "parameters against 18,188 in the head (`results/DIAG_EQUIVARIANCE.md`) -- so a "
        "`ratio_l2` below 1 can mean *fewer parameters* rather than *smaller gradients*. "
        "A verdict of \"the GNN path is untrained\" is only safe when both agree.\n",
        "`clip_grad_norm_` rescales every gradient by one common factor, which cancels in "
        "both ratios, so clipping does not affect these numbers.\n",
        "**Limitation, DQN only.** `_maybe_resume` restores model, optimizer, RNG and CMDP "
        "state but **not** the `ReplayBuffer`, so the DQN runs refill `replay_start` steps "
        "with the converged policy before learning resumes. The DQN ratio therefore "
        "describes gradient flow at convergence on near-on-policy data, not the historical "
        "training mixture. PPO is unaffected: it is on-policy, so a fresh `RolloutBuffer` "
        "is faithful.\n",
        "Median and IQR across updates, not a single number -- one update can be a fluke.\n",
        "| algo | backbone | seed | n updates | ratio_l2 median [IQR] | ratio_rms median [IQR] |",
        "|---|---|---|---|---|---|",
    ]
    for (algo, bb, seed), g in df.groupby(["algo", "backbone", "seed"]):
        def stat(col):
            q1, med, q3 = g[col].quantile([0.25, 0.5, 0.75])
            return f"{med:.4f} [{q1:.4f}, {q3:.4f}]"
        lines.append(f"| `{algo}_{bb}` | {bb} | {seed} | {len(g)} | "
                     f"{stat('ratio_l2')} | {stat('ratio_rms')} |")

    lines.append("\n## Per variant\n")
    lines.append("| variant | ratio_l2 median | ratio_rms median | n updates |")
    lines.append("|---|---|---|---|")
    for (algo, bb), g in df.groupby(["algo", "backbone"]):
        lines.append(f"| `{algo}_{bb}` | {g.ratio_l2.median():.4f} | "
                     f"{g.ratio_rms.median():.4f} | {len(g)} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
