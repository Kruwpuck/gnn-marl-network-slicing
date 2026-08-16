"""Agent hyperparameters, read from configs/experiment_config.yaml.

goal1.md C2 requires every hyperparameter to come from that one file. Until 2026-08-16
lr/gamma/clip_eps and friends lived as Python defaults in agents/*.py, so a reader of the
config could not see the numbers the wave actually ran with. The values there are
bit-identical to the defaults they replaced (tests/test_hparams_identity.py), so this is a
reproducibility fix, not a retune.

Constructors resolve through here rather than taking the values as arguments, so anything
that rebuilds an agent -- evaluation, diagnostics, ablations -- gets the same numbers
without having to know they exist.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml"


@lru_cache(maxsize=8)
def hparams(family: str, config_path: str | None = None) -> dict:
    """Hyperparameters for 'dqn' or 'ppo'. Keys outside the two family blocks (n_actions,
    hidden) are shared and merge into the returned dict.

    config_path matters: a wave job runs against configs/generated/floor_*.yaml, a full
    copy of the base config, so passing the job's own config keeps the agent and the env
    reading the same file.
    """
    cfg = yaml.safe_load(Path(config_path or _DEFAULT_CONFIG).read_text())["agent"]
    shared = {k: v for k, v in cfg.items() if k not in ("dqn", "ppo")}
    return {**shared, **cfg[family]}


def resolve(family: str, config_path: str | None, overrides: dict) -> dict:
    """hparams() plus caller overrides, rejecting unknown keys.

    Without the check a typo (`clip_epsilon=0.1`) would be silently dropped and the run
    would report a hyperparameter it never used -- the exact failure this module exists to
    prevent.
    """
    hp = hparams(family, config_path)
    unknown = set(overrides) - set(hp)
    if unknown:
        raise ValueError(f"unknown {family} hyperparameter(s): {sorted(unknown)}; "
                         f"known: {sorted(hp)}")
    return {**hp, **overrides}
