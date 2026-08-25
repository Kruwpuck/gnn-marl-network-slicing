"""Per-gNB resilient minimum-rate constraint (docs/revisi/PLAN-02).

The constraint lives in the environment, not in any agent class, so symmetry across the
eight algorithms is structural rather than aspirational: identical actions produce an
identical penalised reward because the same code computes it.

The first test is the one that matters most. Wave v5's `resilient=none` arm is meant to be
the v4 baseline, and it is only a valid comparator if switching the block off changes
nothing at all.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import yaml

from envs.network_slicing_env import NetworkSlicingEnv

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "experiment_config.yaml"


def _config_without_resilient(tmp_path: Path) -> str:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.pop("resilient", None)
    out = tmp_path / "no_resilient.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def _arm(env, f_min_mbps=8.0, mode="learned", every=20, alpha_mu=0.5, alpha_z=0.05):
    """Turn the constraint on post-construction, the same way test_env.py adjusts the CMDP
    knobs. f_min is set well above what the policy under test can reach, so the shortfall
    is genuine rather than incidental."""
    env.resilient_mode = mode
    env.resilient_f_min = f_min_mbps * 1e6
    env.dual_update_every = every
    env.alpha_mu = alpha_mu
    env.alpha_z = alpha_z
    env.resilient_c = 0.0     # so z is driven purely by mu in these tests
    return env


def _rollout(env, seed, steps, action):
    obs, _ = env.reset(seed=seed)
    trace = [np.asarray(obs, dtype=np.float64).copy()]
    info: dict = {}
    for _ in range(steps):
        obs, reward, _, truncated, info = env.step([action] * env.n_gnb)
        trace.append(np.asarray(obs, dtype=np.float64).copy())
        trace.append(np.array([reward, float(info["lam"])]))
        trace.append(np.asarray(info["embb_thr_bps"], dtype=np.float64).copy())
        if truncated:
            obs, _ = env.reset(seed=seed)
    return trace, set(info.keys())


def test_mode_none_is_identical_to_no_resilient_block(tmp_path):
    """`resilient=none` must be bit-identical to the code path that existed before the
    block was added -- otherwise the v5 `none` arm is not the v4 baseline it claims to be."""
    a = NetworkSlicingEnv()
    b = NetworkSlicingEnv(config_path=_config_without_resilient(tmp_path))
    assert a.resilient_mode == "none", "the committed config must ship the arm switched off"

    trace_a, keys_a = _rollout(a, seed=7, steps=200, action=5)
    trace_b, keys_b = _rollout(b, seed=7, steps=200, action=5)

    assert keys_a == keys_b, f"info keys differ: {keys_a ^ keys_b}"
    assert len(trace_a) == len(trace_b)
    for i, (x, y) in enumerate(zip(trace_a, trace_b)):
        assert np.array_equal(x, y), f"divergence at trace element {i}"
    a.close()
    b.close()


def test_identical_actions_give_identical_penalised_reward():
    """Gate C1 in miniature, on the new term: the constraint is computed by the
    environment, so no algorithm can receive a different one for the same action."""
    a = _arm(NetworkSlicingEnv())
    b = _arm(NetworkSlicingEnv())
    a.reset(seed=3)
    b.reset(seed=3)
    for _ in range(120):
        _, ra, _, _, ia = a.step([4] * a.n_gnb)
        _, rb, _, _, ib = b.step([4] * b.n_gnb)
        assert ra == rb
        assert np.array_equal(ia["mu"], ib["mu"])
        assert np.array_equal(ia["z"], ib["z"])
        assert np.array_equal(ia["resilient_shortfall"], ib["resilient_shortfall"])
    a.close()
    b.close()


def test_mu_and_z_never_reach_the_policy_state_dict():
    """PLAN-02 section 5 / Larangan 3. The dual state lives in a plain dict on the env, so
    the model dimension cannot come to depend on the number of gNB."""
    env = _arm(NetworkSlicingEnv())
    state = env.get_cmdp_state()
    assert len(state["mu"]) == env.n_gnb
    assert len(state["z"]) == env.n_gnb
    assert isinstance(state["mu"], list) and isinstance(state["mu"][0], float)

    env.reset(seed=1)
    for _ in range(40):
        env.step([0] * env.n_gnb)
    saved = env.get_cmdp_state()
    fresh = _arm(NetworkSlicingEnv())
    fresh.set_cmdp_state(saved)
    assert np.array_equal(fresh._mu, env._mu)
    assert np.array_equal(fresh._z, env._z)
    env.close()
    fresh.close()


def test_set_cmdp_state_accepts_a_pre_resilient_checkpoint():
    """A v4 checkpoint was written before mu and z existed. Resuming from it must keep the
    configured initial values rather than raise."""
    env = _arm(NetworkSlicingEnv())
    env.set_cmdp_state({"lam": 2.0, "total_steps": 100, "viol_window": [0.1, 0.2]})
    assert env._lam == 2.0
    assert env._mu.shape == (env.n_gnb,)
    assert np.all(env._mu == env.mu_init)
    env.close()


def test_mu_rises_under_sustained_shortfall():
    env = _arm(NetworkSlicingEnv())
    env.reset(seed=5)
    mu_before = env._mu.copy()
    for _ in range(100):
        env.step([10] * env.n_gnb)  # all PRB to URLLC -> eMBB starved -> real shortfall
    assert np.all(env._mu > mu_before), "mu must rise while the rate sits below f_min"
    env.close()


def test_z_moves_slower_than_mu():
    """Timescale separation, PLAN-02 section 4. If the slack keeps up with the multiplier
    it loosens to chase the policy and the constraint goes vacuous."""
    env = _arm(NetworkSlicingEnv(), alpha_mu=0.5, alpha_z=0.005)
    env.reset(seed=6)
    for _ in range(100):
        env.step([10] * env.n_gnb)
    assert np.all(env._mu > 0.0)
    assert np.all(env._z < env._mu), "z must lag mu, not track it"
    env.close()


def test_fixed_mode_keeps_slack_at_zero():
    """The honesty arm of PLAN-02 section 8: if `fixed` and `learned` agree, the learnable
    slack claim is not supported and the mechanism is a plain minimum-rate constraint."""
    env = _arm(NetworkSlicingEnv(), mode="fixed")
    env._z[:] = 3.0  # even a non-zero z_init must be ignored in this mode
    env.reset(seed=8)
    for _ in range(60):
        _, _, _, _, info = env.step([10] * env.n_gnb)
    assert np.array_equal(info["z"], np.full(env.n_gnb, 3.0)), "fixed must not update z"
    expected = np.maximum(0.0, (env.resilient_f_min - info["embb_thr_bps"])
                          / max(env.embb_min_throughput, 1.0))
    assert np.allclose(info["resilient_shortfall"], expected), "fixed must ignore z entirely"
    env.close()


def test_mu_and_z_persist_across_reset():
    env = _arm(NetworkSlicingEnv())
    env.reset(seed=1)
    for _ in range(40):
        env.step([10] * env.n_gnb)
    mu_after, z_after = env._mu.copy(), env._z.copy()
    env.reset(seed=2)
    assert np.array_equal(env._mu, mu_after), "mu must not reset with the episode"
    assert np.array_equal(env._z, z_after), "z must not reset with the episode"
    env.close()


def test_missing_f_min_fails_loudly(tmp_path):
    """An unset f_min would constrain against 0, which every policy satisfies -- the arm
    would look like it ran when it did not."""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["resilient"]["mode"] = "learned"
    cfg["resilient"]["f_min_mbps"] = None
    path = tmp_path / "no_fmin.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="f_min_mbps"):
        NetworkSlicingEnv(config_path=str(path))


def test_penalty_is_normalised_so_the_clip_does_not_swallow_it():
    """PLAN-02 section 11 plus the calibration-round-3 failure mode: an un-normalised
    shortfall is ~1e6, so any useful mu would saturate the [-10, 10] clip on every step."""
    env = _arm(NetworkSlicingEnv())
    env.reset(seed=9)
    for _ in range(60):
        _, _, _, _, info = env.step([10] * env.n_gnb)
    assert np.all(info["resilient_shortfall"] < 100.0), "shortfall is not on the reward scale"
    assert info["reward_clip_frac"] < 0.05, (
        f"clipped on {info['reward_clip_frac']:.1%} of steps -- above the 5% ceiling "
        "PLAN-02 section 11 sets for splitting the objective and penalty clips")
    env.close()
