import sys
sys.path.insert(0, "/home/habb/Kuliah/gnn-marl-network-slicing")

import numpy as np
import pytest
from traffic.poisson_traffic import PoissonTraffic
from traffic.mmpp_traffic import MMPPTraffic


def test_poisson_mean_close_to_lambda():
    lam = 50.0
    rng = np.random.default_rng(42)
    gen = PoissonTraffic(lambda_arrival=lam, packet_size_bits=256, rng=rng)
    arrivals = [gen.step()[0] for _ in range(10000)]
    assert abs(np.mean(arrivals) - lam) < 2.0, f"Mean {np.mean(arrivals)} far from lambda {lam}"


def test_poisson_total_bits_matches_n():
    rng = np.random.default_rng(0)
    gen = PoissonTraffic(lambda_arrival=10.0, packet_size_bits=512, rng=rng)
    for _ in range(100):
        n, bits = gen.step()
        assert bits == n * 512


def test_poisson_reset_changes_rng():
    gen = PoissonTraffic(rng=np.random.default_rng(1))
    s1 = [gen.step()[0] for _ in range(10)]
    gen.reset(np.random.default_rng(1))
    s2 = [gen.step()[0] for _ in range(10)]
    assert s1 == s2


def test_mmpp_state_transitions():
    rng = np.random.default_rng(99)
    gen = MMPPTraffic(r_burst_to_idle=0.9, r_idle_to_burst=0.9, rng=rng)
    states = []
    gen.state = 0
    for _ in range(200):
        gen.step()
        states.append(gen.state)
    assert 0 in states and 1 in states, "MMPP never transitioned between states"


def test_mmpp_burst_rate_higher_than_idle():
    rng_burst = np.random.default_rng(7)
    rng_idle = np.random.default_rng(7)
    gen = MMPPTraffic(lambda_burst=4e6, lambda_idle=0.5e6, r_burst_to_idle=0.0,
                      r_idle_to_burst=0.0, rng=rng_burst)
    gen.state = 1
    burst_bits = np.mean([gen.step() for _ in range(200)])

    gen2 = MMPPTraffic(lambda_burst=4e6, lambda_idle=0.5e6, r_burst_to_idle=0.0,
                       r_idle_to_burst=0.0, rng=rng_idle)
    gen2.state = 0
    idle_bits = np.mean([gen2.step() for _ in range(200)])
    assert burst_bits > idle_bits
