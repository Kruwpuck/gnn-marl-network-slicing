import numpy as np


class PoissonTraffic:
    """URLLC sporadic traffic: homogeneous Poisson arrivals."""

    def __init__(self, lambda_arrival: float = 50.0, packet_size_bits: int = 256,
                 rng: np.random.Generator | None = None):
        self.lambda_arrival = lambda_arrival
        self.packet_size_bits = packet_size_bits
        self.rng = rng if rng is not None else np.random.default_rng()

    def step(self, dt: float = 1.0) -> tuple[int, int]:
        """Return (n_arrivals, total_bits) for one slot of duration dt."""
        n = int(self.rng.poisson(self.lambda_arrival * dt))
        return n, n * self.packet_size_bits

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
