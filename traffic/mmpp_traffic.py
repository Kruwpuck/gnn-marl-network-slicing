import numpy as np


class MMPPTraffic:
    """
    eMBB bursty traffic: 2-state Markov-Modulated Poisson Process.
    States: 0=idle, 1=burst.
    """

    def __init__(self, lambda_burst: float = 4.0e6, lambda_idle: float = 0.5e6,
                 r_burst_to_idle: float = 0.2, r_idle_to_burst: float = 0.8,
                 rng: np.random.Generator | None = None):
        self.lambda_burst = lambda_burst
        self.lambda_idle = lambda_idle
        self.r_b2i = r_burst_to_idle
        self.r_i2b = r_idle_to_burst
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = 0

    def step(self, dt: float = 1.0) -> float:
        """Advance one slot. Return bits_arrived."""
        if self.state == 0:
            if self.rng.random() < 1.0 - np.exp(-self.r_i2b * dt):
                self.state = 1
            lam = self.lambda_idle
        else:
            if self.rng.random() < 1.0 - np.exp(-self.r_b2i * dt):
                self.state = 0
            lam = self.lambda_burst
        return float(self.rng.poisson(lam * dt))

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.state = 0
        if rng is not None:
            self.rng = rng
