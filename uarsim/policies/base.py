"""Policy interface.

A policy maps a batch of feature vectors to a batch of actions. Working in
batches keeps evaluation fast and lets the learned policies run a single
forward pass over the whole test set.
"""
from __future__ import annotations

import numpy as np


class Policy:
    name = "policy"

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 10) float array -> (N,) int array of Action values."""
        raise NotImplementedError

    def decide(self, x: np.ndarray) -> int:
        """Convenience for a single feature vector."""
        return int(self.decide_batch(np.asarray(x, dtype=np.float32)[None, :])[0])
