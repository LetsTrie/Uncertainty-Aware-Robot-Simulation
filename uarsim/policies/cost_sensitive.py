"""Cost-sensitive learned policy.

Instead of classifying the optimal action, this MLP *regresses* the three
expected costs [C(ACT), C(ASK), C(DEFER)] and then chooses

    a* = argmin_a  Ehat[C(a, world)]

This is the risk-aware version: it reasons about the magnitude of each mistake
(an unsafe collision costs far more than a clarification) rather than treating
every decision boundary equally, which is what turns the project from plain
classification into selective autonomy.
"""
from __future__ import annotations

import numpy as np

from ..nn import forward_numpy, load_model
from .base import Policy


class CostSensitivePolicy(Policy):
    name = "cost_sensitive"

    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    @classmethod
    def from_path(cls, path, device=None):
        model, meta = load_model(path, device)
        assert meta.get("kind") == "cost", "expected a cost-regression checkpoint"
        return cls(model, device)

    def predicted_costs(self, X: np.ndarray) -> np.ndarray:
        return forward_numpy(self.model, X, self.device)

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        return self.predicted_costs(X).argmin(axis=1).astype(np.int64)
