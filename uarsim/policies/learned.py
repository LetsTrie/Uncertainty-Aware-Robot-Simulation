"""Learned assistance policy: an MLP classifier over the oracle
optimal action, trained with cross-entropy. At inference it takes the argmax
of the softmax; the softmax also provides a confidence used for calibration.
"""
from __future__ import annotations

import numpy as np

from ..nn import forward_numpy, load_model
from .base import Policy


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LearnedPolicy(Policy):
    name = "learned"

    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    @classmethod
    def from_path(cls, path, device=None):
        model, meta = load_model(path, device)
        assert meta.get("kind") == "classifier", "expected a classifier checkpoint"
        return cls(model, device)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return _softmax(forward_numpy(self.model, X, self.device))

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        return self.probabilities(X).argmax(axis=1).astype(np.int64)
