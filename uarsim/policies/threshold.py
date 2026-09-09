"""Fixed-threshold uncertainty policy.

Collapse the five signals into a single weighted uncertainty U and split the
range with two thresholds:  U < t_act -> ACT,  U >= t_defer -> DEFER,  else ASK.
The weights and thresholds come from ThresholdConfig; the frontier experiment
sweeps them.
"""
from __future__ import annotations

import numpy as np

from ..config import ThresholdConfig
from ..data import FEATURE_NAMES
from ..types import Action
from .base import Policy

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


class ThresholdPolicy(Policy):
    name = "threshold"

    def __init__(self, cfg: ThresholdConfig):
        self.cfg = cfg
        self.w = np.array([
            cfg.w_language, cfg.w_perception, cfg.w_planning,
            cfg.w_ood, cfg.w_safety,
        ], dtype=np.float32)
        self.w_sum = float(self.w.sum())
        self._sig_idx = [_IDX[s] for s in
                         ["u_language", "u_perception", "u_planning",
                          "u_ood", "u_safety"]]

    def combined_uncertainty(self, X: np.ndarray) -> np.ndarray:
        sig = X[:, self._sig_idx]
        return (sig * self.w).sum(axis=1) / self.w_sum

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        U = self.combined_uncertainty(X)
        out = np.full(len(X), int(Action.ASK), dtype=np.int64)
        out[U < self.cfg.t_act] = int(Action.ACT)
        out[U >= self.cfg.t_defer] = int(Action.DEFER)
        return out
