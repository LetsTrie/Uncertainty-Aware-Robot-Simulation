"""Always-Act policy: always execute. Fast, zero queries, but unsafe/error-prone."""
from __future__ import annotations

import numpy as np

from ..types import Action
from .base import Policy


class AlwaysAct(Policy):
    name = "always_act"

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), int(Action.ACT), dtype=np.int64)
