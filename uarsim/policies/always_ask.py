"""Always-Ask policy: always ask first. Safe against ambiguity, huge human
burden, and still exposed to physical danger because ASK still executes."""
from __future__ import annotations

import numpy as np

from ..types import Action
from .base import Policy


class AlwaysAsk(Policy):
    name = "always_ask"

    def decide_batch(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), int(Action.ASK), dtype=np.int64)
