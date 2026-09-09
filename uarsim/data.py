"""Flatten episodes to tabular rows and back to model inputs.

The saved dataset is a CSV where each row carries:
  * the 10 observed features the policy is allowed to see,
  * the latents needed to *simulate outcomes* at evaluation time,
  * the oracle labels (expected per-action costs + optimal action),
  * a little metadata (mode, counts).
"""
from __future__ import annotations

import numpy as np

# The 10 features the assistance policy observes (uncertainty signals + scene stats).
FEATURE_NAMES = [
    "u_language", "u_perception", "u_planning", "u_ood", "u_safety",
    "n_objects", "n_matching", "target_distance", "obstacle_count",
    "fragile_in_path",
]

# Normalisers for the raw count-like features -> roughly [0, 1].
_N_OBJ = 8.0
_N_MATCH = 8.0
_N_OBST = 5.0


def episode_to_row(ep) -> dict:
    """Flatten an Episode into a plain dict suitable for a DataFrame row."""
    s = ep.signals
    m = ep.meta
    lat = ep.latents
    row = {
        # observed features
        "u_language": s["u_language"],
        "u_perception": s["u_perception"],
        "u_planning": s["u_planning"],
        "u_ood": s["u_ood"],
        "u_safety": s["u_safety"],
        "n_objects": m["n_objects"] / _N_OBJ,
        "n_matching": m["n_matching"] / _N_MATCH,
        "target_distance": lat["target_distance"],
        "obstacle_count": m["obstacle_count"] / _N_OBST,
        "fragile_in_path": float(m["fragile_in_path"]),
        # latents (for outcome simulation; NOT policy inputs)
        "p_correct_if_act": lat["p_correct_if_act"],
        "p_unsafe": lat["p_unsafe"],
        # oracle labels
        "cost_act": lat["cost_act"],
        "cost_ask": lat["cost_ask"],
        "cost_defer": lat["cost_defer"],
        "optimal_action": lat["optimal_action"],
        # metadata
        "mode": m["mode"],
        "target_novel": m["target_novel"],
    }
    return row


def features_matrix(df) -> np.ndarray:
    """Extract the (N, 10) float32 feature matrix from a DataFrame."""
    return df[FEATURE_NAMES].to_numpy(dtype=np.float32)


def cost_matrix(df) -> np.ndarray:
    """Extract the (N, 3) true expected-cost targets [act, ask, defer]."""
    return df[["cost_act", "cost_ask", "cost_defer"]].to_numpy(dtype=np.float32)
