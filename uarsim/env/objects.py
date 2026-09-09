"""Object vocabulary and geometry helpers.

Every object is represented as a labelled point on the table (think of a
cube standing in for a mug). Positions live in table coordinates:

    x in [-1, 1], y in [0, 1];  the robot base sits in front at BASE.
"""
from __future__ import annotations

import numpy as np

# In-distribution vocabulary (seen during training).
TRAIN_CATEGORIES = ["mug", "bottle", "book", "cup", "bowl"]
TRAIN_COLORS = ["red", "blue", "green", "yellow"]

# Out-of-distribution vocabulary (only appears under distribution shift).
NOVEL_CATEGORIES = ["teapot", "ladle", "figurine", "gadget", "canister"]
NOVEL_COLORS = ["magenta", "teal", "olive", "maroon"]

LANDMARKS = ["laptop", "keyboard", "monitor", "plant"]

BASE = np.array([0.0, -1.2])  # robot base position, in front of the table
_DIAG = float(np.hypot(2.0, 2.2))  # normaliser for distances from the base


def normalized_distance(x: float, y: float) -> float:
    """Distance from the robot base to (x, y), normalized to roughly [0, 1]."""
    d = float(np.hypot(x - BASE[0], y - BASE[1]))
    return min(d / _DIAG, 1.0)


def point_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Shortest distance from point P to segment AB (used for 'fragile in path')."""
    ax, ay, bx, by, px, py = map(float, (ax, ay, bx, by, px, py))
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - cx, py - cy))
