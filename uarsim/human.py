"""Simulated human.

No real person is needed: the simulator knows the true target, so a
clarification question is answered by revealing it. Human help is not free —
asking and deferring carry costs defined in the cost model — which is what
creates the autonomy/effort trade-off the policies must navigate.
"""
from __future__ import annotations


def answer_clarification(episode) -> str:
    """Return the true target id in response to an ASK."""
    return episode.true_target_id


def take_over(episode) -> str:
    """Human performs the task on DEFER; returns the true target id."""
    return episode.true_target_id
