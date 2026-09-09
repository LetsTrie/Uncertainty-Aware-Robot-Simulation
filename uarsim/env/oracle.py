"""The oracle: privileged ground truth used to build labels.

Because the simulator knows the true world, it can compute the latent
probabilities that govern each action's outcome and therefore the exact
cost-minimizing decision. The learned policy never sees any of this — only the
noisy signals derived in ``uncertainty.py``.
"""
from __future__ import annotations

from ..config import CostConfig
from ..costs import expected_costs, optimal_action


def compute_latents(scene, target, n_matches) -> dict:
    """Latent probabilities that drive outcomes for this episode."""
    from .objects import normalized_distance

    # P(correct if ACT): must pick the right match AND perceive it right.
    # Ambiguity (n_matches > 1) and low perception confidence lower this, which
    # is what pushes an otherwise-safe scene toward ASK.
    p_pick = 1.0 / max(n_matches, 1)
    p_correct = p_pick * target.perception_conf

    # P(unsafe if the robot physically executes). Fragile-in-path and an
    # unreachable target are the big drivers; a dense scene and an
    # out-of-distribution target add smaller, decision-relevant risk, so
    # u_ood and u_planning (via clutter) carry safety information too, not just
    # u_safety.
    p_unsafe = 0.02
    if scene.fragile_in_path:
        p_unsafe += 0.45
    if not target.reachable:
        p_unsafe += 0.30
    p_unsafe += 0.12 * float(target.novel)            # OOD -> small DEFER pressure
    p_unsafe += 0.30 * max(0.0, scene.clutter - 0.5)  # dense -> planning risk
    p_unsafe = min(0.95, p_unsafe)

    return {
        "p_correct_if_act": float(p_correct),
        "p_unsafe": float(p_unsafe),
        "target_distance": normalized_distance(target.x, target.y),
    }


def label_episode(latents: dict, cm: CostConfig) -> dict:
    """Attach expected per-action costs and the oracle optimal action."""
    costs = expected_costs(latents["p_correct_if_act"], latents["p_unsafe"], cm)
    from ..types import Action

    return {
        "cost_act": costs[Action.ACT],
        "cost_ask": costs[Action.ASK],
        "cost_defer": costs[Action.DEFER],
        "optimal_action": int(optimal_action(
            latents["p_correct_if_act"], latents["p_unsafe"], cm)),
    }
