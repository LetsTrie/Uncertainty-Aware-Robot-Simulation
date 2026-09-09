"""Cost model and the oracle optimal action.

The whole project reduces to risk-aware action selection:

    a* = argmin_a  E[ C(a, world) ]

We can compute these expectations exactly because, in simulation, we know
the latent probabilities that govern each outcome:

    p_correct : P(robot acts on the correct target | it just ACTs)
    p_unsafe  : P(an unsafe collision occurs | the robot physically executes)

Modeling choices (documented so the assumptions are explicit):

  * ACT   executes immediately. It can be wrong (identity error) or unsafe.
  * ASK   resolves *referential* ambiguity — a human names the target, so the
          identity error effectively vanishes (p_correct -> 1). It does NOT
          remove physical danger: the robot still executes, so the unsafe risk
          remains. This is why, for genuinely dangerous scenes, DEFER strictly
          dominates ASK.
  * DEFER hands the task to a human: no identity error, no unsafe collision,
          but it costs the most human effort.
"""
from __future__ import annotations

from .config import CostConfig
from .types import Action


def expected_costs(p_correct: float, p_unsafe: float, cm: CostConfig) -> dict:
    """Expected cost of each action given the latent world probabilities."""
    cost_act = (
        p_unsafe * cm.c_unsafe
        + (1.0 - p_unsafe) * (1.0 - p_correct) * cm.c_wrong
    )
    cost_ask = cm.c_ask + p_unsafe * cm.c_unsafe
    cost_defer = cm.c_defer
    return {Action.ACT: cost_act, Action.ASK: cost_ask, Action.DEFER: cost_defer}


def optimal_action(p_correct: float, p_unsafe: float, cm: CostConfig) -> Action:
    """The cost-minimizing action — the oracle label for supervised learning."""
    costs = expected_costs(p_correct, p_unsafe, cm)
    return min(costs, key=costs.get)


def simulate_outcome(action: Action, p_correct: float, p_unsafe: float, rng):
    """Sample a realized (outcome, cost) for a chosen action.

    Used at evaluation time to turn a policy's decisions into success / unsafe
    / query statistics. ``rng`` is a numpy Generator.
    """
    from .types import Outcome

    if action == Action.DEFER:
        return Outcome.SUCCESS, 0.0  # human completes it safely (effort tracked separately)

    # ACT and ASK both physically execute -> exposed to the unsafe risk.
    if rng.random() < p_unsafe:
        return Outcome.UNSAFE, 0.0

    if action == Action.ASK:
        # Clarification fixes identity; if it didn't collide, it succeeds.
        return Outcome.SUCCESS, 0.0

    # Plain ACT: identity may still be wrong.
    if rng.random() < p_correct:
        return Outcome.SUCCESS, 0.0
    return Outcome.WRONG_OBJECT, 0.0


def realized_cost(action: Action, outcome, cm: CostConfig) -> float:
    """Cost bookkeeping for a realized (action, outcome) pair."""
    from .types import Outcome

    cost = 0.0
    if action == Action.ASK:
        cost += cm.c_ask
    elif action == Action.DEFER:
        cost += cm.c_defer
    if outcome == Outcome.WRONG_OBJECT:
        cost += cm.c_wrong
    elif outcome == Outcome.UNSAFE:
        cost += cm.c_unsafe
    return cost
