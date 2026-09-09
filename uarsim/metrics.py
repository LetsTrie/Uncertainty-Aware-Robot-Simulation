"""Evaluate a policy on a dataset of episodes and compute metrics.

We roll out each policy's decisions, simulate realized outcomes from the stored
latents, and aggregate:

  success_rate : task completed correctly (autonomously, via clarified ACT, or
                 via human on DEFER)
  unsafe_rate  : fraction of episodes with an unsafe collision (the key risk)
  ask_rate     : fraction of clarification questions
  defer_rate   : fraction handed fully to a human
  human_effort : ask_rate + defer_rate (any human involvement)
  mean_cost    : mean realized cost under the cost model (lower is better)
  agreement    : fraction matching the oracle optimal action
"""
from __future__ import annotations

import numpy as np

from .costs import realized_cost, simulate_outcome
from .data import features_matrix
from .types import Action, Outcome


def evaluate_policy(policy, df, cm, seed: int = 12345) -> dict:
    rng = np.random.default_rng(seed)
    X = features_matrix(df)
    actions = policy.decide_batch(X)

    p_correct = df["p_correct_if_act"].to_numpy()
    p_unsafe = df["p_unsafe"].to_numpy()
    optimal = df["optimal_action"].to_numpy()

    n = len(df)
    n_success = n_unsafe = n_wrong = 0
    total_cost = 0.0
    for i in range(n):
        a = Action(int(actions[i]))
        outcome, _ = simulate_outcome(a, float(p_correct[i]), float(p_unsafe[i]), rng)
        total_cost += realized_cost(a, outcome, cm)
        if outcome == Outcome.SUCCESS:
            n_success += 1
        elif outcome == Outcome.UNSAFE:
            n_unsafe += 1
        elif outcome == Outcome.WRONG_OBJECT:
            n_wrong += 1

    ask_rate = float(np.mean(actions == int(Action.ASK)))
    defer_rate = float(np.mean(actions == int(Action.DEFER)))
    return {
        "n": n,
        "success_rate": n_success / n,
        "unsafe_rate": n_unsafe / n,
        "wrong_rate": n_wrong / n,
        "ask_rate": ask_rate,
        "defer_rate": defer_rate,
        "human_effort": ask_rate + defer_rate,
        "mean_cost": total_cost / n,
        "agreement": float(np.mean(actions == optimal)),
    }


def format_table(rows: list[dict], cols=None) -> str:
    """Render a list of metric dicts as a fixed-width text table."""
    cols = cols or ["method", "success_rate", "unsafe_rate", "ask_rate",
                    "defer_rate", "human_effort", "mean_cost", "agreement"]
    widths = {c: max(len(c), max((len(_fmt(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        out.append("  ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
