"""Assemble the standard set of policies for the experiments.

Kept in the package (not in a script) so the experiment scripts don't have to
import one another.
"""
from __future__ import annotations

from pathlib import Path

from .policies import (AlwaysAct, AlwaysAsk, CostSensitivePolicy,
                       LearnedPolicy, ThresholdPolicy)


def build_policies(cfg, model_dir: str | Path):
    """Return [(label, policy), ...]; learned policies included if trained."""
    model_dir = Path(model_dir)
    policies = [
        ("Always Act", AlwaysAct()),
        ("Always Ask", AlwaysAsk()),
        ("Fixed Threshold", ThresholdPolicy(cfg.threshold)),
    ]
    learned = model_dir / "learned.pt"
    cost = model_dir / "cost_sensitive.pt"
    if learned.exists():
        policies.append(("Learned Policy", LearnedPolicy.from_path(learned)))
    else:
        print("(!) results/models/learned.pt not found — run train_policy.py "
              "to include the learned policies.")
    if cost.exists():
        policies.append(("Cost-aware Learned", CostSensitivePolicy.from_path(cost)))
    return policies
