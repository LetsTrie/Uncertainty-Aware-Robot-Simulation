"""Unit tests for the decision layer: cost model, oracle, generation, policies."""
import numpy as np
import pandas as pd

from uarsim.config import Config, CostConfig
from uarsim.costs import expected_costs, optimal_action, simulate_outcome
from uarsim.data import FEATURE_NAMES, episode_to_row, features_matrix
from uarsim.env.scene_generator import sample_episode
from uarsim.metrics import evaluate_policy
from uarsim.policies import (AlwaysAct, AlwaysAsk, CostSensitivePolicy,
                             LearnedPolicy, ThresholdPolicy)
from uarsim.types import Action, Outcome

CM = CostConfig()


def test_optimal_action_regimes():
    # safe + confident -> ACT
    assert optimal_action(p_correct=0.99, p_unsafe=0.02, cm=CM) == Action.ACT
    # safe + ambiguous -> ASK
    assert optimal_action(p_correct=0.45, p_unsafe=0.02, cm=CM) == Action.ASK
    # dangerous -> DEFER
    assert optimal_action(p_correct=0.9, p_unsafe=0.6, cm=CM) == Action.DEFER


def test_expected_costs_defer_constant():
    c = expected_costs(0.5, 0.5, CM)
    assert c[Action.DEFER] == CM.c_defer
    # ASK never cheaper than its own floor
    assert c[Action.ASK] >= CM.c_ask


def test_simulate_outcome_bounds():
    rng = np.random.default_rng(0)
    # a perfectly safe, certain ACT always succeeds
    out, _ = simulate_outcome(Action.ACT, p_correct=1.0, p_unsafe=0.0, rng=rng)
    assert out == Outcome.SUCCESS
    # DEFER always succeeds (human)
    out, _ = simulate_outcome(Action.DEFER, 0.0, 0.9, rng)
    assert out == Outcome.SUCCESS


def test_episode_row_has_all_features():
    ep = sample_episode(Config(), "train", np.random.default_rng(1))
    row = episode_to_row(ep)
    for name in FEATURE_NAMES:
        assert name in row and 0.0 <= row[name] <= 1.0 + 1e-6
    assert row["optimal_action"] in (0, 1, 2)
    assert 0.0 <= row["p_unsafe"] <= 1.0


def test_signals_in_unit_range():
    rng = np.random.default_rng(2)
    for _ in range(200):
        ep = sample_episode(Config(), "shift", rng)
        for v in ep.signals.values():
            assert 0.0 <= v <= 1.0


def _tiny_df(n=400, mode="train"):
    rng = np.random.default_rng(3)
    return pd.DataFrame(episode_to_row(sample_episode(Config(), mode, rng))
                        for _ in range(n))


def test_baseline_policies_shapes():
    df = _tiny_df()
    X = features_matrix(df)
    for pol in (AlwaysAct(), AlwaysAsk(), ThresholdPolicy(Config().threshold)):
        a = pol.decide_batch(X)
        assert a.shape == (len(df),)
        assert set(np.unique(a)).issubset({0, 1, 2})


def test_always_ask_metrics_sane():
    df = _tiny_df()
    m = evaluate_policy(AlwaysAsk(), df, CM)
    assert m["ask_rate"] == 1.0
    assert 0.0 <= m["success_rate"] <= 1.0
    assert 0.0 <= m["unsafe_rate"] <= 1.0


def test_learned_policies_train_and_decide():
    from uarsim.train import train_classifier, train_cost
    from uarsim.data import cost_matrix
    df = _tiny_df(n=1500)
    X = features_matrix(df)
    clf = train_classifier(X, df["optimal_action"].to_numpy(), Config().train)
    reg = train_cost(X, cost_matrix(df), Config().train)
    for pol in (LearnedPolicy(clf), CostSensitivePolicy(reg)):
        a = pol.decide_batch(X)
        assert set(np.unique(a)).issubset({0, 1, 2})
