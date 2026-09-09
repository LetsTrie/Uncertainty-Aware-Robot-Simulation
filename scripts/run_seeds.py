"""Repeat the main comparison and the ablation over several random seeds.

Reports mean +/- std for every metric, so the headline results carry error
bars instead of resting on a single run. Runs entirely on CPU/MPS.

Usage:
  python scripts/run_seeds.py                    # 5 seeds (main), 3 (ablation)
  python scripts/run_seeds.py --seeds 5 --ablation-seeds 3 --n-train 20000
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uarsim.config import load_config
from uarsim.data import FEATURE_NAMES, cost_matrix, features_matrix
from uarsim.env.scene_generator import episode_stream
from uarsim.data import episode_to_row
from uarsim.metrics import evaluate_policy
from uarsim.policies import (AlwaysAct, AlwaysAsk, CostSensitivePolicy,
                             LearnedPolicy, ThresholdPolicy)
from uarsim.policies.learned import _softmax
from uarsim.nn import forward_numpy, get_device
from uarsim.train import train_classifier
from uarsim.policies.base import Policy

METRICS = ["success_rate", "unsafe_rate", "human_effort", "mean_cost", "agreement"]
SIGNALS = ["u_language", "u_perception", "u_planning", "u_ood", "u_safety"]


def gen_df(cfg, mode, n, seed):
    return pd.DataFrame(episode_to_row(ep)
                        for ep in episode_stream(cfg, mode, n, seed))


class _MaskedLearned(Policy):
    def __init__(self, model, device, mask):
        self.model, self.device, self.mask = model, device, mask

    def decide_batch(self, X):
        X = X.copy()
        if self.mask is not None:
            X[:, self.mask] = 0.0
        return _softmax(forward_numpy(self.model, X, self.device)).argmax(1)


def agg(records, keys):
    """records: list of dict rows -> DataFrame of mean/std grouped by keys."""
    df = pd.DataFrame(records)
    out = df.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    return out


def fmt_pm(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.train.epochs = args.epochs
    device = get_device()
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    print(f"Device: {device} | seeds: {args.seeds} (main), "
          f"{args.ablation_seeds} (ablation)")

    main_rows, abl_rows = [], []
    for s in range(args.seeds):
        base = 1000 * (s + 1)
        train = gen_df(cfg, "train", args.n_train, base + 1)
        tid = gen_df(cfg, "train", args.n_test, base + 2)
        tshift = gen_df(cfg, "shift", args.n_test, base + 3)

        X = features_matrix(train)
        yc = train["optimal_action"].to_numpy()
        clf = train_classifier(X, yc, cfg.train, device)
        from uarsim.train import train_cost
        reg = train_cost(X, cost_matrix(train), cfg.train, device)

        policies = [("Always Act", AlwaysAct()), ("Always Ask", AlwaysAsk()),
                    ("Fixed Threshold", ThresholdPolicy(cfg.threshold)),
                    ("Learned Policy", LearnedPolicy(clf, device)),
                    ("Cost-aware Learned", CostSensitivePolicy(reg, device))]
        for split, df in [("in-distribution", tid), ("distribution-shift", tshift)]:
            for name, pol in policies:
                m = evaluate_policy(pol, df, cfg.cost)
                m.update({"seed": s, "split": split, "method": name})
                main_rows.append(m)

        # ablation on the shifted set for the first few seeds
        if s < args.ablation_seeds:
            variants = [("Full", None)] + [(f"- {sig}", idx[sig]) for sig in SIGNALS]
            for label, mask in variants:
                Xtr = X.copy()
                if mask is not None:
                    Xtr[:, mask] = 0.0
                mdl = train_classifier(Xtr, yc, cfg.train, device)
                m = evaluate_policy(_MaskedLearned(mdl, device, mask), tshift, cfg.cost)
                m.update({"seed": s, "method": label})
                abl_rows.append(m)
        print(f"  seed {s+1}/{args.seeds} done")

    main_agg = agg(main_rows, ["split", "method"])
    abl_agg = agg(abl_rows, ["method"])
    main_agg.to_csv(_bootstrap.TABLE_DIR / "main_results_seeds.csv", index=False)
    abl_agg.to_csv(_bootstrap.TABLE_DIR / "ablation_seeds.csv", index=False)

    order = ["Always Act", "Always Ask", "Fixed Threshold",
             "Learned Policy", "Cost-aware Learned"]
    for split in ["in-distribution", "distribution-shift"]:
        print(f"\n=== {split}  (mean ± std over {args.seeds} seeds) ===")
        print(f"{'method':<20}{'success':>16}{'unsafe':>16}{'effort':>16}{'cost':>16}")
        sub = main_agg[main_agg["split"] == split].set_index("method")
        for name in order:
            r = sub.loc[name]
            print(f"{name:<20}"
                  f"{fmt_pm(r[('success_rate','mean')], r[('success_rate','std')]):>16}"
                  f"{fmt_pm(r[('unsafe_rate','mean')], r[('unsafe_rate','std')]):>16}"
                  f"{fmt_pm(r[('human_effort','mean')], r[('human_effort','std')]):>16}"
                  f"{fmt_pm(r[('mean_cost','mean')], r[('mean_cost','std')]):>16}")

    print(f"\n=== ablation on shift (mean ± std over {args.ablation_seeds} seeds) ===")
    print(f"{'model':<16}{'success':>16}{'unsafe':>16}{'cost':>16}")
    ai = abl_agg.set_index("method")
    for label in ["Full"] + [f"- {s}" for s in SIGNALS]:
        r = ai.loc[label]
        print(f"{label:<16}"
              f"{fmt_pm(r[('success_rate','mean')], r[('success_rate','std')]):>16}"
              f"{fmt_pm(r[('unsafe_rate','mean')], r[('unsafe_rate','std')]):>16}"
              f"{fmt_pm(r[('mean_cost','mean')], r[('mean_cost','std')]):>16}")
    print(f"\nWrote main_results_seeds.csv and ablation_seeds.csv")


if __name__ == "__main__":
    main()
