"""Ablations: which uncertainty signals actually matter?

For each uncertainty signal we retrain the classifier with that signal zeroed
out (in both training and evaluation), then measure performance on the shifted
test set. Comparing to the full model tells us which signals carry the load.

Writes results/tables/ablation.csv.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from uarsim.config import load_config
from uarsim.data import FEATURE_NAMES, features_matrix
from uarsim.metrics import evaluate_policy, format_table
from uarsim.nn import MLP, forward_numpy, get_device
from uarsim.policies.base import Policy
from uarsim.policies.learned import _softmax

SIGNALS = ["u_language", "u_perception", "u_planning", "u_ood", "u_safety"]


class _MaskedLearned(Policy):
    """Wrap a trained model, zeroing the given feature indices at inference."""

    def __init__(self, model, device, mask_idx):
        self.model, self.device, self.mask_idx = model, device, mask_idx

    def decide_batch(self, X):
        X = X.copy()
        if self.mask_idx is not None:
            X[:, self.mask_idx] = 0.0
        return _softmax(forward_numpy(self.model, X, self.device)).argmax(1).astype(np.int64)


def _train_classifier(X, y, tcfg, device):
    model = MLP(in_dim=len(FEATURE_NAMES), hidden=tcfg.hidden, out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                           weight_decay=tcfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X.copy()).to(device)
    yt = torch.from_numpy(y.copy()).to(device)
    n = len(Xt)
    for _ in range(tcfg.epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, tcfg.batch_size):
            j = idx[i:i + tcfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[j]), yt[j])
            loss.backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.train.epochs = args.epochs
    device = get_device()
    print(f"Device: {device}  (ablation retrains one model per row)")

    train_df = pd.read_csv(_bootstrap.DATA_DIR / "train.csv")
    shift_df = pd.read_csv(_bootstrap.DATA_DIR / "test_shift.csv")
    X = features_matrix(train_df)
    y = train_df["optimal_action"].to_numpy(dtype=np.int64)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    variants = [("Full", None)] + [(f"- {s}", idx[s]) for s in SIGNALS]
    rows = []
    for label, mask in variants:
        Xtr = X.copy()
        if mask is not None:
            Xtr[:, mask] = 0.0
        model = _train_classifier(Xtr, y, cfg.train, device)
        pol = _MaskedLearned(model, device, mask)
        m = evaluate_policy(pol, shift_df, cfg.cost)
        m["method"] = label
        rows.append(m)
        print(f"  trained: {label}")

    print("\nAblation on the shifted test set (drop one signal at a time):\n")
    print(format_table(rows, cols=["method", "success_rate", "unsafe_rate",
                                   "human_effort", "mean_cost", "agreement"]))
    pd.DataFrame(rows).to_csv(_bootstrap.TABLE_DIR / "ablation.csv", index=False)
    print(f"\nWrote {(_bootstrap.TABLE_DIR / 'ablation.csv').relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
