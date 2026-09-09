"""Train the learned policies on the generated dataset.

Trains two models and saves them to results/models/:
  learned.pt        classifier over the oracle optimal action (cross-entropy)
  cost_sensitive.pt regressor of the three expected costs (MSE), argmin at test

Usage:
  python scripts/train_policy.py
  python scripts/train_policy.py --epochs 20 --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from uarsim.config import load_config
from uarsim.data import FEATURE_NAMES, cost_matrix, features_matrix
from uarsim.nn import MLP, get_device, save_model


def _loaders(X, y, batch, device, shuffle=True):
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(Xt)
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, batch):
            j = idx[i:i + batch]
            yield Xt[j], yt[j]
            if i + batch >= n:
                break
        break


def _train(model, X, y, task, tcfg, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                           weight_decay=tcfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss() if task == "classifier" else nn.MSELoss()
    for ep in range(tcfg.epochs):
        total, nb = 0.0, 0
        for xb, yb in _loaders(X, y, tcfg.batch_size, device):
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if (ep + 1) % max(1, tcfg.epochs // 8) == 0 or ep == 0:
            print(f"    epoch {ep+1:>3}/{tcfg.epochs}  loss={total/nb:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    tcfg = cfg.train
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = get_device()
    print(f"Device: {device}")

    train_df = pd.read_csv(_bootstrap.DATA_DIR / "train.csv")
    X = features_matrix(train_df)
    print(f"Training on {len(X):,} episodes, {len(FEATURE_NAMES)} features.")

    # --- Classifier (learned policy) --------------------------------------
    print("\n[1/2] Classifier over optimal action:")
    y_cls = train_df["optimal_action"].to_numpy(dtype=np.int64)
    clf = MLP(in_dim=len(FEATURE_NAMES), hidden=tcfg.hidden, out_dim=3)
    _train(clf, X, y_cls, "classifier", tcfg, device)
    save_model(clf, _bootstrap.MODEL_DIR / "learned.pt", {
        "kind": "classifier", "in_dim": len(FEATURE_NAMES),
        "hidden": list(tcfg.hidden), "out_dim": 3, "features": FEATURE_NAMES,
    })

    # --- Cost regressor (cost-sensitive policy) ---------------------------
    print("\n[2/2] Cost regressor (expected costs):")
    y_cost = cost_matrix(train_df)
    reg = MLP(in_dim=len(FEATURE_NAMES), hidden=tcfg.hidden, out_dim=3)
    _train(reg, X, y_cost, "cost", tcfg, device)
    save_model(reg, _bootstrap.MODEL_DIR / "cost_sensitive.pt", {
        "kind": "cost", "in_dim": len(FEATURE_NAMES),
        "hidden": list(tcfg.hidden), "out_dim": 3, "features": FEATURE_NAMES,
    })

    print(f"\nSaved models to {_bootstrap.MODEL_DIR.relative_to(_bootstrap.ROOT)}/")


if __name__ == "__main__":
    main()
