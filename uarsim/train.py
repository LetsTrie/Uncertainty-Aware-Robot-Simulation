"""Shared training loops for the learned policies.

Factored out so train_policy.py, ablation.py, run_seeds.py and the cost sweep
all train the same way instead of each carrying its own loop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nn import MLP, get_device


def _fit(model, X, y, loss_fn, cfg, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y)).to(device)
    n = len(Xt)
    for _ in range(cfg.epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            j = idx[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[j]), yt[j])
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_classifier(X, y, cfg, device=None, hidden=None) -> nn.Module:
    """Cross-entropy classifier over the oracle optimal action."""
    device = device or get_device()
    model = MLP(in_dim=X.shape[1], hidden=hidden or cfg.hidden, out_dim=3)
    return _fit(model, X.astype(np.float32), y.astype(np.int64),
                nn.CrossEntropyLoss(), cfg, device)


def train_cost(X, Y, cfg, device=None, hidden=None) -> nn.Module:
    """MSE regressor of the three expected costs (for the cost-sensitive policy)."""
    device = device or get_device()
    model = MLP(in_dim=X.shape[1], hidden=hidden or cfg.hidden, out_dim=Y.shape[1])
    return _fit(model, X.astype(np.float32), Y.astype(np.float32),
                nn.MSELoss(), cfg, device)


def train_regressor(X, y, cfg, device=None, hidden=None) -> nn.Module:
    """Generic 1-output regressor (used by the learned uncertainty estimators)."""
    device = device or get_device()
    y = y.reshape(-1, 1).astype(np.float32)
    model = MLP(in_dim=X.shape[1], hidden=hidden or cfg.hidden, out_dim=1)
    return _fit(model, X.astype(np.float32), y, nn.MSELoss(), cfg, device)
