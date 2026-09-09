"""Shared PyTorch pieces: the tiny MLP and Mac-friendly device selection.

Everything here runs comfortably on a MacBook — CPU, or Apple's MPS backend if
available. No CUDA required.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Prefer Apple MPS, then CUDA (if you later move to a GPU box), else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    """The small assistance-policy network: 10 -> hidden... -> out."""

    def __init__(self, in_dim: int = 10, hidden=(32, 64, 32), out_dim: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def save_model(model: nn.Module, path, meta: dict):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path, device=None) -> tuple[nn.Module, dict]:
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = MLP(in_dim=meta["in_dim"], hidden=tuple(meta["hidden"]),
                out_dim=meta["out_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def forward_numpy(model: nn.Module, X: np.ndarray, device=None) -> np.ndarray:
    """Run the model on a numpy feature matrix; return numpy logits/outputs."""
    device = device or next(model.parameters()).device
    # .copy() ensures a writable array (pandas .to_numpy() can be read-only),
    # which avoids a torch.from_numpy non-writable-tensor warning.
    t = torch.from_numpy(np.asarray(X, dtype=np.float32).copy()).to(device)
    return model(t).cpu().numpy()
