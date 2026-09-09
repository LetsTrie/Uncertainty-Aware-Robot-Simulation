"""Uncertainty calibration of the learned policy.

We measure *decision calibration*: when the classifier is X% confident in the
action it chooses, is that action actually the cost-optimal one X% of the time?
A well-calibrated policy's confidence tracks its reliability — which is exactly
what the ACT/ASK/DEFER system depends on. We report Expected Calibration Error
(ECE) and a reliability diagram, on both the in-distribution and shifted test
sets (calibration typically degrades under shift).

Writes results/plots/calibration.png and results/tables/calibration.csv.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uarsim.data import features_matrix
from uarsim.policies import LearnedPolicy


def reliability(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            rows.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc = float(correct[m].mean())
        c = float(conf[m].mean())
        rows.append((0.5 * (lo + hi), c, acc, int(m.sum())))
        ece += (m.sum() / n) * abs(acc - c)
    return rows, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    model_path = _bootstrap.MODEL_DIR / "learned.pt"
    if not model_path.exists():
        raise SystemExit("Train first: python scripts/train_policy.py")
    policy = LearnedPolicy.from_path(model_path)

    splits = {"in-distribution": "test_id.csv", "distribution-shift": "test_shift.csv"}
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")

    table_rows = []
    for name, fname in splits.items():
        df = pd.read_csv(_bootstrap.DATA_DIR / fname)
        X = features_matrix(df)
        probs = policy.probabilities(X)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = (pred == df["optimal_action"].to_numpy()).astype(float)
        rows, ece = reliability(conf, correct)
        print(f"{name:>20}:  ECE = {ece:.4f}   accuracy = {correct.mean():.3f}")

        cs = [r[1] for r in rows]
        accs = [r[2] for r in rows]
        plt.plot(cs, accs, "-o", ms=4, label=f"{name} (ECE={ece:.3f})")
        for centre, c, acc, cnt in rows:
            table_rows.append({"split": name, "bin_center": centre,
                               "mean_confidence": c, "accuracy": acc, "count": cnt})

    plt.xlabel("Predicted confidence")
    plt.ylabel("Actual fraction optimal")
    plt.title("Reliability diagram (decision calibration)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = _bootstrap.PLOT_DIR / "calibration.png"
    plt.savefig(path, dpi=150)
    pd.DataFrame(table_rows).to_csv(_bootstrap.TABLE_DIR / "calibration.csv", index=False)
    print(f"Wrote {path.relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
