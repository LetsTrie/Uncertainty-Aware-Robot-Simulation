"""Replace a simulated uncertainty signal with a learned estimator.

The five signals are modeled, not produced by real models — a staged design.
This script demonstrates the seam is real: it trains a small network to predict
the privileged collision probability from the *other* observable features
(no access to the true u_safety), swaps that learned estimate in for the
simulated u_safety, retrains the policy on it, and shows performance holds.

If a learned estimate can stand in for the simulated signal here, a real
perception/planning model can stand in for it later — without touching the
policy, cost model, or metrics.

Writes results/tables/learned_signals.csv.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uarsim.config import load_config
from uarsim.data import FEATURE_NAMES, features_matrix
from uarsim.metrics import evaluate_policy, format_table
from uarsim.nn import forward_numpy, get_device
from uarsim.policies import LearnedPolicy
from uarsim.train import train_classifier, train_regressor

SAFETY = FEATURE_NAMES.index("u_safety")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.train.epochs = args.epochs
    device = get_device()

    train = pd.read_csv(_bootstrap.DATA_DIR / "train.csv")
    shift = pd.read_csv(_bootstrap.DATA_DIR / "test_shift.csv")
    Xtr, Xsh = features_matrix(train), features_matrix(shift)
    ytr = train["optimal_action"].to_numpy()

    # 1) Learn a safety-uncertainty estimator from every feature EXCEPT u_safety.
    keep = [i for i in range(len(FEATURE_NAMES)) if i != SAFETY]
    g = train_regressor(Xtr[:, keep], train["p_unsafe"].to_numpy(), cfg.train, device)
    est_tr = np.clip(forward_numpy(g, Xtr[:, keep], device).ravel(), 0, 1)
    est_sh = np.clip(forward_numpy(g, Xsh[:, keep], device).ravel(), 0, 1)

    mse = float(np.mean((est_tr - train["p_unsafe"].to_numpy()) ** 2))
    corr = float(np.corrcoef(est_sh, shift["p_unsafe"].to_numpy())[0, 1])
    print(f"Learned safety estimator:  train MSE = {mse:.4f}   "
          f"corr with true risk (shift) = {corr:.3f}")

    # 2) Two policies: one on the simulated u_safety, one on the learned estimate.
    clf_sim = train_classifier(Xtr, ytr, cfg.train, device)
    Xtr_learn = Xtr.copy(); Xtr_learn[:, SAFETY] = est_tr
    clf_learn = train_classifier(Xtr_learn, ytr, cfg.train, device)

    shift_learn = shift.copy(); shift_learn["u_safety"] = est_sh

    rows = []
    m = evaluate_policy(LearnedPolicy(clf_sim, device), shift, cfg.cost)
    m["method"] = "Simulated u_safety"; rows.append(m)
    m = evaluate_policy(LearnedPolicy(clf_learn, device), shift_learn, cfg.cost)
    m["method"] = "Learned u_safety"; rows.append(m)

    print("\nPolicy on shifted scenes — simulated signal vs learned estimate:\n")
    print(format_table(rows, cols=["method", "success_rate", "unsafe_rate",
                                   "human_effort", "mean_cost", "agreement"]))
    pd.DataFrame(rows).to_csv(_bootstrap.TABLE_DIR / "learned_signals.csv", index=False)
    print(f"\nWrote {(_bootstrap.TABLE_DIR / 'learned_signals.csv').relative_to(_bootstrap.ROOT)}")
    print("\nComparable numbers => the pipeline runs on a learned signal, not just "
          "the simulated one.")


if __name__ == "__main__":
    main()
