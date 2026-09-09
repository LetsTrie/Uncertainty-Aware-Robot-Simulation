"""Main experiment: compare every policy on both test sets.

Writes results/tables/main_results.csv and prints the comparison table.

Run generate_dataset.py and train_policy.py first.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from uarsim.config import load_config
from uarsim.metrics import evaluate_policy, format_table
from uarsim.registry import build_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    policies = build_policies(cfg, _bootstrap.MODEL_DIR)
    test_sets = {
        "in-distribution": pd.read_csv(_bootstrap.DATA_DIR / "test_id.csv"),
        "distribution-shift": pd.read_csv(_bootstrap.DATA_DIR / "test_shift.csv"),
    }

    all_rows = []
    for split_name, df in test_sets.items():
        print(f"\n=== Test set: {split_name}  (n={len(df):,}) ===")
        rows = []
        for label, pol in policies:
            m = evaluate_policy(pol, df, cfg.cost)
            m["method"] = label
            m["split"] = split_name
            rows.append(m)
        print(format_table(rows))
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    cols = ["split", "method", "success_rate", "unsafe_rate", "wrong_rate",
            "ask_rate", "defer_rate", "human_effort", "mean_cost", "agreement", "n"]
    out = out[cols]
    path = _bootstrap.TABLE_DIR / "main_results.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path.relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
