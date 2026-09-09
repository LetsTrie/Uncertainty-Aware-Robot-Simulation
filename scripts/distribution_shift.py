"""Distribution-shift analysis (RQ3).

Hypothesis: under distribution shift, an uncertainty-aware policy should ask for
help *more* rather than confidently make more mistakes. This script reports, per
policy, the change in human effort and error rate from the in-distribution test
set to the shifted one.

Writes results/tables/distribution_shift.csv.
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

    id_df = pd.read_csv(_bootstrap.DATA_DIR / "test_id.csv")
    shift_df = pd.read_csv(_bootstrap.DATA_DIR / "test_shift.csv")
    policies = build_policies(cfg, _bootstrap.MODEL_DIR)

    rows = []
    for label, pol in policies:
        a = evaluate_policy(pol, id_df, cfg.cost)
        b = evaluate_policy(pol, shift_df, cfg.cost)
        rows.append({
            "method": label,
            "effort_id": a["human_effort"],
            "effort_shift": b["human_effort"],
            "d_effort": b["human_effort"] - a["human_effort"],
            "err_id": a["unsafe_rate"] + a["wrong_rate"],
            "err_shift": b["unsafe_rate"] + b["wrong_rate"],
            "d_err": (b["unsafe_rate"] + b["wrong_rate"])
                     - (a["unsafe_rate"] + a["wrong_rate"]),
        })

    print("\nDistribution shift: in-distribution  ->  shifted\n")
    print(format_table(rows, cols=["method", "effort_id", "effort_shift",
                                   "d_effort", "err_id", "err_shift", "d_err"]))
    print("\nReading: a good uncertainty-aware policy shows POSITIVE d_effort "
          "(asks/defers more)\nwhile keeping d_err small — it converts new "
          "uncertainty into help requests, not mistakes.")

    pd.DataFrame(rows).to_csv(_bootstrap.TABLE_DIR / "distribution_shift.csv",
                              index=False)
    print(f"\nWrote {(_bootstrap.TABLE_DIR / 'distribution_shift.csv').relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
