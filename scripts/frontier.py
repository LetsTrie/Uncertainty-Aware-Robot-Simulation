"""The autonomy/safety frontier plot (the flagship figure).

Sweeps the threshold policy's ASK band and, for each setting, measures human
effort vs. error (unsafe + wrong). Overlays the fixed corner baselines and the
learned policies as single points, so you can see how much human effort each
approach needs to reach a given level of safety.

Writes results/plots/frontier.png and results/tables/frontier.csv.
"""
from __future__ import annotations

import argparse
import copy

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")  # headless: works on any Mac with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uarsim.config import load_config
from uarsim.metrics import evaluate_policy
from uarsim.policies import (AlwaysAct, AlwaysAsk, CostSensitivePolicy,
                             LearnedPolicy, ThresholdPolicy)


def _error_rate(m):
    return m["unsafe_rate"] + m["wrong_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test_shift",
                    choices=["test_id", "test_shift"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(_bootstrap.DATA_DIR / f"{args.split}.csv")

    # Sweep the ASK band: vary t_act and t_defer symmetrically around 0.5.
    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        tc = copy.deepcopy(cfg.threshold)
        tc.t_act = float(max(0.0, t - 0.15))
        tc.t_defer = float(min(1.0, t + 0.15))
        m = evaluate_policy(ThresholdPolicy(tc), df, cfg.cost)
        sweep.append({"t": float(t), "human_effort": m["human_effort"],
                      "error_rate": _error_rate(m), "unsafe_rate": m["unsafe_rate"],
                      "mean_cost": m["mean_cost"]})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(_bootstrap.TABLE_DIR / "frontier.csv", index=False)

    # Reference points.
    points = {"Always Act": AlwaysAct(), "Always Ask": AlwaysAsk()}
    learned = _bootstrap.MODEL_DIR / "learned.pt"
    cost = _bootstrap.MODEL_DIR / "cost_sensitive.pt"
    if learned.exists():
        points["Learned"] = LearnedPolicy.from_path(learned)
    if cost.exists():
        points["Cost-aware"] = CostSensitivePolicy.from_path(cost)
    ref = {name: evaluate_policy(p, df, cfg.cost) for name, p in points.items()}

    plt.figure(figsize=(7, 5))
    plt.plot(sweep_df["human_effort"], sweep_df["error_rate"],
             "-o", ms=4, color="#444", label="Threshold sweep")
    markers = {"Always Act": "s", "Always Ask": "D",
               "Learned": "^", "Cost-aware": "*"}
    colors = {"Always Act": "#d62728", "Always Ask": "#1f77b4",
              "Learned": "#2ca02c", "Cost-aware": "#9467bd"}
    for name, m in ref.items():
        plt.scatter(m["human_effort"], _error_rate(m),
                    s=140 if name == "Cost-aware" else 90,
                    marker=markers[name], color=colors[name], zorder=5,
                    edgecolor="k", linewidth=0.6, label=name)

    plt.xlabel("Human effort  (ask rate + defer rate)")
    plt.ylabel("Error rate  (unsafe + wrong object)")
    plt.title(f"Autonomy–safety frontier ({args.split})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = _bootstrap.PLOT_DIR / "frontier.png"
    plt.savefig(path, dpi=150)
    print(f"Wrote {path.relative_to(_bootstrap.ROOT)}")
    print(f"Wrote {(_bootstrap.TABLE_DIR / 'frontier.csv').relative_to(_bootstrap.ROOT)}")


if __name__ == "__main__":
    main()
