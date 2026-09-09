"""Show the cost model is the risk dial.

As the cost of an unsafe collision rises relative to the cost of asking for
help, the optimal policy trades autonomy for safety: it defers more, and the
unsafe rate falls. Sweeping that cost traces the full autonomy/safety spectrum
the operator can choose from — evidence the cost model is doing real work, not
sitting inert behind a fixed threshold.

Writes results/plots/cost_sweep.png and results/tables/cost_sweep.csv.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uarsim.config import CostConfig, load_config
from uarsim.costs import optimal_action, realized_cost, simulate_outcome
from uarsim.types import Action, Outcome

C_UNSAFE_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def sweep(df, cm_base):
    rng = np.random.default_rng(0)
    p_correct = df["p_correct_if_act"].to_numpy()
    p_unsafe = df["p_unsafe"].to_numpy()
    rows = []
    for cu in C_UNSAFE_GRID:
        cm = CostConfig(c_wrong=cm_base.c_wrong, c_unsafe=cu,
                        c_ask=cm_base.c_ask, c_defer=cm_base.c_defer)
        acts = np.array([int(optimal_action(pc, pu, cm))
                         for pc, pu in zip(p_correct, p_unsafe)])
        n = len(acts)
        succ = unsafe = 0
        cost = 0.0
        r2 = np.random.default_rng(0)
        for i in range(n):
            a = Action(int(acts[i]))
            out, _ = simulate_outcome(a, float(p_correct[i]), float(p_unsafe[i]), r2)
            cost += realized_cost(a, out, cm)
            succ += out == Outcome.SUCCESS
            unsafe += out == Outcome.UNSAFE
        rows.append({
            "c_unsafe": cu,
            "act": float(np.mean(acts == int(Action.ACT))),
            "ask": float(np.mean(acts == int(Action.ASK))),
            "defer": float(np.mean(acts == int(Action.DEFER))),
            "success_rate": succ / n,
            "unsafe_rate": unsafe / n,
            "human_effort": float(np.mean(acts != int(Action.ACT))),
            "mean_cost": cost / n,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_shift",
                    choices=["test_id", "test_shift"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(_bootstrap.DATA_DIR / f"{args.split}.csv")

    res = sweep(df, cfg.cost)
    res.to_csv(_bootstrap.TABLE_DIR / "cost_sweep.csv", index=False)

    act_c, ask_c, def_c = "#2f9366", "#bd7c1c", "#bd3f3f"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = res["c_unsafe"]

    ax1.stackplot(x, res["act"], res["ask"], res["defer"],
                  labels=["ACT", "ASK", "DEFER"],
                  colors=[act_c, ask_c, def_c], alpha=.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("cost of an unsafe collision  (c_unsafe)")
    ax1.set_ylabel("share of decisions")
    ax1.set_title("The cost of failure sets the autonomy level")
    ax1.set_ylim(0, 1)
    ax1.axvline(3.0, color="#444", ls="--", lw=1)
    ax1.text(3.1, 0.04, "default", fontsize=8, color="#444")
    ax1.legend(loc="center left", framealpha=.9)

    ax2.plot(x, res["unsafe_rate"], "-o", ms=4, color=def_c, label="unsafe rate")
    ax2.plot(x, res["human_effort"], "-o", ms=4, color="#1f5fa6", label="human effort")
    ax2.set_xscale("log")
    ax2.set_xlabel("cost of an unsafe collision  (c_unsafe)")
    ax2.set_ylabel("rate")
    ax2.set_title("Costlier failure → more help, fewer collisions")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=.3)
    ax2.legend(framealpha=.9)

    fig.tight_layout()
    path = _bootstrap.PLOT_DIR / "cost_sweep.png"
    fig.savefig(path, dpi=150)
    print(res.to_string(index=False))
    print(f"\nWrote {path.relative_to(_bootstrap.ROOT)} and cost_sweep.csv")


if __name__ == "__main__":
    main()
