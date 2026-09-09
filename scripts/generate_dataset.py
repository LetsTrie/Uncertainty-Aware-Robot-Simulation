"""Generate the episode dataset.

Writes four CSVs to results/data/:
  train.csv, val.csv        (in-distribution)
  test_id.csv               (in-distribution test)
  test_shift.csv            (out-of-distribution test)

Usage:
  python scripts/generate_dataset.py                 # defaults
  python scripts/generate_dataset.py --config configs/experiment.yaml
  python scripts/generate_dataset.py --n-train 5000  # quick smoke run
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import pandas as pd

from uarsim.config import load_config
from uarsim.data import episode_to_row
from uarsim.env.scene_generator import episode_stream


def _build(cfg, mode, n, seed) -> pd.DataFrame:
    t0 = time.time()
    rows = [episode_to_row(ep) for ep in episode_stream(cfg, mode, n, seed)]
    df = pd.DataFrame(rows)
    print(f"  {mode:>6} split: {len(df):>7,} episodes  ({time.time()-t0:.1f}s)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test-id", type=int, default=None)
    ap.add_argument("--n-test-shift", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg.data
    if args.n_train is not None:
        d.n_train = args.n_train
    if args.n_val is not None:
        d.n_val = args.n_val
    if args.n_test_id is not None:
        d.n_test_id = args.n_test_id
    if args.n_test_shift is not None:
        d.n_test_shift = args.n_test_shift

    base = d.seed
    print("Generating episodes...")
    splits = {
        "train": _build(cfg, "train", d.n_train, base + 1),
        "val": _build(cfg, "train", d.n_val, base + 2),
        "test_id": _build(cfg, "train", d.n_test_id, base + 3),
        "test_shift": _build(cfg, "shift", d.n_test_shift, base + 4),
    }
    for name, df in splits.items():
        path = _bootstrap.DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path.relative_to(_bootstrap.ROOT)}")

    # Quick label distribution sanity check.
    print("\nOptimal-action distribution (train):")
    counts = splits["train"]["optimal_action"].value_counts().sort_index()
    for a, c in counts.items():
        name = {0: "ACT", 1: "ASK", 2: "DEFER"}[int(a)]
        print(f"  {name:<6} {c:>7,}  ({c/len(splits['train']):.1%})")


if __name__ == "__main__":
    main()
