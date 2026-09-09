"""Evaluate the trained policies on a chosen scene backend.

Reuses the trained models and the metrics; only the *scene source* changes.
Start on a laptop with the synthetic backend to confirm everything works, then
run the same command with --backend maniskill on a GPU box.

Examples:
  # laptop, no GPU — proves the flow end to end:
  python scripts/run_backend.py --backend synthetic --n 2000

  # on a GPU box (after installing requirements-maniskill.txt):
  python scripts/run_backend.py --backend maniskill --n 500 --num-cubes 3
  python scripts/run_backend.py --backend maniskill --n 500 --mode shift

Outputs results/tables/backend_results.csv and (maniskill only) a few sample
frames under results/plots/backend_frames/.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uarsim.config import load_config
from uarsim.metrics import evaluate_policy, format_table
from uarsim.backends.backend import sample_to_row
from uarsim.registry import build_policies


def make_backend(name, cfg, args):
    if name == "synthetic":
        from uarsim.backends.synthetic_backend import SyntheticBackend
        return SyntheticBackend(cfg, mode=args.mode)
    if name == "maniskill":
        from uarsim.backends.maniskill_backend import ManiSkillBackend
        return ManiSkillBackend(cfg, num_cubes=args.num_cubes, mode=args.mode,
                                seed=args.seed)
    raise SystemExit(f"unknown backend: {name}")


def _save_frames(frames, n=6):
    if not frames:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = _bootstrap.PLOT_DIR / "backend_frames"
    out.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames[:n]):
        if fr is None:
            continue
        plt.figure(figsize=(3, 3))
        plt.imshow(np.asarray(fr).astype("uint8"))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"episode_{i:02d}.png", dpi=120)
        plt.close()
    print(f"Saved sample frames to {out.relative_to(_bootstrap.ROOT)}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="synthetic",
                    choices=["synthetic", "maniskill"])
    ap.add_argument("--n", type=int, default=1000, help="episodes to evaluate")
    ap.add_argument("--mode", default="train", choices=["train", "shift"])
    ap.add_argument("--num-cubes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not (_bootstrap.MODEL_DIR / "learned.pt").exists():
        raise SystemExit("Train the policies first: python scripts/train_policy.py")

    print(f"Backend: {args.backend}  |  episodes: {args.n}  |  mode: {args.mode}")
    backend = make_backend(args.backend, cfg, args)
    rng = np.random.default_rng(args.seed)

    rows, frames = [], []
    t0 = time.time()
    for i in range(args.n):
        sample = backend.sample_episode(rng)
        rows.append(sample_to_row(sample, cfg.cost))
        if sample.frame is not None and len(frames) < 6:
            frames.append(sample.frame)
        if (i + 1) % max(1, args.n // 10) == 0:
            print(f"  {i+1:>5}/{args.n} episodes  ({time.time()-t0:.1f}s)")
    backend.close()

    df = pd.DataFrame(rows)
    policies = build_policies(cfg, _bootstrap.MODEL_DIR)
    print(f"\n=== {args.backend} backend, mode={args.mode}, n={len(df):,} ===")
    results = []
    for label, pol in policies:
        m = evaluate_policy(pol, df, cfg.cost)
        m["method"] = label
        results.append(m)
    print(format_table(results))

    path = _bootstrap.TABLE_DIR / "backend_results.csv"
    pd.DataFrame(results)[
        ["method", "success_rate", "unsafe_rate", "ask_rate", "defer_rate",
         "human_effort", "mean_cost", "agreement", "n"]
    ].to_csv(path, index=False)
    print(f"\nWrote {path.relative_to(_bootstrap.ROOT)}")
    _save_frames(frames)


if __name__ == "__main__":
    main()
