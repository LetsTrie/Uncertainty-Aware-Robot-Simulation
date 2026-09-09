"""Collect a full result set across several scene configurations in one run.

Sweeps a small grid of (mode, num_cubes), evaluates every policy on each, saves
a few rendered frames per configuration, and writes one combined table to
results/tables/maniskill_results.csv.

Meant to be run on a GPU box with the real backend:
  /content/ms/bin/python scripts/collect_maniskill.py --n 500

It also runs with --backend synthetic (num_cubes is ignored there) so the loop
can be checked on a laptop with no GPU. Each configuration is wrapped so that if
one fails, the others still complete and you keep partial results.
"""
from __future__ import annotations

import argparse
import time
import traceback

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uarsim.config import load_config
from uarsim.metrics import evaluate_policy, format_table
from uarsim.backends.backend import sample_to_row
from uarsim.registry import build_policies

# (mode, num_cubes) configurations to sweep. num_cubes only affects ManiSkill.
GRID = [("train", 3), ("shift", 3), ("shift", 4)]


def make_backend(name, cfg, mode, num_cubes, seed):
    if name == "synthetic":
        from uarsim.backends.synthetic_backend import SyntheticBackend
        return SyntheticBackend(cfg, mode=mode)
    if name == "maniskill":
        from uarsim.backends.maniskill_backend import ManiSkillBackend
        return ManiSkillBackend(cfg, num_cubes=num_cubes, mode=mode, seed=seed)
    raise SystemExit(f"unknown backend: {name}")


def _save_frames(frames, label, n=4):
    if not frames:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = _bootstrap.PLOT_DIR / "maniskill_frames" / label
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
    print(f"    frames -> {out.relative_to(_bootstrap.ROOT)}/")


def run_config(backend_name, cfg, mode, num_cubes, n, seed):
    label = f"{mode}_c{num_cubes}"
    print(f"\n=== config: {label}  (n={n}) ===")
    backend = make_backend(backend_name, cfg, mode, num_cubes, seed)
    rng = np.random.default_rng(seed)

    rows, frames = [], []
    t0 = time.time()
    for i in range(n):
        sample = backend.sample_episode(rng)
        rows.append(sample_to_row(sample, cfg.cost))
        if sample.frame is not None and len(frames) < 4:
            frames.append(sample.frame)
        if (i + 1) % max(1, n // 5) == 0:
            print(f"    {i+1:>5}/{n}  ({time.time()-t0:.1f}s)")
    backend.close()

    df = pd.DataFrame(rows)
    policies = build_policies(cfg, _bootstrap.MODEL_DIR)
    results = []
    for method, pol in policies:
        m = evaluate_policy(pol, df, cfg.cost)
        m.update({"config": label, "mode": mode, "num_cubes": num_cubes,
                  "method": method})
        results.append(m)
    print(format_table(results))
    _save_frames(frames, label)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="maniskill",
                    choices=["synthetic", "maniskill"])
    ap.add_argument("--n", type=int, default=500, help="episodes per config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not (_bootstrap.MODEL_DIR / "learned.pt").exists():
        raise SystemExit("Train the policies first: python scripts/train_policy.py")

    all_results = []
    for k, (mode, num_cubes) in enumerate(GRID):
        try:
            all_results += run_config(args.backend, cfg, mode, num_cubes,
                                      args.n, args.seed + k)
        except Exception:
            print(f"\n(!) config {mode}_c{num_cubes} failed — continuing:\n"
                  + traceback.format_exc())

    if not all_results:
        raise SystemExit("No configuration completed successfully.")

    out = pd.DataFrame(all_results)[
        ["config", "mode", "num_cubes", "method", "success_rate", "unsafe_rate",
         "ask_rate", "defer_rate", "human_effort", "mean_cost", "agreement", "n"]
    ]
    path = _bootstrap.TABLE_DIR / "maniskill_results.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote combined table -> {path.relative_to(_bootstrap.ROOT)}")

    # Headline summary: the two learned policies across configurations.
    print("\n=== summary: learned policies across configurations ===")
    keep = out[out["method"].isin(["Learned Policy", "Cost-aware Learned"])]
    print(keep[["config", "method", "success_rate", "unsafe_rate",
                "human_effort", "mean_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
