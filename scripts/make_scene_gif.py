"""Render a montage GIF of ManiSkill scenes.  (GPU only — run on Colab.)

Resets the TabletopChoice scene many times and stitches the rendered frames into
an animated GIF — a quick visual of the variety of scenes the policy judges.

  /content/ms/bin/python scripts/make_scene_gif.py --n 24
Writes results/plots/scenes.gif
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from uarsim.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="number of scenes")
    ap.add_argument("--num-cubes", type=int, default=3)
    ap.add_argument("--mode", default="shift", choices=["train", "shift"])
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from uarsim.backends.maniskill_backend import ManiSkillBackend  # GPU import
    import imageio  # ships with ManiSkill

    backend = ManiSkillBackend(cfg, num_cubes=args.num_cubes, mode=args.mode)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(args.n):
        s = backend.sample_episode(rng)
        if s.frame is not None:
            frames.append(np.asarray(s.frame).astype("uint8"))
    backend.close()

    if not frames:
        raise SystemExit("No frames captured — is render_mode set on the env?")
    out = _bootstrap.PLOT_DIR / "scenes.gif"
    imageio.mimsave(out, frames, duration=1.0 / args.fps, loop=0)
    print(f"Wrote {out.relative_to(_bootstrap.ROOT)}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
