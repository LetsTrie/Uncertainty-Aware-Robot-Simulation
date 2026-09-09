"""A GPU-free backend that generates abstract tabletop scenes.

This is the reference backend: it runs anywhere and gives the numbers every
other backend is compared against. It reuses the synthetic scene generator
directly, so its behavior matches the standalone dataset scripts.
"""
from __future__ import annotations

from ..config import Config
from ..env.scene_generator import sample_episode
from .backend import Backend, EpisodeSample, episode_to_sample


class SyntheticBackend(Backend):
    name = "synthetic"

    def __init__(self, cfg: Config | None = None, mode: str = "train"):
        self.cfg = cfg or Config()
        self.mode = mode  # "train" (in-distribution) or "shift" (harder scenes)

    def sample_episode(self, rng) -> EpisodeSample:
        ep = sample_episode(self.cfg, self.mode, rng)
        return episode_to_sample(ep)
