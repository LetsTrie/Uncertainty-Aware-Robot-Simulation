"""The small interface every scene backend implements.

A backend's only job is to hand back, per episode, the same information the
synthetic world produces — the 10 observed features and the privileged latents —
so the existing metrics and policies work verbatim. Where those numbers come
from (a Python generator, or a rendered ManiSkill scene) is the backend's
private business.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..costs import optimal_action
from ..data import FEATURE_NAMES
from ..types import Episode


@dataclass
class EpisodeSample:
    """One episode's worth of policy inputs + ground truth."""

    features: np.ndarray          # shape (10,), ordered like data.FEATURE_NAMES
    latents: dict                 # p_correct_if_act, p_unsafe, target_distance
    meta: dict = field(default_factory=dict)  # instruction text, counts, mode...
    frame: object = None          # optional RGB image (H,W,3) for logging/video


def episode_to_sample(ep: Episode) -> EpisodeSample:
    """Convert a fully-built :class:`Episode` into an :class:`EpisodeSample`.

    Both backends construct an ``Episode`` (via the shared world helpers) and
    then call this, so features are guaranteed to line up with ``FEATURE_NAMES``.
    """
    from ..data import episode_to_row

    row = episode_to_row(ep)
    features = np.array([row[name] for name in FEATURE_NAMES], dtype=np.float32)
    latents = {
        "p_correct_if_act": ep.latents["p_correct_if_act"],
        "p_unsafe": ep.latents["p_unsafe"],
        "target_distance": ep.latents["target_distance"],
    }
    return EpisodeSample(features=features, latents=latents, meta=dict(ep.meta),
                         frame=None)


class Backend:
    """Abstract scene source. Subclasses implement :meth:`sample_episode`."""

    name = "backend"

    def sample_episode(self, rng) -> EpisodeSample:
        raise NotImplementedError

    def close(self):
        """Release resources (e.g. the ManiSkill env). Safe no-op by default."""


def sample_to_row(sample: EpisodeSample, cm) -> dict:
    """Flatten an :class:`EpisodeSample` into a DataFrame row the metrics
    code understands.

    We compute the oracle ``optimal_action`` here (same cost model as the
    synthetic world) so evaluation can report agreement with the oracle.
    """
    row = dict(zip(FEATURE_NAMES, [float(v) for v in sample.features]))
    row["p_correct_if_act"] = sample.latents["p_correct_if_act"]
    row["p_unsafe"] = sample.latents["p_unsafe"]
    row["optimal_action"] = int(optimal_action(
        sample.latents["p_correct_if_act"], sample.latents["p_unsafe"], cm))
    row["mode"] = sample.meta.get("mode", "backend")
    return row
