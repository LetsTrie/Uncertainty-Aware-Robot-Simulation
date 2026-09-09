"""Sample randomized episodes.

An episode = a tabletop scene + a natural-language instruction + privileged
ground truth (true target, latents, oracle label) + observed uncertainty
signals. ``mode="train"`` draws in-distribution scenes; ``mode="shift"`` draws
the harder out-of-distribution scenes for the distribution-shift experiments.
"""
from __future__ import annotations

import numpy as np

from ..config import Config, SceneConfig
from ..types import Episode, Obj, Scene
from ..uncertainty import compute_signals
from . import objects as vocab
from .objects import point_segment_distance
from .oracle import compute_latents, label_episode
from .task_generator import make_instruction, matching_ids


def _sample_object(oid: str, scene_cfg: SceneConfig, rng) -> Obj:
    novel = rng.random() < scene_cfg.novel_prob
    if novel:
        category = rng.choice(vocab.NOVEL_CATEGORIES)
        # A novel object may also carry a novel color.
        color = rng.choice(
            vocab.NOVEL_COLORS if rng.random() < 0.5 else vocab.TRAIN_COLORS)
    else:
        category = rng.choice(vocab.TRAIN_CATEGORIES)
        color = rng.choice(vocab.TRAIN_COLORS)

    x = float(rng.uniform(-1.0, 1.0))
    y = float(rng.uniform(0.0, 1.0))
    reachable = rng.random() > scene_cfg.p_unreachable
    fragile = rng.random() < scene_cfg.p_fragile

    # Identification confidence: high for familiar objects, degraded by novelty.
    conf = 0.97
    if novel:
        conf -= rng.uniform(0.30, 0.55)
    conf -= rng.uniform(0.0, 0.05)  # generic sensing noise
    conf = float(min(0.99, max(0.35, conf)))

    return Obj(id=oid, category=category, color=color, x=x, y=y,
               reachable=reachable, fragile=fragile, novel=novel,
               perception_conf=conf)


def _clutter(objs) -> float:
    """Scene density in [0, 1]: more objects, packed closer => higher."""
    n = len(objs)
    density = (n - 2) / 6.0  # count component
    if n >= 2:
        pts = np.array([[o.x, o.y] for o in objs])
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        d = d[~np.eye(n, dtype=bool)]
        closeness = 1.0 - min(1.0, float(d.mean()) / 1.5)
    else:
        closeness = 0.0
    return float(min(1.0, max(0.0, 0.5 * density + 0.5 * closeness)))


def sample_episode(cfg: Config, mode: str, rng) -> Episode:
    scene_cfg: SceneConfig = cfg.scene if mode == "train" else cfg.shift

    n = int(rng.integers(scene_cfg.min_objects, scene_cfg.max_objects + 1))
    objs = [_sample_object(f"obj_{i}", scene_cfg, rng) for i in range(n)]

    landmark = str(rng.choice(vocab.LANDMARKS))
    landmarks = {landmark: (float(rng.uniform(-1, 1)), float(rng.uniform(0, 1)))}

    # 'Fragile in path': a fragile object near the base->target line.
    target = objs[int(rng.integers(0, n))]
    fragile_in_path = any(
        o.fragile and o.id != target.id and point_segment_distance(
            o.x, o.y, vocab.BASE[0], vocab.BASE[1], target.x, target.y) < 0.22
        for o in objs
    )
    scene = Scene(objects=objs, landmarks=landmarks,
                  fragile_in_path=fragile_in_path, clutter=_clutter(objs))

    mode_probs = (scene_cfg.p_full_ref, scene_cfg.p_cat_only, scene_cfg.p_generic)
    instruction = make_instruction(target, landmark, mode_probs, rng)
    matches = matching_ids(instruction, objs)
    if target.id not in matches:  # safety net; construction guarantees this
        matches.append(target.id)

    latents = compute_latents(scene, target, len(matches))
    label = label_episode(latents, cfg.cost)
    latents.update(label)
    signals = compute_signals(scene, target, len(matches), latents, scene_cfg, rng)

    meta = {
        "mode": mode,
        "n_objects": n,
        "n_matching": len(matches),
        "obstacle_count": sum(o.fragile for o in objs),
        "target_novel": int(target.novel),
        "fragile_in_path": int(fragile_in_path),
    }
    return Episode(scene=scene, instruction=instruction,
                   true_target_id=target.id, matches=matches,
                   latents=latents, signals=signals, meta=meta)


def episode_stream(cfg: Config, mode: str, n: int, seed: int):
    """Yield ``n`` episodes with a reproducible RNG."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield sample_episode(cfg, mode, rng)
