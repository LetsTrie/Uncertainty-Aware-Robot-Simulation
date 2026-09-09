"""Read a real ManiSkill scene and reuse the synthetic-world decision math on it.

Flow per episode:
  1. reset the ManiSkill env  -> object poses (+ an RGB frame, + segmentation)
  2. translate that real state into our ``Scene`` / ``Obj`` / ``Instruction``
  3. call the SAME helpers (oracle latents, uncertainty signals, label)
  4. hand back an :class:`EpisodeSample`

So the only genuinely new, GPU-dependent code is 1-2 (reading the sim).
Everything that decides ACT/ASK/DEFER is the code already exercised by the
synthetic backend.

    !!! Validate on the GPU box. Object-pose and segmentation access can vary by
    ManiSkill version; the risky spots are wrapped in try/except with clear
    fallbacks, and the exact lines are marked `# MANISKILL:`.
"""
from __future__ import annotations

import numpy as np

from ..config import Config, SceneConfig
from ..env import objects as vocab
from ..env.objects import point_segment_distance
from ..env.oracle import compute_latents, label_episode
from ..env.scene_generator import _clutter
from ..env.task_generator import make_instruction, matching_ids
from ..types import Instruction, Obj, Scene
from ..uncertainty import compute_signals
from .backend import Backend, EpisodeSample, episode_to_sample

# Affine map from ManiSkill table meters -> our abstract table frame
# (x in [-1,1], y in [0,1]). Objects are randomized within ~+-0.12 m.
_XY_SPAN = 0.12


def _to_frame(mx: float, my: float) -> tuple[float, float]:
    x = float(np.clip(mx / _XY_SPAN, -1.0, 1.0))
    y = float(np.clip((my + _XY_SPAN) / (2 * _XY_SPAN), 0.0, 1.0))
    return x, y


def _pose_xy(actor) -> tuple[float, float]:
    """Best-effort read of an actor's (x, y) world position in meters."""
    p = actor.pose.p  # MANISKILL: expected shape (num_envs, 3) tensor
    try:
        arr = p[0].detach().cpu().numpy()
    except Exception:
        arr = np.asarray(p).reshape(-1, 3)[0]
    return float(arr[0]), float(arr[1])


class ManiSkillBackend(Backend):
    name = "maniskill"

    def __init__(self, cfg: Config | None = None, num_cubes: int = 3,
                 mode: str = "train", obs_mode: str = "rgbd", seed: int = 0):
        from .maniskill_env import CUBE_COLORS, ensure_available  # noqa
        ensure_available()
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  registers built-in envs
        from . import maniskill_env  # noqa: F401  registers TabletopChoice-v0

        self.cfg = cfg or Config()
        self.scene_cfg: SceneConfig = self.cfg.scene if mode == "train" else self.cfg.shift
        self.mode = mode
        self.num_cubes = num_cubes
        self._colors = list(CUBE_COLORS.keys())

        # MANISKILL: env creation. render_mode="rgb_array" lets us grab frames.
        self.env = gym.make(
            "TabletopChoice-v0", num_envs=1, num_cubes=num_cubes,
            obs_mode=obs_mode, render_mode="rgb_array",
        )
        self._last_obs = None
        self.env.reset(seed=seed)

    # -- translation: real sim state -> our dataclasses ---------------------
    def _build_scene(self, rng) -> tuple[Scene, list[Obj], Obj]:
        base = self.env.unwrapped
        objs: list[Obj] = []
        for i, actor in enumerate(base.cubes):
            mx, my = _pose_xy(actor)
            x, y = _to_frame(mx, my)
            color = self._colors[i % len(self._colors)]
            # Perception confidence: high by default, optionally refined from
            # segmentation below. Kept familiar (non-novel) for a first pass.
            conf = float(min(0.99, max(0.5, 0.97 - rng.uniform(0, 0.05))))
            reachable = abs(x) < 0.95  # crude workspace bound in our frame
            objs.append(Obj(id=f"cube_{i}", category="cube", color=color,
                            x=x, y=y, reachable=reachable, fragile=False,
                            novel=False, perception_conf=conf))

        # The fragile object is a hazard, not a valid target.
        fx, fy = _pose_xy(base.fragile)
        fragile_xy = _to_frame(fx, fy)
        fragile_obj = Obj(id="fragile", category="glass", color="clear",
                          x=fragile_xy[0], y=fragile_xy[1], reachable=True,
                          fragile=True, novel=False, perception_conf=0.9)

        self._refine_perception_from_segmentation(objs)  # optional, safe

        all_objs = objs + [fragile_obj]
        clutter = _clutter(all_objs)
        # target chosen among the real cubes only
        target = objs[int(rng.integers(0, len(objs)))]
        fragile_in_path = point_segment_distance(
            fragile_obj.x, fragile_obj.y, vocab.BASE[0], vocab.BASE[1],
            target.x, target.y) < 0.22
        scene = Scene(objects=all_objs, landmarks={"laptop": (0.0, 0.9)},
                      fragile_in_path=fragile_in_path, clutter=clutter)
        return scene, objs, target

    def _refine_perception_from_segmentation(self, objs):
        """Optionally lower perception_conf for occluded cubes.

        Uses the base camera's segmentation if present. Any failure here is
        non-fatal — we simply keep the default confidences.
        """
        obs = self._last_obs
        if not isinstance(obs, dict):
            return
        try:  # MANISKILL: segmentation location/shape varies by version.
            seg = obs["sensor_data"]["base_camera"]["segmentation"]
            seg = np.asarray(seg).reshape(-1)
            total = seg.size
            if total == 0:
                return
            # Fewer visible pixels (heavier occlusion) -> less confidence.
            unique, counts = np.unique(seg, return_counts=True)
            frac = counts / total
            median_frac = float(np.median(frac[frac > 0])) if frac.size else 0.0
            for o in objs:
                # Scale confidence by how "visible" a typical object is.
                o.perception_conf = float(np.clip(
                    o.perception_conf * (0.6 + 0.4 * min(1.0, median_frac * 20)),
                    0.5, 0.99))
        except Exception:
            return

    def sample_episode(self, rng) -> EpisodeSample:
        # MANISKILL: reset returns (obs, info) in the Gymnasium API.
        out = self.env.reset()
        self._last_obs = out[0] if isinstance(out, tuple) else out

        scene, cubes, target = self._build_scene(rng)

        # Instruction + ambiguity: identical logic to the synthetic world.
        cfg = self.scene_cfg
        mode_probs = (cfg.p_full_ref, cfg.p_cat_only, cfg.p_generic)
        instruction: Instruction = make_instruction(target, "laptop", mode_probs, rng)
        matches = matching_ids(instruction, cubes)
        if target.id not in matches:
            matches.append(target.id)

        latents = compute_latents(scene, target, len(matches))
        latents.update(label_episode(latents, self.cfg.cost))
        signals = compute_signals(scene, target, len(matches), latents, cfg, rng)

        from ..types import Episode
        ep = Episode(scene=scene, instruction=instruction,
                     true_target_id=target.id, matches=matches,
                     latents=latents, signals=signals,
                     meta={"mode": self.mode, "n_objects": len(scene.objects),
                           "n_matching": len(matches),
                           "obstacle_count": sum(o.fragile for o in scene.objects),
                           "target_novel": 0,
                           "fragile_in_path": int(scene.fragile_in_path),
                           "instruction": instruction.text})
        sample = episode_to_sample(ep)

        # Attach a render frame for optional logging/video.
        try:  # MANISKILL: render() returns a tensor/array of frames.
            frame = self.env.render()
            arr = frame[0] if hasattr(frame, "__len__") else frame
            sample.frame = np.asarray(arr.detach().cpu()) if hasattr(arr, "detach") \
                else np.asarray(arr)
        except Exception:
            sample.frame = None
        return sample

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
