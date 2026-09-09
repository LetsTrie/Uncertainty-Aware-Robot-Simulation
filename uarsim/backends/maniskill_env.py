"""A minimal ManiSkill 3 tabletop scene.

    !!! IMPORTANT — this file touches the ManiSkill / SAPIEN API and is written
    from the ManiSkill 3 docs, but it has NOT been run on a GPU from this
    machine. Treat it as a strong first draft to validate on a GPU box. If an
    import or method name mismatches your installed ManiSkill version, paste the
    traceback back and we fix it. Until then, use --backend synthetic.

The scene: a table, a Panda arm, ``num_cubes`` differently-colored cubes, and
one pale "fragile" cube standing in for a glass. That gives us:

  * referential ambiguity — "the cube" matches all cubes, "the red cube" one;
  * a safety hazard — a fragile object that may sit in the reach path.

The robot does not need to actually grasp for the decision experiment: the
policy is evaluated on whether it chooses ACT/ASK/DEFER well, and outcomes are
derived from privileged geometry. Closed-loop execution via motion planning is
an optional extension (see MANISKILL_SETUP.md).
"""
from __future__ import annotations

import numpy as np

try:
    import sapien  # noqa: F401
    import torch
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.pose import Pose
    _MANISKILL_AVAILABLE = True
except Exception as _e:  # pragma: no cover - only importable on a GPU box
    _MANISKILL_AVAILABLE = False
    _IMPORT_ERROR = _e


# Cube colors (RGBA). The first ``num_cubes`` are the "real" targets; the
# fragile object is drawn pale and is not a valid grasp target.
CUBE_COLORS = {
    "red": [0.9, 0.1, 0.1, 1.0],
    "blue": [0.1, 0.2, 0.9, 1.0],
    "green": [0.1, 0.8, 0.2, 1.0],
    "yellow": [0.9, 0.8, 0.1, 1.0],
}
FRAGILE_COLOR = [0.85, 0.85, 0.9, 0.6]
HALF = 0.02  # cube half-size in meters


if _MANISKILL_AVAILABLE:

    @register_env("TabletopChoice-v0", max_episode_steps=100)
    class TabletopChoiceEnv(BaseEnv):
        """Colored cubes + a fragile object on a table with a Panda arm."""

        SUPPORTED_ROBOTS = ["panda"]

        def __init__(self, *args, num_cubes: int = 3, robot_uids="panda", **kwargs):
            self.num_cubes = int(num_cubes)
            self._colors = list(CUBE_COLORS.keys())
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        # A single fixed camera looking at the table, used for segmentation.
        @property
        def _default_sensor_configs(self):
            pose = sapien_utils.look_at(eye=[0.35, 0.0, 0.55],
                                        target=[-0.05, 0.0, 0.05])
            return [CameraConfig("base_camera", pose, 128, 128,
                                 np.pi / 2, 0.01, 100)]

        @property
        def _default_human_render_camera_configs(self):
            pose = sapien_utils.look_at(eye=[0.6, 0.4, 0.6],
                                        target=[0.0, 0.0, 0.1])
            return [CameraConfig("render_camera", pose, 512, 512,
                                 np.pi / 3, 0.01, 100)]

        def _load_scene(self, options: dict):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            self.cubes = []
            for i in range(self.num_cubes):
                color = CUBE_COLORS[self._colors[i % len(self._colors)]]
                # An initial_pose is given so ManiSkill doesn't warn about
                # unset builder poses; the real per-episode pose is randomized
                # in _initialize_episode below.
                cube = actors.build_cube(
                    self.scene, half_size=HALF, color=color,
                    name=f"cube_{i}", body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0.0, -0.1 + 0.1 * i, HALF]))
                self.cubes.append(cube)

            self.fragile = actors.build_cube(
                self.scene, half_size=HALF, color=FRAGILE_COLOR,
                name="fragile", body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.2, 0.0, HALF]))

        def _initialize_episode(self, env_idx, options: dict):
            with torch.device(self.device):
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                # Randomize xy on the table; keep a small z so cubes rest on it.
                for obj in self.cubes + [self.fragile]:
                    xy = torch.rand((b, 2)) * 0.24 - 0.12  # ~[-0.12, 0.12] m
                    z = torch.full((b, 1), HALF)
                    obj.set_pose(Pose.create_from_pq(
                        p=torch.cat([xy, z], dim=1)))

        # The decision experiment reads privileged state directly, so the task's
        # own success flag is unused; return a neutral evaluation.
        def evaluate(self):
            return {
                "success": torch.zeros(self.num_envs, dtype=bool,
                                       device=self.device),
            }

        def _get_obs_extra(self, info):
            return dict()


def ensure_available():
    """Raise a clear, actionable error if ManiSkill could not be imported."""
    if not _MANISKILL_AVAILABLE:
        raise ImportError(
            "ManiSkill is not importable in this environment "
            f"({type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}).\n"
            "The ManiSkill backend needs a CUDA GPU; on a laptop use "
            "--backend synthetic.\n"
            "On a GPU box: pip install -r requirements-maniskill.txt "
            "(see MANISKILL_SETUP.md)."
        )
