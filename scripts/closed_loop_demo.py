"""Closed-loop grasp on an ACT decision.  (GPU only — EXPERIMENTAL.)

Today, outcomes are scored from privileged scene geometry. The honest next step
is to *actually execute* on ACT: motion-plan a grasp of the chosen cube and read
ManiSkill's real success signal, so the safety/success numbers come from a robot
that really tried.

    !!! NOT YET RUN ON A GPU. This uses ManiSkill's motion-planning API
    (PandaArmMotionPlanningSolver + mplib), which varies by version. Treat it as
    a first draft to debug on Colab. `--builtin` falls back to ManiSkill's own,
    known-good PickCube motion-planning demo so you can confirm the capability
    exists on your GPU before debugging the custom grasp.

Usage (on Colab, in the 3.11 env):
  /content/ms/bin/python scripts/closed_loop_demo.py --builtin      # capability check
  /content/ms/bin/python scripts/closed_loop_demo.py --n 5          # our env (experimental)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401
import numpy as np


def run_builtin(n):
    """Run ManiSkill's own tested PickCube motion-planning solution."""
    cmd = [sys.executable, "-m",
           "mani_skill.examples.motionplanning.panda.run",
           "-e", "PickCube-v1", "--num-traj", str(n)]
    print("Capability check — running ManiSkill's built-in grasp planner:\n  "
          + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=False)


def run_custom(n, num_cubes):
    """Attempt a motion-planned grasp of the ACT target in TabletopChoice.

    EXPERIMENTAL — the motion-planning calls below are the likely spots to fix.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from uarsim.backends import maniskill_env  # noqa: F401  registers env
    try:
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver)  # MANISKILL: path may differ by version
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Could not import ManiSkill's motion planner ({e}).\n"
            "Run with --builtin first to confirm motion planning works, then we\n"
            "adapt this import to your ManiSkill version.")

    env = gym.make("TabletopChoice-v0", num_envs=1, num_cubes=num_cubes,
                   obs_mode="state", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    successes = 0
    for i in range(n):
        env.reset(seed=i)
        base = env.unwrapped
        target = base.cubes[0]                         # stand-in for the ACT target
        p = target.pose.p[0].detach().cpu().numpy()
        import sapien
        planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False)
        # MANISKILL: a top-down grasp of the cube, then lift.
        grasp = sapien.Pose(p=[float(p[0]), float(p[1]), float(p[2]) + 0.10],
                            q=[0, 1, 0, 0])
        planner.move_to_pose_with_screw(grasp)
        planner.move_to_pose_with_screw(
            sapien.Pose(p=[float(p[0]), float(p[1]), float(p[2]) + 0.02],
                        q=[0, 1, 0, 0]))
        planner.close_gripper()
        planner.move_to_pose_with_screw(
            sapien.Pose(p=[float(p[0]), float(p[1]), float(p[2]) + 0.18],
                        q=[0, 1, 0, 0]))
        lifted = float(target.pose.p[0][2]) > float(p[2]) + 0.05
        successes += int(lifted)
        planner.close()
        print(f"  episode {i+1}: {'lifted' if lifted else 'missed'}")
    env.close()
    print(f"\nGrasp success: {successes}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builtin", action="store_true",
                    help="run ManiSkill's own PickCube grasp demo (capability check)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--num-cubes", type=int, default=3)
    args = ap.parse_args()
    if args.builtin:
        run_builtin(args.n)
    else:
        run_custom(args.n, args.num_cubes)


if __name__ == "__main__":
    main()
