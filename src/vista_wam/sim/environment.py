"""Create an RMBench environment without modifying the upstream checkout."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from vista_wam.config import load_config

CAMERAS = ("head_camera", "left_camera", "right_camera")


def create_env(seed: int, output: str | Path, task: str = "put_back_block", camera: str = "D435"):
    """Create one deterministic RMBench episode and bind MPLib EE planners.

    The factory applies camera and planner settings only inside this Python
    process. It does not patch files in the pinned RMBench repository.
    """
    import mplib
    import yaml

    cfg_all = load_config()
    base = Path(cfg_all["third_party"]["rmbench"])
    if not (base / "envs").is_dir():
        raise FileNotFoundError(f"RMBench is missing: {base}")
    os.chdir(base)
    sys.path.insert(0, str(base))
    from envs.robot.robot import Robot

    task_class = getattr(importlib.import_module(f"envs.{task}"), task)

    class GripperInitialization:
        def plan_grippers(self, now_val, target_val):
            return {"num_step": 200, "per_step": (target_val-now_val)/200,
                    "result": np.linspace(now_val, target_val, 200)}

    class TOPPOnly:
        joint_vel_limits = np.ones(6)
        joint_acc_limits = np.ones(6)
        TOPP = mplib.Planner.TOPP

    def scoped_planner(self, scene=None):
        self.need_topp = False
        self.communication_flag = False
        self.left_planner = GripperInitialization()
        self.right_planner = GripperInitialization()
        self.left_mplib_planner = TOPPOnly()
        self.right_mplib_planner = TOPPOnly()

    Robot.set_planner = scoped_planner
    cfg = yaml.safe_load((base / "task_config/demo_clean.yml").read_text())
    cfg["camera"].update(head_camera_type=camera, wrist_camera_type=camera)
    embodiments = yaml.safe_load((base / "task_config/_embodiment_config.yml").read_text())
    robot_path = (base / embodiments["aloha-agilex"]["file_path"]).resolve()
    robot_cfg = yaml.safe_load((robot_path / "config.yml").read_text())
    cameras = yaml.safe_load((base / "task_config/_camera_config.yml").read_text())
    head = cameras[camera]
    cfg.update(task_name=task, seed=seed, now_ep_num=0, render_freq=0,
               save_data=False, collect_data=False, save_path=str(Path(output).resolve()),
               eval_mode=True, eval_video_log=False, need_plan=False,
               dual_arm_embodied=True, left_robot_file=str(robot_path),
               right_robot_file=str(robot_path), left_embodiment_config=robot_cfg,
               right_embodiment_config=robot_cfg,
               head_camera_h=head["h"], head_camera_w=head["w"])
    env = task_class()
    env.setup_demo(**cfg)

    from envs.robot.planner import MplibPlanner

    class CompatibleMplibPlanner(MplibPlanner):
        def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
            if constraint_pose is not None:
                raise ValueError("Constrained EE planning is outside this harness")
            return super().plan_path(curr_joint_pos, target_gripper_pose, arms_tag=arms_tag)

    for arm in ("left", "right"):
        planner = CompatibleMplibPlanner(
            getattr(env.robot, f"{arm}_urdf_path"), getattr(env.robot, f"{arm}_srdf_path"),
            getattr(env.robot, f"{arm}_move_group"), getattr(env.robot, f"{arm}_entity_origion_pose"),
            getattr(env.robot, f"{arm}_entity"), planner_type="mplib_screw", scene=None)
        setattr(env.robot, f"{arm}_planner", planner)
    env.robot.communication_flag = False
    receipt = {"environment": "RMBench", "task": task, "seed": seed,
               "camera": camera, "repository": str(base),
               "upstream_files_modified": False}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "environment_receipt.json").write_text(json.dumps(receipt, indent=2))
    return env


def save_observation(env, destination: str | Path):
    """Persist the exact three RGB observations and joint vector used by the model."""
    from PIL import Image
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    obs = env.get_obs()
    np.savez(destination, state=np.asarray(obs["joint_action"]["vector"]),
             **{cam: obs["observation"][cam]["rgb"] for cam in CAMERAS})
    for cam in CAMERAS:
        Image.fromarray(obs["observation"][cam]["rgb"]).save(
            destination.with_name(f"{destination.stem}_{cam}.png"))
    return obs

