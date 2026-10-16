"""Scorer-only physical snapshots; these values never enter the policy input."""
import numpy as np

from vista_wam.control.robotwin_contract import observed_raw


def pose_record(obj):
    pose = obj.get_pose()
    return {"p": [float(x) for x in pose.p], "q_wxyz": [float(x) for x in pose.q]}


def snapshot(env, observation):
    """Record the complete suffix boundary for reproducibility."""
    return {
        "block": pose_record(env.block),
        "mats": {name: pose_record(mat) for name, mat in zip(("left", "right", "front", "back"), env.mat_lst)},
        "button": pose_record(env.button),
        "eef_raw": observed_raw(observation).tolist(),
        "joint_action": np.asarray(observation["joint_action"]["vector"]).tolist(),
        "stage_id": int(env.stage_id), "press_cnt": int(env.press_cnt),
        "eval_success": bool(env.eval_success), "command_count": int(env.take_action_cnt),
        "scene_timestep": float(env.scene.get_timestep()),
    }

