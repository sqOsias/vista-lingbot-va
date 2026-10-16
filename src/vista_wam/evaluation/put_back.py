"""Shared put_back_block prefix and target-specific scoring."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from vista_wam.control.task_executor import TaskSpatialExecutor
from vista_wam.sim.environment import save_observation

MAT_NAMES = ("left", "right", "front", "back")


def target_score(env, target_name: str) -> dict:
    """Mirror the native placement threshold for one named mat, for scoring only."""
    index = MAT_NAMES.index(target_name)
    block = np.asarray(env.block.get_pose().p, dtype=float)
    target = np.asarray(env.mat_lst[index].get_pose().p, dtype=float)
    distances = {name: float(np.linalg.norm(block[:2]-np.asarray(env.mat_lst[i].get_pose().p)[:2]))
                 for i, name in enumerate(MAT_NAMES)}
    success = bool(abs(block[0]-target[0]) < .03 and abs(block[1]-target[1]) < .03
                   and block[2] < .77 and env.is_right_gripper_open())
    return {"success": success, "target": target_name, "block_xyz": block.tolist(),
            "target_xyz": target.tolist(), "xy_error_m": distances[target_name],
            "nearest_mat": min(distances, key=distances.get),
            "score_only_NOT_policy_input": True}


def run_auxiliary_prefix(env, output: str | Path) -> dict:
    """Build the common center-and-button suffix boundary with an external executor."""
    output = Path(output); prefix = output / "auxiliary_prefix"; prefix.mkdir()
    executor = TaskSpatialExecutor(env, prefix, env.step_lim)
    block_pose = np.asarray(env.block.get_pose().p, dtype=float)
    block = {"center_xy_m": block_pose[:2].tolist(), "table_z_m": .7400001287460327,
             "height_m": .04, "major_axis_xy": [1., 0.], "minor_axis_xy": [0., 1.]}
    executor.apply({"action": "pick_place", "rotation_xyz_degrees": [0,0,0]},
                   block, np.asarray([.1, -.1, .7400001287460327]))
    save_observation(env, output / "prefix_block_center.npz")
    button_pose = np.asarray(env.button.get_pose().p, dtype=float)
    button = {"center_xy_m": button_pose[:2].tolist(), "table_z_m": .7400001287460327,
              "height_m": .039, "major_axis_xy": [1.,0.], "minor_axis_xy": [0.,1.]}
    executor.apply({"action": "press", "rotation_xyz_degrees": [0,0,0]}, button)
    executor.close(); save_observation(env, output / "prefix_button_once.npz")
    valid = int(env.stage_id) == 1 and int(env.press_cnt) == 1 and bool(env.check_block_in_center())
    receipt = {"valid_suffix_boundary": valid, "actual_commands": executor.steps,
               "action_source": "external analytic executor",
               "provided_to_wam_history": False}
    (output / "auxiliary_prefix_receipt.json").write_text(json.dumps(receipt, indent=2))
    if not valid:
        raise RuntimeError("Auxiliary prefix did not reach the suffix boundary")
    return receipt

