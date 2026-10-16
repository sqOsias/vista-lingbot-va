"""Match the released RoboTwin dataset/client's numerical rotation convention.

Inputs: absolute RMBench/RoboTwin WXYZ poses or model relative 16D actions.
Outputs: absolute WXYZ commands and actual-history tensors matching upstream
get_relative_pose/add_eef_pose. Raw WXYZ components are deliberately interpreted
as scalar-last by scipy on BOTH sides; this reproduces training arithmetic,
not a claim that raw absolute quaternions physically use XYZW. No training.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from vista_wam.control.standard_contract import quaternion


def observed_raw(observation):
    """Return measured left/right XYZ,WXYZ,grip without reordering components."""
    end = observation['endpose']; result = []
    for arm in ['left', 'right']:
        pose = np.asarray(end[arm+'_endpose'], dtype=float)
        result.extend(pose[:3]); result.extend(quaternion(pose[3:7]))
        result.append(float(end[arm+'_gripper']))
    value = np.asarray(result)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError('Expected measured finite 16D raw poses')
    return value


def encode_raw(absolute, origin):
    """Encode measured WXYZ components using the author's dataset arithmetic."""
    absolute, origin = np.asarray(absolute), np.asarray(origin); result = []
    if absolute.shape != (16,) or origin.shape != (16,):
        raise ValueError('Expected two 16D poses')
    for offset in [0, 8]:
        result.extend(absolute[offset:offset+3]-origin[offset:offset+3])
        rotation = Rotation.from_quat(quaternion(origin[offset+3:offset+7])).inv() * Rotation.from_quat(quaternion(absolute[offset+3:offset+7]))
        result.extend(rotation.as_quat()); result.append(float(absolute[offset+7]))
    return np.asarray(result)


def decode_raw(relative, origin):
    """Invert upstream encoding, returning raw WXYZ commands to the environment."""
    relative, origin = np.asarray(relative), np.asarray(origin); result = []
    if relative.shape != (16,) or origin.shape != (16,) or not np.isfinite(relative).all():
        raise ValueError('Expected finite 16D relative actions')
    for offset in [0, 8]:
        result.extend(origin[offset:offset+3]+relative[offset:offset+3])
        rotation = Rotation.from_quat(quaternion(origin[offset+3:offset+7])) * Rotation.from_quat(quaternion(relative[offset+3:offset+7]))
        result.extend(rotation.as_quat()); result.append(float(relative[offset+7]))
    return np.asarray(result)


def native_history(states, origin, first=False):
    """Pack actual raw poses as (16,2,16); first block uses initial padding."""
    required = 16 if first else 32
    if len(states) != required:
        raise ValueError('Incomplete measured history')
    rows = [encode_raw(state, origin) for state in states]
    if first:
        rows = [encode_raw(origin, origin)]*16+rows
    return np.asarray(rows).reshape(2, 16, 16).transpose(2, 0, 1)
