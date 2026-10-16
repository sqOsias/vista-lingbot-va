"""Action boundary for frozen LingBot evaluation.

Inputs: RMBench WXYZ end poses, 16-channel LingBot XYZW relative poses,
and quantile statistics. Outputs: validated absolute EE commands and actual
executed relative action history. This module does not predict or train actions.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion(value):
    """Return a unit quaternion; reject non-finite or zero-length input."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-8:
        raise ValueError('Invalid quaternion')
    return value / np.linalg.norm(value)


def observed_eef(observation):
    """Convert required RMBench endpose fields to 16-channel XYZW poses.

    Missing fields raise an error instead of silently creating zero poses.
    """
    end = observation['endpose']
    result = []
    for arm in ['left', 'right']:
        pose = np.asarray(end[arm + '_endpose'], dtype=np.float64)
        result.extend(pose[:3])
        result.extend(quaternion(pose[3:7])[[1, 2, 3, 0]])
        result.append(float(end[arm + '_gripper']))
    value = np.asarray(result)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError('Invalid observed EE state')
    return value


def relative_to_command(relative, origin):
    """Decode 16-channel XYZW relative poses into RMBench WXYZ EE commands.

    Translation is relative to the episode origin, rotations compose
    R_origin @ R_relative. Gripper values are absolute, not pose deltas.
    """
    relative = np.asarray(relative, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if relative.shape != (16,) or origin.shape != (16,) or not np.isfinite(relative).all():
        raise ValueError('Expected finite 16-channel relative action')
    result = []
    for offset in [0, 8]:
        result.extend(origin[offset:offset+3] + relative[offset:offset+3])
        rotation = Rotation.from_quat(quaternion(origin[offset+3:offset+7])) * Rotation.from_quat(quaternion(relative[offset+3:offset+7]))
        result.extend(rotation.as_quat()[[3, 0, 1, 2]])
        result.append(float(relative[offset+7]))
    return np.asarray(result)


def absolute_to_relative(absolute, origin):
    """Encode measured XYZW absolute EE states as episode-relative history.

    Only measured/actually executed states should be supplied, never unused
    predictions. This is the inverse of relative_to_command after WXYZ→XYZW.
    """
    absolute = np.asarray(absolute, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    result = []
    for offset in [0, 8]:
        result.extend(absolute[offset:offset+3] - origin[offset:offset+3])
        rotation = Rotation.from_quat(quaternion(origin[offset+3:offset+7])).inv() * Rotation.from_quat(quaternion(absolute[offset+3:offset+7]))
        result.extend(rotation.as_quat())
        result.append(float(absolute[offset+7]))
    return np.asarray(result)


def command_to_absolute(command):
    """Convert a validated RMBench WXYZ command to LingBot XYZW pose order."""
    command = np.asarray(command, dtype=np.float64)
    result = command.copy()
    for offset in [0, 8]:
        result[offset+3:offset+7] = quaternion(command[offset+3:offset+7])[[1, 2, 3, 0]]
    return result


def quantile_roundtrip(value, lower, upper):
    """Normalize then restore values, using the upstream 1e-6 convention."""
    value, lower, upper = map(np.asarray, [value, lower, upper])
    if np.any(upper < lower):
        raise ValueError('Inverted quantiles')
    scaled = (value - lower) / (upper - lower + 1e-6) * 2 - 1
    return (scaled + 1) / 2 * (upper - lower + 1e-6) + lower
