"""Contain a known TOPP parameterization error at the planner boundary.

Input: existing scoped planners. Output: explicit Failure status for the known
parameterization exception, preserving other exceptions and original commands.
Rejected paths are not executed or counted as successful motion. This guard
does not add collision avoidance, invent replacement trajectories or train.
"""


class ParameterizationGuard:
    """Delegate planning and reject only the recorded TOPP runtime failure."""

    def __init__(self, planner):
        """Keep the original planner and an independent rejection log."""
        self.planner = planner; self.rejections = []

    def __getattr__(self, name):
        """Delegate unchanged gripper and planner attributes."""
        return getattr(self.planner, name)

    def plan_path(self, *args, **kwargs):
        """Return original output, or an explicit rejected path on known TOPP error."""
        try:
            return self.planner.plan_path(*args, **kwargs)
        except RuntimeError as exc:
            if 'Fail to parameterize path' not in str(exc): raise
            reason={'error_type':type(exc).__name__,'error':str(exc)}
            self.rejections.append(reason)
            return {'status':'Failure','rejection_reason':reason}


def install_guard(env):
    """Wrap only this environment's left/right planners, returning both guards."""
    result={}
    for arm in ['left','right']:
        guard=ParameterizationGuard(getattr(env.robot,arm+'_planner'))
        setattr(env.robot,arm+'_planner',guard);result[arm]=guard
    return result
