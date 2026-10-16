"""Task-skill extension over the frozen observe spatial control boundary.

Inputs: grounded object/destination, validated rotation and chosen arm.
Outputs: actual common-EE primitives for cover rotation, battery polarity,
moving blocks/T pieces and physical buttons. No original task helpers or
hidden actor pose is called; both memory conditions use this same extension.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from vista_wam.control.spatial_executor import SpatialExecutor


class TaskSpatialExecutor(SpatialExecutor):
    """Ground and execute planner proposals using the shared measured-EE backend."""

    def move(self,arm,xyz,q,grip,count,stage):
        """Use the validated twofold waypoint reduction from the final executor."""
        return super().move(arm,xyz,q,grip,max(2,math.ceil(count/2)),stage)

    def pick_place(self,proposal,destination,arm=None,rotation_degrees=None):
        """Preserve the calibrated TCP offset when placing with world rotations."""
        arm,q,grasp=self.grasp(proposal,arm,True)
        if rotation_degrees is not None:
            q=(Rotation.from_euler('xyz',rotation_degrees,degrees=True)*Rotation.from_quat(q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
        direction=Rotation.from_quat(q[[1,2,3,0]]).apply([1,0,0]);destination=np.asarray(destination)
        tcp=np.r_[destination[:2],destination[2]+.45*proposal['height_m']+.002];target=tcp-.12*direction
        self.move(arm,target+[0,0,.10],q,0.,20,'transfer');self.move(arm,target,q,0.,16,'place');self.move(arm,target,q,1.,8,'release');self.move(arm,target+[0,0,.09],q,1.,16,'retreat')

    def apply(self,action,proposal,destination=None):
        """Dispatch an admissible primitive, returning the actual attempt count."""
        arm=None;name=action['action'];start=self.steps
        if name=='lift':self.grasp(proposal,arm,True)
        elif name=='pick_place':self.pick_place(proposal,destination,arm,action.get('rotation_xyz_degrees'))
        elif name=='press':self.press(proposal,arm)
        elif name=='push':
            selected=arm or ('left' if proposal['center_xy_m'][0]<0 else 'right');q=self.analytic_pose(proposal);height=proposal['table_z_m']+.45*proposal['height_m']+.12
            current=np.r_[proposal['center_xy_m'],height];target=np.r_[np.asarray(destination)[:2],height]
            self.move(selected,current+[0,0,.05],q,0.,20,'push_approach');self.move(selected,current,q,0.,12,'push_contact');self.move(selected,target,q,0.,24,'push');self.move(selected,target+[0,0,.08],q,1.,16,'push_retreat')
        else:raise ValueError('Primitive must have a physical implementation')
        if name!='lift':
            selected=arm or ('left' if proposal['center_xy_m'][0]<0 else 'right');offset=0 if selected=='left' else 8
            self.move(selected,self.origin[offset:offset+3],self.origin[offset+3:offset+7],1.,16,'park_for_visual_feedback')
        return self.steps-start
