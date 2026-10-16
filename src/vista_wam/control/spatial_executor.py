"""Shared RGB-D/analytic EE executor for memory ablation and task adapters.

Inputs: calibrated metric object proposals, requested primitive and actual
robot state. Outputs: actual 16D EE commands and measured trace. The observe
The grasp uses the validated stage lengths and geometry; no task actor or hidden
target/state is read. Other skills extend the same planner/control boundary.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from vista_wam.control.robotwin_contract import observed_raw


class SpatialExecutor:
    """Execute geometric primitives using sensors/proprioception and native done."""

    def __init__(self,env,output,budget):
        """Bind unchanged environment control, output trace and action budget."""
        self.env=env;self.output=Path(output);self.budget=min(budget,env.step_lim);self.steps=0
        self.trace=(self.output/'actual_actions.jsonl').open('w');self.origin=observed_raw(env.get_obs());self.events=[]

    def available(self):
        """Return whether native terminal/budget allows another real command."""
        return self.steps<self.budget and not self.env.eval_success

    def execute(self,command,stage):
        """Execute one unchanged EE command and record only actual robot evidence."""
        if not self.available():return
        self.env.take_action(command,action_type='ee');self.steps+=1;measured=observed_raw(self.env.get_obs())
        self.trace.write(json.dumps({'step':self.steps,'stage':stage,'commanded_action':np.asarray(command).tolist(),'actual_raw_wxyz':measured.tolist(),'native_done':bool(self.env.eval_success)})+'\n');self.trace.flush()

    def hold(self,count):
        """Wait with measured initial joints; do not insert a wall or alter task state."""
        hold=np.asarray(self.env.get_obs()['joint_action']['vector'])
        for _ in range(count):
            if not self.available():break
            self.env.take_action(hold,action_type='qpos');self.steps+=1

    def move(self,arm,xyz,q,grip,count,stage):
        """Interpolate measured EE pose to a requested bounded pose/grip."""
        offset=0 if arm=='left' else 8;start=observed_raw(self.env.get_obs());q=np.asarray(q)
        rotation=Slerp([0,1],Rotation.from_quat([start[offset+3:offset+7][[1,2,3,0]],q[[1,2,3,0]]]))
        for index in range(count):
            if not self.available():break
            fraction=(index+1)/count;command=start.copy();command[offset:offset+3]=(1-fraction)*start[offset:offset+3]+fraction*np.asarray(xyz)
            command[offset+3:offset+7]=rotation(fraction).as_quat()[[3,0,1,2]];command[offset+7]=(1-fraction)*start[offset+7]+fraction*grip;self.execute(command,stage)

    def analytic_pose(self,proposal):
        """Construct a downward pose with its opening axis along object width."""
        x=np.array([0.,0.,-1.]);y=np.r_[proposal['minor_axis_xy'],0.]
        return Rotation.from_matrix(np.column_stack([x,y,np.cross(x,y)])).as_quat()[[3,0,1,2]]

    def grasp(self,proposal,arm=None,lift=True):
        """Apply the validated analytic approach/descend/close/lift geometry."""
        arm=arm or ('left' if proposal['center_xy_m'][0]<0 else 'right');q=self.analytic_pose(proposal);direction=Rotation.from_quat(q[[1,2,3,0]]).apply([1,0,0])
        tcp=np.r_[proposal['center_xy_m'],proposal['table_z_m']+.45*proposal['height_m']];grasp=tcp-.12*direction
        for stage,destination,grip,count in [('approach',grasp+[0,0,.09],1.,24),('descend',grasp,1.,16),('close',grasp,0.,8)]:self.move(arm,destination,q,grip,count,stage)
        if lift:self.move(arm,grasp+[0,0,.14],q,0.,24,'lift')
        return arm,q,grasp

    def pick_place(self,proposal,destination,arm=None,rotation_degrees=None):
        """Grasp/lift, transfer to sensor-grounded destination, release and retreat."""
        arm,q,grasp=self.grasp(proposal,arm,True)
        if rotation_degrees is not None:
            q=(Rotation.from_euler('xyz',rotation_degrees,degrees=True)*Rotation.from_quat(q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
        destination=np.asarray(destination);target=np.r_[destination[:2],destination[2]+.45*proposal['height_m']+.12+.002]
        self.move(arm,target+[0,0,.10],q,0.,20,'transfer');self.move(arm,target,q,0.,16,'place');self.move(arm,target,q,1.,8,'release');self.move(arm,target+[0,0,.09],q,1.,16,'retreat')

    def press(self,proposal,arm=None):
        """Press a visible button cap using measured top depth and a closed gripper."""
        arm=arm or ('left' if proposal['center_xy_m'][0]<0 else 'right');q=self.analytic_pose(proposal)
        top=np.r_[proposal['center_xy_m'],proposal['table_z_m']+proposal['height_m']+.12]
        self.move(arm,top+[0,0,.035],q,0.,20,'press_approach');self.move(arm,top-[0,0,.025],q,0.,8,'press_down');self.move(arm,top+[0,0,.06],q,0.,8,'press_release')

    def close(self):
        """Close the trace after native result and independent scoring are recorded."""
        self.trace.close()
