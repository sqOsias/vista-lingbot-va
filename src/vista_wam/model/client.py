"""Evaluation-side client for the explicit LingBot condition/history API.

Inputs: real RMBench observations and actually executed EE states. Outputs:
HTTP receipts, aligned history NPZ and native action tensors. Uses no task
truth; privileged oracle goals must be created only by labeled diagnostics.
"""
import json
from pathlib import Path
import urllib.error
import urllib.request
import numpy as np
from vista_wam.control.standard_contract import observed_eef, absolute_to_relative


def request(operation, port=18767, **fields):
    """Send an explicit operation, returning JSON or the service's full error."""
    payload=json.dumps(dict(operation=operation,**fields)).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=payload,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=600) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def commit_measured(states, keyframes, origin, output, first=False):
    """Persist measured history in native (16,2,16) order and commit it.

    First block has 16 executed steps and 4 sampled observations; its first
    frame is the real stationary origin. Later blocks require 32 steps and 8
    observations. Every keyframe corresponds to four completed environment actions.
    """
    required=16 if first else 32
    if len(states)!=required or len(keyframes)!=required//4:
        raise ValueError('Incomplete actual history block')
    values=[absolute_to_relative(state,origin) for state in states]
    if first:values=[absolute_to_relative(origin,origin)]*16+values
    native=np.asarray(values).reshape(2,16,16).transpose(2,0,1)
    output=Path(output).resolve();np.savez(output,relative_eef=native)
    return request('commit',observations=[str(Path(p).resolve()) for p in keyframes],
                   state_npz=str(output),history_source='measured_executed')


def warm_real_context(env, output, initial_path, initial_observation):
    """Prime native history using initial RGB and 16 actual stationary hold actions.

    Returns episode origin and the commit receipt. The hold is shared protocol
    calibration; this is not an official unrestricted native benchmark rollout.
    """
    from vista_wam.sim.environment import save_observation
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    origin=observed_eef(initial_observation)
    request('prime',observation_npz=str(initial_path))
    hold=np.asarray(initial_observation['joint_action']['vector'])
    states=[];keyframes=[]
    for index in range(16):
        env.take_action(hold,action_type='qpos')
        obs=env.get_obs();states.append(observed_eef(obs))
        if (index+1)%4==0:
            p=output/f'hold_{index+1:04d}.npz';save_observation(env,p);keyframes.append(p)
    if env.fail_flag or env.get_obs_cnt<20:raise RuntimeError('Original observe-period rule failed')
    receipt=commit_measured(states,keyframes,origin,output/'actual_hold_history.npz',first=True)
    return origin,receipt
