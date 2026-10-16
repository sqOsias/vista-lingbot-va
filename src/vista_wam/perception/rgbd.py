"""Turn a VLM instance box and calibrated RGB-D into a metric target proposal.

Inputs: real rendered RGB, camera-space Position/depth buffer, camera transform
and VLM box. Outputs: visible target center, planar axes, table height and
uncertainty. No actor identifiers, object poses or task labels are accepted.
Simulation depth is ideal; this does not establish real-camera robustness.
"""
import numpy as np


def metric_target(rgb,position,camera_to_world,box):
    """Estimate elevated foreground inside a normalized visual box, or reject it."""
    rgb=np.asarray(rgb)/255.;position=np.asarray(position);matrix=np.asarray(camera_to_world);h,w=rgb.shape[:2]
    world=position[...,:3]@matrix[:3,:3].T+matrix[:3,3];valid=(position[...,3]<1)&np.isfinite(world).all(-1)
    lower=np.indices((h,w))[0]>h*.5
    white=(rgb.min(-1)>.65)&((rgb.max(-1)-rgb.min(-1))<.16)
    surface=world[...,2][valid&lower&white&(world[...,2]>.5)&(world[...,2]<.9)]
    if len(surface)<100:raise ValueError('Insufficient visible table depth')
    counts,edges=np.histogram(surface,bins=np.arange(.5,.902,.002));peak=edges[np.argmax(counts)]+.001
    nearby=surface[abs(surface-peak)<.003];table=float(np.median(nearby))
    x1,y1,x2,y2=np.asarray(box)*[w,h,w,h]/1000
    xx,yy=np.meshgrid(np.arange(w),np.arange(h));inside=(xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)
    mask=valid&inside&(world[...,2]>table+.003)&(world[...,2]<table+.10)
    cloud=world[mask]
    if len(cloud)<20:raise ValueError('Insufficient elevated instance depth')
    # Robust visible extents reduce bias from background pixels and occluded sides.
    low,high=np.quantile(cloud[:,:2],[.05,.95],axis=0);center=(low+high)/2
    covariance=np.cov(cloud[:,:2].T);eigenvalues,axes=np.linalg.eigh(covariance);major=axes[:,-1]
    if major[0]<0:major=-major
    height=float(np.quantile(cloud[:,2],.95)-table)
    return {'source':'VLM_box_and_calibrated_RGBD_only','center_xy_m':center.tolist(),'visible_center_z_m':float(np.median(cloud[:,2])),
            'table_z_m':table,'height_m':height,'major_axis_xy':major.tolist(),'minor_axis_xy':[-float(major[1]),float(major[0])],
            'cloud_points':len(cloud),'extent_xy_m':(high-low).tolist(),'axis_eigenvalue_ratio':float(eigenvalues[-1]/max(eigenvalues[0],1e-12)),
            'uncertainty':'Visible surfaces and VLM box can bias center; no object identity ground truth or full mesh used.'},mask,cloud
