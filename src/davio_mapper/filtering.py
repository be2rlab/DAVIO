"""Conservative multi-view support masks on the original predicted depth grid."""
import cv2
import numpy as np


def filter_depth(sm, cfg):
    depth = sm['depth']
    valid = np.isfinite(depth) & (depth > cfg.get('depth_min_m', .15)) & (depth < cfg.get('depth_max_m', 20.))
    if 'sky' in sm:
        valid &= ~sm['sky'].astype(bool)
    if 'conf' in sm:
        conf = sm['conf']
        for i in range(len(depth)):
            finite = conf[i][valid[i] & np.isfinite(conf[i])]
            threshold = np.quantile(finite, cfg.get('depth_conf_quantile', .2)) if len(finite) else np.inf
            valid[i] &= np.isfinite(conf[i]) & (conf[i] >= threshold)
    # Mixed pixels at depth discontinuities tend to produce floating surfaces.
    for i, z in enumerate(depth):
        safe = np.where(np.isfinite(z), z, 0).astype(np.float32)
        spread = cv2.dilate(safe, np.ones((3,3), np.uint8)) - cv2.erode(safe, np.ones((3,3), np.uint8))
        valid[i] &= spread < cfg.get('depth_edge_relative', .15) * np.maximum(safe, .01)
    support = np.zeros(depth.shape, np.uint8)
    h, w = depth.shape[1:]
    v, u = np.mgrid[:h, :w]
    pixels = np.c_[u.ravel(), v.ravel(), np.ones(h*w)]
    tolerance = cfg.get('depth_consistency_relative', .1)
    for i in range(len(depth)):
        selected = np.flatnonzero(valid[i].ravel())
        xyz = (pixels[selected] @ np.linalg.inv(sm['intrinsics'][i]).T) * depth[i].ravel()[selected,None]
        votes = np.zeros(len(selected), np.uint8)
        for j in range(len(depth)):
            if i == j: continue
            transform = np.linalg.inv(sm['poses'][j]) @ sm['poses'][i]
            points = xyz @ transform[:3,:3].T + transform[:3,3]
            positive = points[:,2] > .01
            uv = np.rint((points @ sm['intrinsics'][j].T)[:,:2] / np.maximum(points[:,2:3], .01)).astype(int)
            inside = positive & (uv[:,0]>=0) & (uv[:,0]<w) & (uv[:,1]>=0) & (uv[:,1]<h)
            ids = np.flatnonzero(inside); x,y = uv[inside].T
            z = depth[j,y,x]
            consistent = valid[j,y,x] & (np.abs(z-points[inside,2]) <= tolerance * np.maximum(z,.01))
            votes[ids[consistent]] += 1
        support[i].ravel()[selected] = votes
    sm['valid'] = valid & (support >= int(cfg.get('depth_min_support', 1)))
    return float(sm['valid'].mean())
