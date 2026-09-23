import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def voxel_downsample(points, voxel):
    """Deterministic one-point-per-voxel subsample, keeping original order."""
    if voxel <= 0:
        return points
    keys = np.floor(np.asarray(points, float) / voxel).astype(np.int64)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def camera_track(ds, period_s, until=None):
    gt = ds.groundtruth()
    body_camera = np.eye(4)
    body_camera[:3, :3] = np.asarray(ds.R_CtoI, float)
    body_camera[:3, 3] = np.asarray(ds.p_IC, float)
    times = np.asarray(gt.t, float)
    keep, last = [], -np.inf
    for i, t in enumerate(times):
        if until is not None and t > until + 1e-9:
            break
        if t - last >= period_s - 1e-9:
            keep.append(i)
            last = t
    keep = np.asarray(keep, int)
    if gt.valid is not None:
        keep = keep[np.asarray(gt.valid, bool)[keep]]
    world = np.repeat(np.eye(4)[None], len(keep), axis=0)
    world[:, :3, :3] = gt.R[keep]
    world[:, :3, 3] = gt.p[keep]
    return world @ body_camera, gt.t[keep]


def _visible_against_reference(points, camera, K, shape, near, far, tolerance, dilate):
    height, width = shape
    local = points @ camera[:3, :3].T + camera[:3, 3]
    ahead = (local[:, 2] > near) & (local[:, 2] < far)
    if not ahead.any():
        return np.zeros(len(points), bool)
    index = np.flatnonzero(ahead)
    z = local[index, 2]
    uv = (local[index] @ K.T)[:, :2] / z[:, None]
    u, v = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index, z, pixel = index[inside], z[inside], (v[inside] * width + u[inside])
    if not len(index):
        return np.zeros(len(points), bool)
    # Nearest reference point per pixel, vectorised: sort by (pixel, depth) and broadcast
    # each group's first depth back over the group.
    order = np.lexsort((z, pixel))
    sorted_pixel, sorted_z = pixel[order], z[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_pixel[1:] != sorted_pixel[:-1])]
    buffer = np.full(height * width, np.inf)
    buffer[sorted_pixel[starts]] = sorted_z[starts]
    if dilate > 1:
        buffer = minimum_filter(buffer.reshape(height, width), size=int(dilate),
                                mode='nearest').ravel()
    out = np.zeros(len(points), bool)
    out[index[z <= buffer[pixel] + tolerance]] = True
    return out


def build(ds, reference, reference_sha, K, shape, *, near=.15, far=20.,
          period_s=.1, voxel=.01, tolerance=.03, dilate=5, until=None):
    points = voxel_downsample(np.asarray(reference, float), voxel)
    cameras, times = camera_track(ds, period_s, until)
    centres = cameras[:, :3, 3]
    crop = cKDTree(centres).query(points, distance_upper_bound=far)[0] < far
    visible = np.zeros(len(points), bool)
    subset = np.flatnonzero(crop)
    inside = points[subset]
    for camera in cameras:
        visible[subset] |= _visible_against_reference(
            inside, np.linalg.inv(camera), K, shape, near, far, tolerance, dilate)
    provenance = dict(
        kind='ground-truth camera support, frozen per sequence',
        reference_sha256=reference_sha, reference_points=int(len(points)),
        groundtruth=ds.groundtruth_provenance() if hasattr(ds, 'groundtruth_provenance') else None,
        cameras=int(len(cameras)), support_period_s=float(period_s),
        **({'until': float(until)} if until is not None else {}),
        first_time=float(times[0]) if len(times) else None,
        last_time=float(times[-1]) if len(times) else None,
        voxel_m=float(voxel), near_m=float(near), far_m=float(far),
        occlusion_tolerance_m=float(tolerance), occlusion_dilate_px=int(dilate),
        image_shape=[int(shape[0]), int(shape[1])], intrinsics=np.asarray(K, float).tolist(),
        crop_points=int(crop.sum()), visible_points=int(visible.sum()),
        algorithm='reference-camera frustum + dilated point z-buffer', algorithm_version=2,
        occlusion='point z-buffer over the reference cloud; surrogate with no guaranteed '
                  'direction of error (sampling gaps can leak occluded points, dilation can '
                  'exclude visible ones); a like-for-like comparison, not a bound')
    provenance['support_id'] = _digest({k: v for k, v in provenance.items()
                                        if k not in ('crop_points', 'visible_points')})
    provenance['mask_sha256'] = hashlib.sha256(
        np.packbits(visible).tobytes() + np.packbits(crop).tobytes()).hexdigest()
    return points, crop, visible, provenance


def load_or_build(cache_dir, ds, reference, reference_sha, K, shape, **kwargs):
    """Build once per (sequence, reference, parameters); reuse byte-identically after."""
    probe = dict(reference_sha256=reference_sha, voxel_m=float(kwargs.get('voxel', .01)),
                 near_m=float(kwargs.get('near', .15)), far_m=float(kwargs.get('far', 20.)),
                 support_period_s=float(kwargs.get('period_s', .1)),
                 occlusion_tolerance_m=float(kwargs.get('tolerance', .03)),
                 occlusion_dilate_px=int(kwargs.get('dilate', 5)),
                 **({'until': float(kwargs['until'])} if kwargs.get('until') is not None else {}),
                 # Part of the cache identity: a changed algorithm must not reuse an old mask.
                 algorithm_version=2,
                 image_shape=[int(shape[0]), int(shape[1])],
                 intrinsics=np.asarray(K, float).tolist(),
                 groundtruth=ds.groundtruth_provenance() if hasattr(ds, 'groundtruth_provenance') else None)
    key = _digest(probe)[:16]
    path = Path(cache_dir) / f'{getattr(ds, "seq", "sequence")}_{key}.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return (data['points'], data['crop'], data['visible'],
                    json.loads(str(data['provenance'])))
    points, crop, visible, provenance = build(
        ds, reference, reference_sha, K, shape, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, points=points, crop=crop, visible=visible,
                        provenance=json.dumps(provenance))
    temporary.replace(path)
    return points, crop, visible, provenance
