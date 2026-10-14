from pathlib import Path

import numpy as np

from . import sim3


def world_points(sm, index, node, pixel_step=2):
    depth = sm['depth'][index]
    v, u = np.mgrid[0:depth.shape[0]:pixel_step, 0:depth.shape[1]:pixel_step]
    z = depth[v, u].astype(float)
    valid = np.isfinite(z) & (z > 0)
    if 'valid' in sm:
        valid &= sm['valid'][index][v, u]
    c = sm['conf'][index][v, u]
    valid &= np.isfinite(c) & (c > 0)
    if 'sky' in sm:
        valid &= ~sm['sky'][index][v, u].astype(bool)
    pixels = np.column_stack((u[valid], v[valid], np.ones(valid.sum())))
    xyz = (pixels @ np.linalg.inv(sm['intrinsics'][index]).T) * z[valid, None]
    xyz = xyz * sim3.scale(node)
    transform = sim3.pose(node) @ sm['poses'][index]
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    return xyz, np.clip(sm['rgb'][index][v[valid], u[valid]], 0, 255).astype(np.uint8)


def reduce_voxels(keys, values):
    """Sum the rows of ``values`` that share a voxel."""
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, values = keys[order], values[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))]
    return keys[starts], np.add.reduceat(values, starts, axis=0)


def merge_voxels(left, right):
    return reduce_voxels(np.concatenate((left[0], right[0])), np.concatenate((left[1], right[1])))


def fuse_map(submaps, nodes, voxel_size=.04, pixel_step=4):
    owners = {}
    for m, sm in enumerate(submaps):
        for i, key in enumerate(sm['frame_ids']):
            candidate = (abs(i - (len(sm['frame_ids']) - 1) / 2), m, i)
            if key not in owners or candidate < owners[key]:
                owners[key] = candidate
    levels = []         # merged pairwise, so the whole map is not sorted again for every image
    for _, m, i in owners.values():
        sm = submaps[m]
        xyz, rgb = world_points(sm, i, nodes[sm['id']], pixel_step)
        good = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[good], rgb[good]
        if not len(xyz):
            continue
        chunk = reduce_voxels(np.floor(xyz / voxel_size).astype(np.int64), np.column_stack((xyz, rgb, np.ones(len(xyz)))))
        level = 0
        while level < len(levels) and levels[level] is not None:
            chunk = merge_voxels(levels[level], chunk)
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(chunk)
        else:
            levels[level] = chunk
    merged = None
    for chunk in levels:
        if chunk is not None:
            merged = chunk if merged is None else merge_voxels(merged, chunk)
    if merged is None:
        raise ValueError('Cannot export an empty map')
    sums = merged[1]
    return sums[:, :3] / sums[:, 6:7], np.clip(sums[:, 3:6] / sums[:, 6:7], 0, 255).astype(np.uint8)


def write_ply(path, points, colors):
    data = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        data[name] = colors[:, i]
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n'
              'property float x\nproperty float y\nproperty float z\nproperty uchar red\n'
              'property uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode())
        stream.write(data.tobytes())


def read_ply(path):
    """The xyz of a PLY file, ASCII or binary."""
    with Path(path).open('rb') as stream:
        fields, count, fmt = [], None, None
        while True:
            line = stream.readline().decode('ascii', 'replace').strip()
            if not line:
                raise ValueError('Truncated PLY header')
            parts = line.split()
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element' and parts[1] == 'vertex':
                count = int(parts[2])
            elif parts[0] == 'property' and len(parts) == 3:
                fields.append((parts[2], parts[1]))
            elif parts[0] == 'end_header':
                break
        if count is None or fmt is None:
            raise ValueError('PLY header lacks a vertex element or format')
        names = [n for n, _ in fields]
        if names[:3] != ['x', 'y', 'z']:
            raise ValueError(f'Expected x,y,z first; got {names[:3]}')
        if fmt == 'ascii':
            flat = np.fromstring(stream.read().decode('ascii', 'replace'), sep=' ', dtype=np.float64)
            return flat.reshape(count, len(fields))[:, :3]
        order = '<' if 'little' in fmt else '>'
        sizes = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                 'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
                 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4'}
        dtype = np.dtype([(n, order + sizes[t]) for n, t in fields])
        data = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)
        return np.column_stack([data['x'], data['y'], data['z']]).astype(float)
