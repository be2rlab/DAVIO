"""Depth of matched keypoints by triangulation between two metric cameras."""
import numpy as np


def triangulate(uv_a, uv_b, K_a, K_b, world_a, world_b, pixel_sigma=1., pose_sigma_m=0.):
    def rays(uv, K, world):
        d = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T @ world[:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)
    uv_a, uv_b = np.asarray(uv_a, float), np.asarray(uv_b, float)
    da, db = rays(uv_a, K_a, world_a), rays(uv_b, K_b, world_b)
    ca, cb = world_a[:3, 3], world_b[:3, 3]
    w0 = ca - cb
    b = np.sum(da * db, axis=1)
    d, e = da @ w0, db @ w0
    denom = 1. - b * b
    parallel = denom < 1e-10
    safe = np.where(parallel, 1., denom)
    s, t = (b * e - d) / safe, (e - b * d) / safe
    point = .5 * ((ca + s[:, None] * da) + (cb + t[:, None] * db))

    def camera(world, K):
        p = (point - world[:3, 3]) @ world[:3, :3]
        z = p[:, 2]
        uv = (p @ np.asarray(K, float).T)[:, :2] / np.where(np.abs(z) > 1e-9, z, 1.)[:, None]
        return z, uv
    za, pa = camera(world_a, K_a)
    zb, pb = camera(world_b, K_b)
    theta = np.arccos(np.clip(b, -1., 1.))
    rho_a, rho_b = np.linalg.norm(point - ca, axis=1), np.linalg.norm(point - cb, axis=1)
    angular = pixel_sigma / (.5 * (K_a[0, 0] + K_b[0, 0]))
    sin = np.maximum(np.sin(theta), 1e-9)
    sigma = np.sqrt((rho_a * angular) ** 2 + (rho_b * angular * np.cos(theta)) ** 2
                    + pose_sigma_m ** 2) / sin
    ok = ~parallel & (za > 1e-3) & (zb > 1e-3) & (s > 0) & (t > 0)
    return dict(depth_b=zb, sigma_log=np.where(ok, sigma / np.maximum(rho_b, 1e-9), np.inf),
                parallax_deg=np.degrees(theta),
                reprojection_px=np.maximum(np.linalg.norm(pa - uv_a, axis=1),
                                           np.linalg.norm(pb - uv_b, axis=1)),
                ok=ok)


def centred_pose_variance(uv_a, uv_b, K_a, K_b, world_a, world_b, pose_sigma_m, eps=1e-4):
    base = np.log(np.maximum(triangulate(uv_a, uv_b, K_a, K_b, world_a, world_b)['depth_b'], 1e-9))
    columns = []
    for axis in range(3):
        moved = np.array(world_b, float)
        moved[axis, 3] += eps
        z = triangulate(uv_a, uv_b, K_a, K_b, world_a, moved)['depth_b']
        columns.append((np.log(np.maximum(z, 1e-9)) - base) / eps)
    J = np.column_stack(columns)
    keep = np.isfinite(J).all(axis=1)
    if not keep.any() or pose_sigma_m <= 0:
        return np.zeros(len(J))
    # The common shift is a median, which is not linear in the error: taken over fixed draws.
    draws = np.random.default_rng(0).normal(size=(3, 256)) * pose_sigma_m
    shift = np.nan_to_num(J) @ draws
    shift -= np.median(shift[keep], axis=0)
    return np.mean(shift ** 2, axis=1)
