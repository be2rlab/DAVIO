import numpy as np

from .filtering import usable


def stack(submaps, poses, scales, cfg):
    depth, K, T, window = [], [], [], []
    for m, sm in enumerate(submaps):
        factor = np.float32(scales[m])
        for i in range(len(sm['depth'])):
            ok = usable(sm['depth'][i], cfg, sm['sky'][i] if 'sky' in sm else None, sm['conf'][i])
            depth.append(np.where(ok, sm['depth'][i] * factor, np.nan).astype(np.float32))
            K.append(np.asarray(sm['intrinsics'][i], float))
            T.append(np.asarray(poses[m], float) @ np.asarray(sm['poses'][i], float))
            window.append(m)
    return dict(depth=np.stack(depth), K=np.stack(K), T=np.stack(T), window=np.asarray(window))


def project(i, D, cos_min, step, gate):
    h, w = D['depth'].shape[1:]
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    v, u = v.ravel(), u.ravel()
    z = D['depth'][i][v, u]
    keep = np.isfinite(z)
    u, v, z = u[keep], v[keep], z[keep].astype(np.float64)
    n = len(D['depth'])
    median = np.full(n, np.nan)
    if len(z) < 100:
        return median, np.zeros(n, int)
    cam = np.column_stack((u, v, np.ones(len(z)))) @ np.linalg.inv(D['K'][i]).T * z[:, None]
    X = (cam @ D['T'][i][:3, :3].T + D['T'][i][:3, 3]).astype(np.float32)
    R, c = D['T'][:, :3, :3].astype(np.float32), D['T'][:, :3, 3].astype(np.float32)
    rel = X[None] - c[:, None]
    Xc = np.matmul(rel, R)
    zc = Xc[..., 2]
    uvw = np.matmul(Xc, D['K'].astype(np.float32).transpose(0, 2, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        ub = np.rint(uvw[..., 0] / zc).astype(np.int32)
        vb = np.rint(uvw[..., 1] / zc).astype(np.int32)
        own = X - c[i]
        cos = (np.einsum('jnk,nk->jn', rel, own)
               / (np.linalg.norm(rel, axis=2) * np.linalg.norm(own, axis=1)[None]))
    ok = (zc > .15) & (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h) & (cos >= cos_min)
    ub, vb = np.where(ok, ub, 0), np.where(ok, vb, 0)
    zb = D['depth'][np.arange(n)[:, None], vb, ub]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(zb) - np.log(zc)
    ok &= np.isfinite(r) & (np.abs(r) < gate)       # beyond the gate: an occlusion, not a scale error
    count = ok.sum(1)
    rows = np.flatnonzero(count >= 50)
    if len(rows):
        median[rows] = np.nanmedian(np.where(ok, r, np.nan)[rows], axis=1)
    return median, count


def agreement(D, cfg):
    n = len(D['depth'])
    cos_min = float(np.cos(np.deg2rad(cfg['revisit_max_angle_deg'])))
    median, count = np.full((n, n), np.nan), np.zeros((n, n), np.int32)
    for i in range(n):
        median[i], count[i] = project(i, D, cos_min, int(cfg['revisit_step']), float(cfg['revisit_gate_log']))
    return median, count


def solve(median, count, window, cfg, rounds=10):
    median, window = np.asarray(median, float), np.asarray(window)
    need = int(cfg['revisit_min_shared'])
    a_all = (median - median.T) / 2
    ok = (count >= need) & (count.T >= need) & (window[:, None] != window[None, :]) & np.isfinite(a_all)
    n = len(median)
    i, j = np.nonzero(np.triu(ok, 1))
    degree = np.bincount(np.r_[i, j], minlength=n)
    if not len(i):
        return np.zeros(n), degree
    a = a_all[i, j]
    prior = np.diag(np.broadcast_to(np.asarray(cfg['revisit_prior_pairs'], float), n))
    c, w = np.zeros(n), np.ones(len(a))
    for _ in range(rounds):
        L, b = np.zeros((n, n)), np.zeros(n)
        np.add.at(L, (i, i), w)
        np.add.at(L, (j, j), w)
        np.add.at(L, (i, j), -w)
        np.add.at(L, (j, i), -w)
        np.add.at(b, j, w * a)
        np.add.at(b, i, -w * a)
        c = np.linalg.solve(L + prior, b)
        r = a - (c[j] - c[i])
        sigma = max(1.4826 * np.median(np.abs(r)), 1e-4)        # about zero, not about the median
        w = np.minimum(1., float(cfg['revisit_huber']) * sigma / np.maximum(np.abs(r), 1e-12))
    return c, degree


def corrections(submaps, poses, scales, cfg):
    D = stack(submaps, poses, scales, cfg)
    c, degree = solve(*agreement(D, cfg), D['window'], cfg)
    info = dict(images=int(len(c)), with_pairs=float(np.mean(degree > 0)),
                pairs_per_image_median=float(np.median(degree[degree > 0])) if (degree > 0).any() else 0.,
                correction_abs_median=float(np.median(np.abs(c))),
                correction_abs_max=float(np.abs(c).max()))
    return [c[D['window'] == m] for m in range(len(submaps))], info
