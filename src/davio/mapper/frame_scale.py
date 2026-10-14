import numpy as np


def backproject(kit, idx):
    uv = np.asarray(kit['pixels'], float)[idx]
    return (np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(kit['K']).T) * np.asarray(kit['depth'], float)[idx][:, None]


def refine_frame_scales(kits, poses, cfg, matcher):
    n = len(kits)
    rows, rhs, depths, pairs, matches = [], [], [], 0, 0
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + int(cfg['frame_scale_pairs']))):
            if kits[i]['descriptors'] is None or kits[j]['descriptors'] is None:
                continue
            m = np.asarray(matcher.match(kits[i], kits[j]), int).reshape(-1, 2)
            if len(m) < int(cfg['frame_scale_min_matches']):
                continue
            pa, pb = backproject(kits[i], m[:, 0]), backproject(kits[j], m[:, 1])
            ra, ta = np.asarray(poses[i])[:3, :3], np.asarray(poses[i])[:3, 3]
            rb, tb = np.asarray(poses[j])[:3, :3], np.asarray(poses[j])[:3, 3]
            block = np.zeros((len(m), 3, n))
            block[:, :, i] = pa @ ra.T
            block[:, :, j] = -(pb @ rb.T)
            rows.append(block.reshape(-1, n))
            rhs.append(np.repeat((tb - ta)[None], len(m), axis=0).ravel())
            depths.append(np.repeat(.5 * (np.linalg.norm(pa, axis=1) + np.linalg.norm(pb, axis=1)), 3))
            pairs += 1
            matches += len(m)
    info = dict(pairs_used=pairs, matches=matches, rms_before_rel=None, rms_after_rel=None)
    if not rows:
        return np.ones(n), info
    depth = np.maximum(np.concatenate(depths), 1e-3)
    a, b = np.vstack(rows) / depth[:, None], np.concatenate(rhs) / depth
    prior_sigma = float(cfg['frame_scale_prior_sigma'])     # a weak s = 1 keeps an unmatched image as it is
    prior_a, prior_b = np.eye(n) / prior_sigma, np.ones(n) / prior_sigma

    def residual_norms(s):
        return np.linalg.norm((a @ s - b).reshape(-1, 3), axis=1)

    s = np.ones(n)
    r0 = residual_norms(s)
    for _ in range(int(cfg['frame_scale_rounds'])):
        r = residual_norms(s)
        sigma = max(1.4826 * float(np.median(r)), 1e-3)
        w = np.minimum(1., 2. * sigma / np.maximum(r, 1e-12))
        w[r > float(cfg['frame_scale_cut']) * sigma] = 0.
        weights = np.repeat(np.sqrt(w), 3)
        s = np.linalg.lstsq(np.vstack((a * weights[:, None], prior_a)), np.concatenate((b * weights, prior_b)), rcond=None)[0]
        if not np.isfinite(s).all() or np.any(s <= 0):
            return np.ones(n), dict(info, reason='degenerate solve')
        s = s / np.exp(np.mean(np.log(s)))
    info.update(rms_before_rel=float(np.sqrt(np.mean(r0 ** 2))),
                rms_after_rel=float(np.sqrt(np.mean(residual_norms(s) ** 2))))
    return s, info
