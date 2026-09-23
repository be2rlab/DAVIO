import numpy as np


def _backproject(kit, idx):
    uv = np.asarray(kit['pixels'], float)[idx]
    return (np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(kit['K']).T) * np.asarray(kit['depth'], float)[idx][:, None]


def refine_frame_scales(kits, poses, cfg, matcher, anchors=None):
    n = len(kits)
    span = int(cfg.get('frame_scale_pairs', 2))
    min_matches = int(cfg.get('frame_scale_min_matches', 30))
    prior_sigma = float(cfg.get('frame_scale_prior_sigma', 1.))
    cut = float(cfg.get('frame_scale_cut', 3.))
    rows, rhs, depths, pairs, matches = [], [], [], 0, 0
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + span)):
            if kits[i]['descriptors'] is None or kits[j]['descriptors'] is None:
                continue
            m = np.asarray(matcher.match(kits[i], kits[j]), int).reshape(-1, 2)
            if len(m) < min_matches:
                continue
            pa, pb = _backproject(kits[i], m[:, 0]), _backproject(kits[j], m[:, 1])
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
    absolute = bool(cfg.get('frame_scale_absolute', False))
    anchor_matches = 0
    if absolute and anchors:
        for i in range(n):
            if kits[i]['descriptors'] is None:
                continue
            for kit_a, pose_a in anchors:
                if kit_a['descriptors'] is None:
                    continue
                m = np.asarray(matcher.match(kits[i], kit_a), int).reshape(-1, 2)
                if len(m) < min_matches:
                    continue
                pi, pa = _backproject(kits[i], m[:, 0]), _backproject(kit_a, m[:, 1])
                ri, ti = np.asarray(poses[i])[:3, :3], np.asarray(poses[i])[:3, 3]
                ra, ta = np.asarray(pose_a)[:3, :3], np.asarray(pose_a)[:3, 3]
                block = np.zeros((len(m), 3, n))
                block[:, :, i] = pi @ ri.T                     # s_i R_i p_i = R_a p_a + t_a - t_i
                rows.append(block.reshape(-1, n))
                rhs.append((pa @ ra.T + ta - ti).ravel())
                depths.append(np.repeat(.5 * (np.linalg.norm(pi, axis=1) + np.linalg.norm(pa, axis=1)), 3))
                anchor_matches += len(m)
    info = dict(pairs_used=pairs, matches=matches, rms_before_rel=None, rms_after_rel=None,
                absolute=absolute, anchor_matches=anchor_matches)
    if not rows:
        return np.ones(n), info
    depth = np.maximum(np.concatenate(depths), 1e-3)
    a, b = np.vstack(rows) / depth[:, None], np.concatenate(rhs) / depth      # relative residuals
    prior_a, prior_b = np.eye(n) / prior_sigma, np.ones(n) / prior_sigma
    if absolute:
        # Anchor matches pin the absolute scale; the prior holds frames without them at one.
        # Residuals stay depth-normalized and the 3-sigma cut stays (no collapse toward zero).
        abs_sigma = float(cfg.get('frame_scale_abs_prior_sigma', .1))
        prior_a, prior_b = np.eye(n) / abs_sigma, np.ones(n) / abs_sigma

    def solve(weights):
        aw = np.vstack((a * weights[:, None], prior_a))
        bw = np.concatenate((b * weights, prior_b))
        return np.linalg.lstsq(aw, bw, rcond=None)[0]

    def residual_norms(s):
        return np.linalg.norm((a @ s - b).reshape(-1, 3), axis=1)

    def normalized(s):
        if not np.isfinite(s).all() or np.any(s <= 0):
            return None
        return s if absolute else s / np.exp(np.mean(np.log(s)))

    s = np.ones(n)
    r0 = residual_norms(s)
    for _ in range(int(cfg.get('frame_scale_rounds', 8))):
        r = residual_norms(s)
        sigma = max(1.4826 * float(np.median(r)), 1e-3)                  # robust sigma, 0.1 % floor
        w = np.minimum(1., 2. * sigma / np.maximum(r, 1e-12))              # Huber knee at 2 sigma
        w[r > cut * sigma] = 0.
        s_new = normalized(solve(np.repeat(np.sqrt(w), 3)))
        if s_new is None:
            return np.ones(n), dict(info, reason='degenerate solve')
        s = s_new
    info.update(rms_before_rel=float(np.sqrt(np.mean(r0 ** 2))),
                rms_after_rel=float(np.sqrt(np.mean(residual_norms(s) ** 2))))
    return s, info
