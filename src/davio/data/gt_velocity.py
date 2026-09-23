import numpy as np


def differentiate_positions(t, p, gap_s=0.1, sg_window_s=0.25, order=3):
    """-> (v, v_valid). `v_valid` is False wherever no in-segment window was available."""
    from scipy.signal import savgol_filter

    t = np.asarray(t, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(len(t), 3)
    v = np.zeros_like(p)
    v_valid = np.zeros(len(t), bool)
    if len(t) < 2:
        return v, v_valid

    dt = np.diff(t)
    period = float(np.median(dt))
    win = max(int(round(sg_window_s / period)) | 1, 5)

    bounds = np.concatenate([[0], np.flatnonzero(dt > gap_s) + 1, [len(t)]])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < win:
            continue
        v[lo:hi] = savgol_filter(p[lo:hi], win, order, deriv=1, delta=period, axis=0)
        half = win // 2
        v_valid[lo + half:hi - half] = True
    return v, v_valid
