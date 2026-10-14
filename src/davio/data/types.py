from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    D: np.ndarray
    model: str
    resolution: tuple


@dataclass
class GroundTruth:
    t: np.ndarray
    p: np.ndarray
    R: np.ndarray
    v: np.ndarray
    bg: object = None
    ba: object = None
    valid: object = None
    v_valid: object = None
    # False for a reference whose ORIENTATION is
    # known-inaccurate (e.g. UZH-FPV's point-based Leica tracking) -- position (and
    # so ATE, which never reads R) stays trustworthy either way.
    # eval.run.gravity_error_deg raises rather than silently score against it.
    orientation_reliable: bool = True
