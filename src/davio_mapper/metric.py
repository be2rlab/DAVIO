import numpy as np
from scipy.spatial.transform import Rotation
from . import sim3

def initialize_metric_submap(local_poses, global_poses, return_info=False):
    local = np.asarray(local_poses)
    world = np.asarray(global_poses)

    rotations = (
        world[:, :3, :3]
        @ local[:, :3, :3].transpose(0, 2, 1)
    )
    r = Rotation.from_matrix(rotations).mean().as_matrix()

    x = local[:, :3, 3]
    y = world[:, :3, 3]

    xc = x - x.mean(0)
    yc = y - y.mean(0)

    local_energy = float(np.sum(xc * xc))
    metric_energy = float(np.sum(yc * yc))

    # Structural property: the local camera configuration contains a
    # non-zero baseline.  Do not use a scale-magnitude threshold here;
    # practical conditioning is a separate property.
    structurally_observable = (
        np.isfinite(local_energy)
        and local_energy > 0.
    )

    signed_scale = (
        float(np.sum((xc @ r.T) * yc) / local_energy)
        if structurally_observable
        else 0.
    )

    positive_fit = (
        structurally_observable
        and np.isfinite(signed_scale)
        and signed_scale > 0.
    )

    if positive_fit:
        seed_scale = signed_scale
        seed_source = "signed_translation_fit"
    elif (
        structurally_observable
        and np.isfinite(metric_energy)
        and metric_energy > 0.
    ):
        # Positive and invariant to a rescaling of DA3 local coordinates.
        seed_scale = float(
            np.sqrt(metric_energy / local_energy)
        )
        seed_source = "rms_baseline_ratio"
    else:
        # Pure technical seed for a truly zero-baseline local configuration.
        # It must not be interpreted as observed metric scale.
        seed_scale = 1.
        seed_source = "unit_fallback"

    if not np.isfinite(seed_scale) or seed_scale <= 0.:
        raise ValueError("Metric submap initialization produced an invalid positive scale")

    out = np.eye(4)
    out[:3, :3] = seed_scale * r
    out[:3, 3] = (
        y.mean(0)
        - seed_scale * r @ x.mean(0)
    )

    info = {
        "structurally_observable": bool(structurally_observable),
        "signed_scale_fit": float(signed_scale),
        "positive_interior_fit": bool(positive_fit),
        "metric_scale_anchored": bool(positive_fit),
        "initialization_incompatible": bool(
            structurally_observable and not positive_fit
        ),
        "seed_scale": float(seed_scale),
        "seed_source": seed_source,
        "local_center_energy": float(local_energy),
        "metric_center_energy": float(metric_energy),
    }

    if return_info:
        return out, structurally_observable, info

    return out, structurally_observable

