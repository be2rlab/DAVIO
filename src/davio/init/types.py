from dataclasses import dataclass, field
import numpy as np

@dataclass
class InitConfig:
    gravity_mag: float = 9.81
    min_pair_angle_rad: float = 0.05
    angle_tol_rad: float = 0.05
    handeye_ratio: float = 0.1
    handeye_certificate: str = "ratio"
    handeye_ratio_n: float = 0.0132
    handeye_min_pairs: int = 0
    handeye_robust_weighting: bool = False   # see handeye.py's docstring
    require_map_for_seed: bool = False
    lever_arm_norm: float = None
    depth_prior_sigma_m: float = None
    pose_prior_sigma_m: float = None
    velocity_prior_sigma_ms: float = None
    # When set, a window whose OWN pose/velocity-row residual exceeds this many metres
    # (or m/s) fails seed_ok: svg_ok already checks that s/v/g are jointly OBSERVABLE,
    # not that the recovered numbers are self-consistent with the backbone's own
    # displacement/velocity evidence.
    pose_rms_max_m: float = None
    velocity_rms_max_ms: float = None
    # When set (degrees), causal_seed_scan withholds an otherwise-releasable candidate
    # until the very next window's own (strictly larger) prefix rotation agrees with it
    # within this tolerance -- a candidate whose rotation moves once one more window is
    # folded in was not actually stable, whatever its own certificate said.
    rotation_confirmation_tol_deg: float = None
    # When set, the filter seed is forward-propagated (init/seed.py:to_init_seed_at) from
    # the window's t0 to its own last consumed frame instead of being injected as a stale
    # t0 mean -- which also makes the filter's own post-boot feature-database purge (it
    # discards every measurement at or before its boot epoch) discard exactly the frames
    # DAVIO's solve already used, rather than leaving the filter to re-use them.
    propagate_seed_to_last_frame: bool = False
    # When set (degrees), overrides the filter's calib_IMUtoCAM rotation prior --
    # otherwise left at the config's init_prior_qc, calibration-grade (EuRoC config:
    # 0.001 rad ~ 0.06 deg) and silently reused even for a seed whose extrinsic is a
    # calibration-free estimate. The measured causal median extrinsic-rotation error
    # across all 11 EuRoC sequences is 4.03 degrees.
    calib_rot_prior_deg: float = None
    # Threads through to causal_seed_scan(..., incremental=...) in eval/run.py, whose
    # `incremental` docstring has the trade-off.
    use_incremental_handeye: bool = False
    # When True, causal_seed_scan routes each prefix through init/release_policy.evaluate
    # -- the joint rotation+bias solve with a Schur certificate
    # (init/joint_calibration.py) scored on a DISJOINT held-out block -- INSTEAD OF the
    # hand-eye certificate + rotation_confirmation_tol_deg pair, never in addition to
    # them: two accept/reject decisions in one run would make the ablation
    # uninterpretable. See multiwindow.joint_release_decision. Refused, not silently
    # combined, with a supplied extrinsic (extrinsic_mode != "calibrate" + R_CtoI_prior)
    # -- see causal_seed_scan's docstring.
    use_joint_calibration: bool = False
    # None (the default) resolves to False at the call site, which is NOT "skip the
    # metric half" -- read joint_release_decision's docstring, the reasoning is load
    # bearing and this field is the only way to override it.
    joint_calibration_require_metric_validation: bool = None
    gramian_threshold: float = 1e-6
    svg_threshold: float = 1e-3
    scale_conditioning_threshold: float = 0.0
    conf_percentile: float = 25.0
    # Matched diagnostic: hold the gyro bias at its prior instead of estimating it, under
    # the same data budget. Distinct from bg_alternations=0, which still runs one step.
    estimate_gyro_bias: bool = True
    ransac_iters: int = 50
    ransac_min_points: int = 10
    ransac_threshold: float = 1e-2
    use_ransac: bool = True
    scale_min: float = 1e-3
    bg_alternations: int = 2
    # Shared by initializer.initialize()'s own alternation loop and
    # multiwindow.global_handeye_auto_bias's default, so the single-window and prefix
    # bias policies stop each hardcoding their own tolerance.
    bg_convergence_tol_rad_s: float = 1e-6
    p_IC_prior: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_CtoI_prior: np.ndarray = None
    extrinsic_mode: str = "calibrate"
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_budget_rad: float = None
    min_window_duration: float = 0.4
    seed: int = 0
