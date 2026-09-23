from collections import deque
from types import SimpleNamespace
import numpy as np
from ..init import feedforward
from ..init.preintegration import ImuSample
from ..init.joint_calibration import solve_joint_calibration, joint_calibration_ok
from ..init.validation import rotational_validation, covariance_recoverable
from ..init.types import InitConfig

MODES = ('supplied', 'extrinsics_free', 'free')


class CalibrationAssistant:
    def __init__(self, settings, calibration=None):
        self.settings = settings
        self.windows = deque(maxlen=int(settings['max_windows']))
        self.config = InitConfig()
        self.last_end = -np.inf
        if settings.get('solver', 'joint') not in ('joint', 'alternating'):
            raise ValueError('assistance.solver must be joint or alternating')
        # joint_calibration reads these through getattr, so an unrecognised name would
        # be a silently inert setting rather than an error. Refuse it here instead.
        for name, value in (settings.get('gates') or {}).items():
            if not name.startswith('joint_'):
                raise ValueError(f'unknown assistance gate {name!r}')
            setattr(self.config, name, float(value))
        # calibration: mode, R_CtoI / p_CinI (supplied mode), K / D / resolution, gravity_mag.
        # None keeps the legacy rotation/bias-only behaviour (no feed-forward state).
        self.calibration = dict(calibration) if calibration else None
        mode = self.mode
        if mode == 'free' and settings.get('pair_angle_tol_rad_free') is not None:
            # DA3 rotations from raw, distorted frames disagree with the gyro by a few degrees
            # per pair; the rectified-frame tolerance admits almost none of them.
            self.config.angle_tol_rad = float(settings['pair_angle_tol_rad_free'])
        if mode not in MODES:
            raise ValueError('calibration.mode must be supplied, extrinsics_free or free')
        if self.calibration is not None and mode == 'supplied' and (
                self.calibration.get('R_CtoI') is None or self.calibration.get('p_CinI') is None):
            raise ValueError('supplied mode needs the rig extrinsics')

    @property
    def mode(self):
        return (self.calibration or {}).get('mode', 'supplied')

    @property
    def feedforward_enabled(self):
        return bool(self.settings.get('feedforward', True)) and self.calibration is not None

    @staticmethod
    def shadow_schedule_ok(settings):
        if not settings.get('shadow_continuous', False):
            return True, ''
        period = float(settings.get('shadow_period_s', 5.))
        history = float(settings.get('imu_history_s', 10.))
        span = (int(settings.get('window_frames', 5)) - 1) * float(settings.get('sample_period_s', .1))
        needed = 2. * period + span
        if needed >= history:
            return False, (f'shadow_period_s={period} needs {needed:.1f}s of block history '
                           f'for three blocks but imu_history_s={history}; every epoch '
                           f'would defer. Lower the period or raise the history.')
        return True, ''

    # --- stage 1: rotation and gyro bias ---------------------------------------------------

    def solve_rotation(self, task, prediction, times, imu):
        rotations = np.asarray(prediction.extrinsics, float)[:, :3, :3]
        if rotations.shape != (len(times), 3, 3) or not np.isfinite(rotations).all():
            raise ValueError('Invalid camera rotation predictions')
        # DA3's rotations are world-to-camera with its own reference view as the world. The
        # hand-eye pairs are relative and do not care; bias_gyro.gyro_bias_rows compares
        # frame i's rotation with the IMU's R_{Ii<-I0} and does, so store R_{Ci<-C0}.
        rotations = rotations @ rotations[0].T
        self.last_end = float(times[-1])
        self.windows.append(SimpleNamespace(t0=times[0], frame_times=times,
                                            camera_rotations=rotations))
        while self.windows and self.windows[0].frame_times[0] < imu[0].t:
            self.windows.popleft()
        if not self.windows:
            return None, None, dict(status='deferred', reason='missing IMU boundary support')

        validate = bool(self.settings.get('validate', True))
        if validate:
            held_out = self.windows[-1]
            # Strictly earlier than the whole validation block; fitting windows may
            # still overlap one another.
            fit = [w for w in self.windows if w.frame_times[-1] < held_out.frame_times[0]]
            if len(fit) < 2:
                return None, None, dict(status='deferred',
                                        reason='need two earlier disjoint-from-validation blocks')
        else:
            held_out, fit = None, list(self.windows)
            if len(fit) < 2:
                return None, None, dict(status='deferred', reason='need two blocks')

        alternating = self.settings.get('solver', 'joint') == 'alternating'
        result = solve_joint_calibration(imu, fit, self.config, alternating_only=alternating,
                                         max_iterations=int(self.settings.get('joint_max_iterations', 15)))
        good, reason = joint_calibration_ok(result, self.config)
        if not good:
            return None, None, dict(status='rejected', reason=reason)
        good, reason = covariance_recoverable(result)
        if not good:
            return None, None, dict(status='rejected', reason=reason)

        rms = worst = None
        if validate:
            rms, worst, n = rotational_validation(imu, held_out, result.r_ctoi,
                                                  result.bias_gyro, self.config)
            if not n or not np.isfinite(rms):
                return None, None, dict(status='deferred', reason='no validation rotations')
            suffix = '_free' if self.mode == 'free' else ''
            rms_limit = float(self.settings.get('validation_rms_deg' + suffix, self.settings['validation_rms_deg']))
            max_limit = float(self.settings.get('validation_max_deg' + suffix, self.settings['validation_max_deg']))
            if rms > rms_limit or worst > max_limit:
                return None, None, dict(status='rejected', reason='later-window rotation disagreement',
                                        rms_deg=rms, max_deg=worst)
        if np.linalg.norm(result.bias_gyro) > self.settings['max_bias_norm_rad_s']:
            return None, None, dict(status='rejected', reason='gyro bias norm limit')
        report = dict(status='released', reason='', R_CtoI=result.r_ctoi.tolist(),
                      bg=result.bias_gyro.tolist(), solver=self.settings.get('solver', 'joint'),
                      validated=validate, rms_deg=rms, max_deg=worst,
                      sigma_min_x_given_b=float(result.sigma_min_x_given_b),
                      sigma_min_bb=float(result.sigma_min_bb),
                      n_pairs=int(result.n_pairs), n_unique_frames=int(result.n_unique_frames),
                      axis_coverage=float(result.axis_coverage),
                      fit_span=[float(fit[0].frame_times[0]), float(fit[-1].frame_times[-1])],
                      validation_span=None if held_out is None else
                      [float(held_out.frame_times[0]), float(held_out.frame_times[-1])],
                      available_sensor_time=float(task['available_sensor_time']),
                      covariance_policy='initial guess only; native covariance recovery')
        return result.r_ctoi, result.bias_gyro, report

    # --- stage 2: feed-forward state -------------------------------------------------------

    def propose(self, task, prediction):
        times = np.asarray(task['times'], float)
        if len(times) < 4 or np.any(np.diff(times) <= 0) or times[-1] <= self.last_end:
            return dict(status='deferred', reason='incomplete or non-increasing block')
        imu = [ImuSample(float(t), np.asarray(w), np.asarray(a)) for t, w, a in task['imu']]
        if not imu:
            return dict(status='deferred', reason='missing IMU')
        mode = self.mode
        report, bg0 = None, None
        legacy = bool(self.settings.get('legacy_rotation_proposal', False))
        if mode == 'supplied' and self.calibration is not None and not legacy:
            r_ctoi = np.asarray(self.calibration['R_CtoI'], float)
            p_cini = np.asarray(self.calibration['p_CinI'], float)
        else:
            r_ctoi, bg0, report = self.solve_rotation(task, prediction, times, imu)
            if r_ctoi is None:
                return report
            p_cini = (np.asarray(self.calibration['p_CinI'], float)
                      if self.calibration is not None and self.calibration.get('p_CinI') is not None
                      else None)
        if not self.feedforward_enabled:
            return report                       # legacy rotation/bias-only candidate
        pred = dict(depth=np.asarray(prediction.depth), conf=getattr(prediction, 'conf', None),
                    intrinsics=np.asarray(prediction.intrinsics, float),
                    extrinsics=np.asarray(prediction.extrinsics, float))
        cfg = dict(feedforward.default_config(), **(self.settings.get('ff') or {}))
        cfg['lever_arm_prior_m'] = float(self.settings.get('lever_arm_prior_m', cfg['lever_arm_prior_m']))
        # Deterministic per window: replaying the same sensor data samples the same points.
        rng = np.random.default_rng(int(round(times[-1] * 1e3)) % (2 ** 31))
        res = feedforward.feedforward_initialize(pred, times, imu, r_ctoi, p_cini, cfg,
                                                 float(self.calibration.get('gravity_mag', 9.81)), rng, bg0=bg0)
        if res.status != 'released':
            return dict(status='deferred', reason='feed-forward: ' + res.reason, ff=res.info,
                        rotation=report)
        ext = pred['extrinsics']
        if ext.shape[1:] == (3, 4):                    # DA3 returns world-to-camera 3x4
            full = np.repeat(np.eye(4)[None], len(ext), axis=0)
            full[:, :3] = ext
            ext = full
        centers = np.linalg.inv(ext)[:, :3, 3]
        out = dict(status='released', reason='', R_CtoI=r_ctoi.tolist(), bg=res.bias_gyro.tolist(),
                   p_CinI=np.asarray(res.lever_arm, float).tolist(), bootstrap=res.state,
                   rotation=report, available_sensor_time=float(task['available_sensor_time']),
                   ff=dict(res.info, scale=res.scale, gravity_I0=res.gravity.tolist(),
                           velocity_I0=res.velocity.tolist(), bias_accel=res.bias_accel.tolist(),
                           sigmas=res.sigmas.tolist(), timings=res.timings,
                           window_times=times.tolist(), da3_centers=centers.tolist(),
                           lever_arm_estimated=bool(p_cini is None and cfg.get('estimate_lever_arm', False))),
                   covariance_policy='feed-forward covariance floors; filter refines')
        if mode == 'free':
            # DA3's intrinsics are on its processed grid; the filter tracks the raw image.
            k = np.median(pred['intrinsics'], axis=0)
            grid_w = int(np.asarray(prediction.processed_images).shape[2])
            factor = float(self.calibration['resolution'][0]) / grid_w
            out['K_raw'] = [float(k[0, 0] * factor), float(k[1, 1] * factor),
                            float(k[0, 2] * factor), float(k[1, 2] * factor)]
            out['D'] = [0., 0., 0., 0.]
        return out
