import csv
from dataclasses import dataclass, field

import numpy as np

from .jpl import exp_so3, skew


def _right_jacobian_so3(phi):
    """Right Jacobian J_r of SO(3) (Forster et al.), needed for the bias Jacobian."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    k = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * k
    return (np.eye(3)
            - ((1.0 - np.cos(theta)) / theta ** 2) * k
            + ((theta - np.sin(theta)) / theta ** 3) * (k @ k))


@dataclass
class ImuSample:
    t: float
    gyro: np.ndarray
    accel: np.ndarray


class ImuStream(list):
    __slots__ = ("times",)

    def __init__(self, samples=()):
        super().__init__(samples)
        self.times = np.fromiter((s.t for s in self), dtype=float, count=len(self))


def _window(samples, t_lo, t_hi):
    times = getattr(samples, "times", None)
    if times is None:
        return [s for s in samples if t_lo <= s.t <= t_hi]
    lo = int(np.searchsorted(times, t_lo, side="left"))
    hi = int(np.searchsorted(times, t_hi, side="right"))
    return list(samples[lo:hi])


def _times_of(samples):
    times = getattr(samples, "times", None)
    if times is not None:
        return times
    return np.fromiter((s.t for s in samples), dtype=float, count=len(samples))


def _interp_at(samples, times, t):
    k = int(np.searchsorted(times, t))
    if k < len(times) and times[k] == t:
        return samples[k]
    lo, hi = samples[k - 1], samples[k]
    span = hi.t - lo.t
    w = 0.0 if span <= 0 else (t - lo.t) / span
    return ImuSample(t=t, gyro=lo.gyro + w * (hi.gyro - lo.gyro),
                     accel=lo.accel + w * (hi.accel - lo.accel))


def _bracketed_window(samples, times, t_start, t_end):
    if len(times) < 2:
        raise ValueError("need >=2 IMU samples, got %d" % len(times))
    tol = 1e-9
    if t_start < times[0] - tol or t_end > times[-1] + tol:
        raise ValueError(
            "requested interval [%.6f, %.6f] exceeds IMU support [%.6f, %.6f]"
            % (t_start, t_end, times[0], times[-1]))
    t_start = max(t_start, float(times[0]))
    t_end = min(t_end, float(times[-1]))
    lo = int(np.searchsorted(times, t_start, side="right"))
    hi = int(np.searchsorted(times, t_end, side="left"))
    start = _interp_at(samples, times, t_start)
    end = _interp_at(samples, times, t_end)
    return [start] + list(samples[lo:hi]) + [end]


def _integrate_step(state, prev, curr, bias_gyro, bias_accel):
    dR, dv, dp, dR_dbg, dv_dba, dp_dba = state
    dt = curr.t - prev.t
    if dt <= 0:
        return state
    omega = 0.5 * (prev.gyro + curr.gyro) - bias_gyro
    accel = 0.5 * (prev.accel + curr.accel) - bias_accel

    # Position/velocity use the rotation at the start of the step, expressed in I0.
    a_in_i0 = dR.T @ accel
    dp = dp + dv * dt + 0.5 * a_in_i0 * dt * dt
    dv = dv + a_in_i0 * dt

    dp_dba = dp_dba + dv_dba * dt - 0.5 * dt * dt * dR.T
    dv_dba = dv_dba - dt * dR.T

    d_theta = omega * dt
    dR_step = exp_so3(d_theta)          # ACTIVE increment
    dR_dbg = dR_step.T @ dR_dbg + _right_jacobian_so3(d_theta) * dt
    dR = dR_step.T @ dR
    return dR, dv, dp, dR_dbg, dv_dba, dp_dba


@dataclass
class Preintegrated:
    dt: float
    dR: np.ndarray
    dv: np.ndarray
    dp: np.ndarray
    dR_dbg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dv_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    dp_dba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


def load_asl_imu(csv_path):
    """Read a EuRoC ASL mav0/imu0/data.csv into ImuSample list (seconds, rad/s, m/s^2)."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            samples.append(ImuSample(
                t=int(row[0]) * 1e-9,
                gyro=np.array([float(row[1]), float(row[2]), float(row[3])]),
                accel=np.array([float(row[4]), float(row[5]), float(row[6])]),
            ))
    samples.sort(key=lambda s: s.t)
    return ImuStream(samples)


def preintegrate(samples, t_start, t_end, bias_gyro=None, bias_accel=None):
    bias_gyro = np.zeros(3) if bias_gyro is None else np.asarray(bias_gyro, float)
    bias_accel = np.zeros(3) if bias_accel is None else np.asarray(bias_accel, float)

    times = _times_of(samples)
    window = _bracketed_window(samples, times, float(t_start), float(t_end))

    state = (np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)),
             np.zeros((3, 3)))
    for prev, curr in zip(window[:-1], window[1:]):
        state = _integrate_step(state, prev, curr, bias_gyro, bias_accel)
    dR, dv, dp, dR_dbg, dv_dba, dp_dba = state

    return Preintegrated(dt=float(t_end - t_start), dR=dR, dv=dv, dp=dp, dR_dbg=dR_dbg,
                         dv_dba=dv_dba, dp_dba=dp_dba)


def preintegrate_series(samples, t_start, t_targets, bias_gyro=None, bias_accel=None):
    bias_gyro = np.zeros(3) if bias_gyro is None else np.asarray(bias_gyro, float)
    bias_accel = np.zeros(3) if bias_accel is None else np.asarray(bias_accel, float)

    targets = [float(t) for t in t_targets]
    if not targets:
        return []
    t_start = float(t_start)
    times = _times_of(samples)
    if len(times) < 2:
        raise ValueError("need >=2 IMU samples, got %d" % len(times))
    tol = 1e-9
    t_max = max(targets)
    if t_start < times[0] - tol or t_max > times[-1] + tol:
        raise ValueError(
            "requested interval [%.6f, %.6f] exceeds IMU support [%.6f, %.6f]"
            % (t_start, t_max, times[0], times[-1]))
    t_start_c = max(t_start, float(times[0]))

    state = (np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)),
             np.zeros((3, 3)))
    prev = _interp_at(samples, times, t_start_c)
    i = int(np.searchsorted(times, t_start_c, side="right"))

    out = []
    for t_tgt in targets:
        t_tgt_c = min(t_tgt, float(times[-1]))
        while i < len(times) and times[i] < t_tgt_c:
            curr = samples[i]
            state = _integrate_step(state, prev, curr, bias_gyro, bias_accel)
            prev = curr
            i += 1
        end = _interp_at(samples, times, t_tgt_c)
        dR, dv, dp, dR_dbg, dv_dba, dp_dba = _integrate_step(
            state, prev, end, bias_gyro, bias_accel)
        out.append(Preintegrated(dt=float(t_tgt - t_start), dR=dR, dv=dv, dp=dp,
                                 dR_dbg=dR_dbg, dv_dba=dv_dba, dp_dba=dp_dba))
    return out
