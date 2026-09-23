from pathlib import Path
import queue
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]

RIG = ROOT / 'config/euroc/estimator_config.yaml'


def test_shadow_loop_bootstraps_once_before_the_first_frame_at_or_after_t_init(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, cfg): pass
        def feed_imu(self, t, w, a): calls.append(('imu', round(t, 3)))
        def feed_camera(self, t, img, cam_id=0): calls.append(('cam', round(t, 3)))
        def initialized(self): return False
        def initialize(self, state): calls.append(('init', round(state['t'], 3)))
        def close(self): pass

    monkeypatch.setitem(sys.modules, 'davio.backends.openvins', types.SimpleNamespace(OpenVinsBackend=Fake))
    from davio.runtime.workers import shadow_loop
    packets = [dict(t=t, image=np.zeros((2, 2), np.uint8),
                    imu=[(t - .01, np.zeros(3), np.zeros(3)), (t + .01, np.zeros(3), np.zeros(3))])
               for t in (1.0, 1.05, 1.10)]
    inbox, outbox = queue.Queue(), queue.Queue()
    inbox.put(None)
    shadow_loop(inbox, outbox, 'cfg', packets, 0,
                bootstrap=dict(t=1.05, q_GtoI=[0, 0, 0, 1], p=[0] * 3, v=[0] * 3, bg=[0] * 3, ba=[0] * 3, sigmas=[.1] * 15))
    kinds = [c[0] for c in calls]
    assert kinds.count('init') == 1
    i = kinds.index('init')
    # IMU up to the bracketing sample, then the bootstrap, then the frame at that time.
    assert calls[i - 1] == ('imu', 1.06) and calls[i + 1] == ('cam', 1.05)


def test_shadow_loop_without_bootstrap_never_calls_initialize(monkeypatch):
    class Fake:
        def __init__(self, cfg): pass
        def feed_imu(self, *a): pass
        def feed_camera(self, *a): pass
        def initialized(self): return False
        def initialize(self, *a): raise AssertionError('rotation/bias-only candidates never bootstrap')
        def close(self): pass

    monkeypatch.setitem(sys.modules, 'davio.backends.openvins', types.SimpleNamespace(OpenVinsBackend=Fake))
    from davio.runtime.workers import shadow_loop
    packets = [dict(t=1.0, image=np.zeros((2, 2), np.uint8), imu=[(1.01, np.zeros(3), np.zeros(3))])]
    inbox, outbox = queue.Queue(), queue.Queue()
    inbox.put(None)
    shadow_loop(inbox, outbox, 'cfg', packets, 0)
    assert outbox.empty()


# --- materialize: what each mode writes into the filter config ------------------------

from davio.runtime.configuration import materialize, read_relative, read_yaml   # noqa: E402

PRIORS = dict(rotation_deg=3., translation_m=.05, focal_px=20., distortion=.1, time_offset_s=.02)
EUROC_K = [458.654, 457.296, 367.215, 248.375]


def _cand(**kw):
    c = dict(R_CtoI=np.eye(3).tolist(), bg=[0, 0, 0], p_CinI=[.01, .02, .03],
             bootstrap=dict(t=1., q_GtoI=[0, 0, 0, 1], p=[0] * 3, v=[0] * 3, bg=[0] * 3, ba=[0] * 3, sigmas=[.1] * 15))
    c.update(kw)
    return c


def test_extrinsics_free_uses_no_supplied_extrinsics_but_keeps_intrinsics(tmp_path):
    path = materialize(RIG, tmp_path / 'x', {}, _cand(), mode='extrinsics_free', priors=PRIORS)
    cam = read_relative(path, 'relative_config_imucam')['cam0']
    est = read_yaml(path)
    t = np.asarray(cam['T_imu_cam'])
    np.testing.assert_allclose(t[:3, :3], np.eye(3))
    np.testing.assert_allclose(t[:3, 3], [.01, .02, .03])
    assert cam['intrinsics'] == EUROC_K and cam['timeshift_cam_imu'] == 0.
    assert est['calib_cam_extrinsics'] and est['calib_cam_timeoffset'] and not est['calib_cam_intrinsics']
    assert est['init_dyn_use'] is False and est['init_imu_thresh'] > 0 and est['init_max_disparity'] <= 1e-6
    assert est['init_prior_qc'] == pytest.approx(np.radians(3.)) and est['init_prior_t'] == .02
    assert 'cam1' not in read_relative(path, 'relative_config_imucam')


def test_free_mode_uses_only_the_candidate_camera(tmp_path):
    path = materialize(RIG, tmp_path / 'f', {}, _cand(K_raw=[400., 401., 376., 240.], D=[0., 0., 0., 0.]),
                       mode='free', priors=PRIORS)
    cam = read_relative(path, 'relative_config_imucam')['cam0']
    est = read_yaml(path)
    assert cam['intrinsics'] == [400., 401., 376., 240.] and cam['distortion_coeffs'] == [0., 0., 0., 0.]
    assert cam['resolution'] == [752, 480] and est['calib_cam_intrinsics'] is True


def test_free_modes_refuse_incomplete_inputs(tmp_path):
    with pytest.raises(ValueError):
        materialize(RIG, tmp_path / 'g', {}, _cand(), mode='free', priors=PRIORS)          # no K_raw
    with pytest.raises(ValueError):
        materialize(RIG, tmp_path / 'h', {}, None, mode='extrinsics_free', priors=PRIORS)  # no candidate
    with pytest.raises(ValueError):
        materialize(RIG, tmp_path / 'i', {}, _cand(), mode='extrinsics_free')             # no priors


def test_supplied_mode_is_unchanged_for_legacy_candidates(tmp_path):
    legacy = dict(R_CtoI=np.eye(3).tolist(), bg=[.01, .02, .03])
    path = materialize(RIG, tmp_path / 'a', {}, legacy)
    a = read_yaml(path)
    assert a['init_dyn_use'] is True and a['init_dyn_bias_g'] == [.01, .02, .03]
    cam = read_relative(path, 'relative_config_imucam')['cam0']
    np.testing.assert_allclose(np.asarray(cam['T_imu_cam'])[:3, 3], [-0.0216401454975, -0.064676986768, 0.00981073058949])


def test_supplied_mode_with_bootstrap_disables_the_native_initializer(tmp_path):
    est = read_yaml(materialize(RIG, tmp_path / 'b', {}, _cand()))
    assert est['init_dyn_use'] is False and est['calib_cam_intrinsics'] is True   # rig's own flag untouched


# --- assistant: the feed-forward candidate -------------------------------------------

def _prediction(s):
    return SimpleNamespace(depth=s['depth'], conf=s['conf'], intrinsics=s['intrinsics'],
                           extrinsics=s['extrinsics'],
                           processed_images=np.zeros((len(s['depth']), 240, 320, 3), np.uint8))


def _task(s):
    return dict(times=list(s['times']), imu=[(m.t, m.gyro, m.accel) for m in s['imu']],
                available_sensor_time=float(s['times'][-1]))


def _settings(**kw):
    base = dict(max_windows=5, feedforward=True, validate=True, solver='joint', validation_rms_deg=1.,
                validation_max_deg=3., max_bias_norm_rad_s=.3, gates={},
                ff=dict(points_per_frame=60, conf_quantile=0., ransac_iterations=60))
    base.update(kw)
    return base


def test_propose_supplied_mode_releases_a_bootstrap_without_hand_eye_windows():
    from test_feedforward import synthetic, G
    from davio.runtime.assistance import CalibrationAssistant
    s = synthetic()
    calib = dict(mode='supplied', R_CtoI=s['r_ctoi'].tolist(), p_CinI=s['p_cini'].tolist(),
                 K=s['intrinsics'][0].tolist(), D=[0.] * 4, resolution=(320, 240), gravity_mag=G)
    a = CalibrationAssistant(_settings(), calib)
    out = a.propose(_task(s), _prediction(s))
    assert out['status'] == 'released', out
    assert abs(out['ff']['scale'] - s['scale']) < .1 * s['scale']
    assert out['bootstrap']['t'] == s['times'][-1] and len(out['bootstrap']['sigmas']) == 15
    np.testing.assert_allclose(out['R_CtoI'], s['r_ctoi'])
    assert out['rotation'] is None and 'K_raw' not in out        # nothing the rig already has


def test_propose_free_mode_scales_da3_intrinsics_to_the_raw_image():
    from test_feedforward import synthetic, G, perpendicular
    from davio.runtime.assistance import CalibrationAssistant
    s = synthetic()
    calib = dict(mode='free', R_CtoI=None, p_CinI=None, K=None, D=None, resolution=(640, 480), gravity_mag=G)
    a = CalibrationAssistant(_settings(validate=False), calib)
    a.solve_rotation = lambda task, pred, times, imu: (s['r_ctoi'], np.zeros(3), dict(status='released'))
    out = a.propose(_task(s), _prediction(s))
    assert out['status'] == 'released', out
    np.testing.assert_allclose(out['K_raw'], [600., 600., 320., 240.])   # 300 px on a 320 grid; raw is 640 wide
    np.testing.assert_allclose(out['p_CinI'], 0.)                          # held at the prior by default
    assert not out['ff']['lever_arm_estimated'] and out['D'] == [0., 0., 0., 0.]


def test_propose_without_calibration_keeps_the_legacy_rotation_candidate():
    """CalibrationAssistant(settings) alone is the pre-DAVIO-CF interface: no bootstrap."""
    from test_feedforward import synthetic
    from davio.runtime.assistance import CalibrationAssistant
    s = synthetic()
    a = CalibrationAssistant(_settings(validate=False))
    a.solve_rotation = lambda task, pred, times, imu: (s['r_ctoi'], np.zeros(3), dict(status='released', R_CtoI=s['r_ctoi'].tolist(), bg=[0, 0, 0]))
    out = a.propose(_task(s), _prediction(s))
    assert out['status'] == 'released' and 'bootstrap' not in out


def test_supplied_mode_needs_the_rig_extrinsics():
    from davio.runtime.assistance import CalibrationAssistant
    with pytest.raises(ValueError):
        CalibrationAssistant(_settings(), dict(mode='supplied', R_CtoI=None, p_CinI=None))


# --- engine: free modes build no native instance ---------------------------------------

class ThreadWorker:
    """In-process stand-in for runtime.workers.Worker so a monkeypatched shim is visible."""

    def __init__(self, target, args=(), capacity=1):
        import threading
        self.input, self.output = queue.Queue(maxsize=capacity), queue.Queue()
        self.thread = threading.Thread(target=target, args=(self.input, self.output, *args), daemon=True)
        self.thread.start()
        self.process = SimpleNamespace(is_alive=self.thread.is_alive)

    def submit(self, value):
        import time
        before = self.output.qsize()
        self.input.put(value)
        if value is None:
            return True
        for _ in range(2000):                      # deterministic: wait for this packet's result
            if self.output.qsize() > before or not self.thread.is_alive():
                return True
            time.sleep(.001)
        return True

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.output.get_nowait())
            except queue.Empty:
                return items

    def close(self):
        self.submit(None)
        self.thread.join(timeout=2.)


def test_engine_free_mode_builds_no_native_instance_and_selects_the_bootstrapped_one(tmp_path, monkeypatch):
    import json
    import yaml
    from davio.data.replay import interval, replay
    import davio.runtime.engine as engine_module
    from davio.runtime.engine import Engine
    from test_system import FakeDataset

    class Vio:
        def __init__(self, cfg): self.t, self.n, self.boot = None, 0, False
        def feed_imu(self, *a): pass
        def feed_camera(self, t, img, cam_id=0): self.t, self.n = t, self.n + self.boot
        def initialized(self): return self.n > 0
        def initialize(self, *a): self.boot = True
        def get_state(self):
            return dict(t=self.t, q_GtoI=[0, 0, 0, 1], p=[0, 0, 0], v=[0, 0, 0], bg=[0] * 3, ba=[0] * 3,
                        q_ItoC=[0, 0, 0, 1], p_IinC=[0, 0, 0], cam_k=[400, 400, 376, 240, 0, 0, 0, 0], dt=0.)

    monkeypatch.setitem(sys.modules, 'openvins_ext', types.SimpleNamespace(VioManager=Vio))
    monkeypatch.setattr(engine_module, 'Worker', ThreadWorker)
    cfg = yaml.safe_load((ROOT / 'config/system.yaml').read_text())
    cfg['seed'] = 0
    cfg['calibration']['mode'] = 'extrinsics_free'
    cfg['mapping']['enabled'] = False
    engine = Engine(cfg, RIG, tmp_path / 'run', meta=dict(dataset='euroc', sequence='fake'))
    assert engine.native is None and cfg['assistance']['native_priority'] is False

    class Vision:                                   # a canned released candidate
        def __init__(self): self.sent, self.process = False, SimpleNamespace(is_alive=lambda: True)
        def submit(self, task): self.sent = True; return True
        def drain(self):
            if not self.sent:
                return []
            self.sent = False
            t = engine.last_t
            return [dict(kind='assist', payload=dict(
                status='released', R_CtoI=np.eye(3).tolist(), bg=[0] * 3, p_CinI=[0] * 3, available_sensor_time=t,
                bootstrap=dict(t=t, q_GtoI=[0, 0, 0, 1], p=[0] * 3, v=[0] * 3, bg=[0] * 3, ba=[0] * 3, sigmas=[.1] * 15)))]
        def close(self): pass

    engine.vision = Vision()
    ds = FakeDataset(n=60)
    ds.load_filter_image = lambda stamp: np.zeros((480, 752), np.uint8)
    replay(engine, ds, *interval(ds), rate=0.)
    engine.close()
    meta = json.loads((tmp_path / 'run/run.json').read_text())
    assert meta['selected'] == 'assisted' and meta['state'] == 'completed', meta.get('error')
    assert not (tmp_path / 'run/native_config').exists()
    cam = read_relative(tmp_path / 'run/assisted_config/estimator.yaml', 'relative_config_imucam')['cam0']
    assert cam['intrinsics'] == EUROC_K                      # supplied intrinsics, candidate extrinsics
    np.testing.assert_allclose(np.asarray(cam['T_imu_cam'])[:3, 3], 0.)
    kinds = [json.loads(l)['kind'] for l in (tmp_path / 'run/events.jsonl').read_text().splitlines() if l]
    assert kinds.count('candidate_started') == 1


def test_engine_retries_after_a_candidate_cannot_catch_up(tmp_path, monkeypatch):
    import json
    import yaml
    from davio.data.replay import interval, replay
    import davio.runtime.engine as engine_module
    from davio.runtime.engine import Engine
    from test_system import FakeDataset

    class Vio:
        def __init__(self, cfg): self.t, self.n, self.boot = None, 0, False
        def feed_imu(self, *a): pass
        def feed_camera(self, t, img, cam_id=0): self.t, self.n = t, self.n + self.boot
        def initialized(self): return self.n > 0
        def initialize(self, *a): self.boot = True
        def get_state(self):
            return dict(t=self.t, q_GtoI=[0, 0, 0, 1], p=[0, 0, 0], v=[0, 0, 0], bg=[0] * 3, ba=[0] * 3,
                        q_ItoC=[0, 0, 0, 1], p_IinC=[0, 0, 0], cam_k=[400, 400, 376, 240, 0, 0, 0, 0], dt=0.)

    shadows = []

    class FlakyWorker(ThreadWorker):
        """The first bootstrapped instance never accepts a packet; later ones are normal."""
        def __init__(self, target, args=(), capacity=1):
            super().__init__(target, args, capacity)
            self.refuse = False
            if getattr(target, '__name__', '') == 'shadow_loop':
                shadows.append(self)
                self.refuse = len(shadows) == 1

        def submit(self, value):
            if self.refuse and value is not None:
                return False
            return super().submit(value)

    monkeypatch.setitem(sys.modules, 'openvins_ext', types.SimpleNamespace(VioManager=Vio))
    monkeypatch.setattr(engine_module, 'Worker', FlakyWorker)
    cfg = yaml.safe_load((ROOT / 'config/system.yaml').read_text())
    cfg['seed'] = 0
    cfg['calibration']['mode'] = 'extrinsics_free'
    cfg['mapping']['enabled'] = False
    cfg['assistance']['max_attempts'] = 3
    engine = Engine(cfg, RIG, tmp_path / 'run', meta=dict(dataset='euroc', sequence='fake'))

    class Vision:                                   # a canned released candidate on request
        def __init__(self): self.sent, self.process = False, SimpleNamespace(is_alive=lambda: True)
        def submit(self, task): self.sent = True; return True
        def drain(self):
            if not self.sent:
                return []
            self.sent = False
            t = engine.last_t
            return [dict(kind='assist', payload=dict(
                status='released', R_CtoI=np.eye(3).tolist(), bg=[0] * 3, p_CinI=[0] * 3, available_sensor_time=t,
                bootstrap=dict(t=t, q_GtoI=[0, 0, 0, 1], p=[0] * 3, v=[0] * 3, bg=[0] * 3, ba=[0] * 3, sigmas=[.1] * 15)))]
        def close(self): pass

    engine.vision = Vision()
    ds = FakeDataset(n=60)
    ds.load_filter_image = lambda stamp: np.zeros((480, 752), np.uint8)
    replay(engine, ds, *interval(ds), rate=0.)
    engine.close()
    events = [json.loads(l) for l in (tmp_path / 'run/events.jsonl').read_text().splitlines() if l]
    kinds = [e['kind'] for e in events]
    assert [e['reason'] for e in events if e['kind'] == 'candidate_rejected'] == ['cannot catch up']
    assert kinds.count('candidate_started') == 2, kinds
    assert kinds.index('candidate_rejected') < kinds.index('candidate_started', kinds.index('candidate_started') + 1)
    meta = json.loads((tmp_path / 'run/run.json').read_text())
    assert meta['selected'] == 'assisted' and meta['state'] == 'completed', meta.get('error')


# --- evaluation: calibration convergence -----------------------------------------------

def test_calibration_convergence_from_pose_events():
    from evaluate_run import calibration_convergence
    from scipy.spatial.transform import Rotation
    from davio.init.jpl import rot_to_quat
    ref = dict(R_CtoI=np.eye(3), p_IC=np.array([.1, 0., 0.]), K=np.diag([400., 400., 1.]), D=np.zeros(4), td=0.)
    ref['K'][:2, 2] = [376., 240.]
    events = []
    for t in np.arange(0., 4., .5):
        rot = Rotation.from_euler('z', 5. * np.exp(-t), degrees=True).as_matrix()      # R_CtoI estimate
        events.append(dict(kind='pose', t=t, state=dict(
            q_ItoC=rot_to_quat(rot.T).tolist(), p_IinC=(-rot.T @ np.array([.1 - .02 * np.exp(-t), 0, 0])).tolist(),
            cam_k=[400 + 10 * np.exp(-t), 400, 376, 240, 0, 0, 0, 0], dt=.001 * np.exp(-t))))
    out = calibration_convergence(events, ref, start=0., checkpoints=(0., 2., 3.))
    assert out['at']['0']['rotation_deg'] == pytest.approx(5., abs=1e-6)
    assert out['at']['0']['rotation_deg'] > out['at']['2']['rotation_deg'] > out['final']['rotation_deg'] > 0.
    assert out['at']['0']['translation_m'] == pytest.approx(.02, abs=1e-9)
    assert out['final']['translation_m'] < .001 and out['final']['focal_px'] < .5
    assert out['at']['0']['time_offset_ms'] == pytest.approx(1.) and out['final']['time_offset_ms'] < .05
    assert calibration_convergence([dict(kind='vision')], ref, 0.) is None
