from pathlib import Path
import json
import re
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pathlib_tmp():
    import tempfile
    return Path(tempfile.mkdtemp())

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from davio.runtime.configuration import materialize, perturb_rig, read_relative, read_yaml   # noqa: E402
from davio.runtime.geometry import camera_pose, json_safe                     # noqa: E402
from davio.runtime.workers import feed_packet                                 # noqa: E402
from davio_mapper import sim3                                                 # noqa: E402
from davio_mapper.graph import Factor, Graph                                  # noqa: E402
from davio_mapper.metric import initialize_metric_submap                      # noqa: E402

RIG = ROOT / 'config/euroc/estimator_config.yaml'


# --- the assistance interface -------------------------------------------------------

def test_candidate_changes_only_rotation_and_dynamic_bias(tmp_path):
    """The paper's central interface claim: translation, intrinsics and timing are untouched."""
    native = materialize(RIG, tmp_path / 'native', {})
    assisted = materialize(RIG, tmp_path / 'assisted', {},
                           dict(R_CtoI=np.eye(3).tolist(), bg=[.01, .02, .03]))
    n, a = read_yaml(native), read_yaml(assisted)
    cam_n = read_relative(native, 'relative_config_imucam')['cam0']
    cam_a = read_relative(assisted, 'relative_config_imucam')['cam0']
    tn, ta = np.asarray(cam_n['T_imu_cam']), np.asarray(cam_a['T_imu_cam'])
    np.testing.assert_allclose(ta[:3, :3], np.eye(3))
    np.testing.assert_allclose(ta[:3, 3], tn[:3, 3])          # camera origin preserved
    assert cam_a['intrinsics'] == cam_n['intrinsics']
    assert cam_a['distortion_coeffs'] == cam_n['distortion_coeffs']
    assert a['init_dyn_bias_g'] == [.01, .02, .03]
    assert a['init_dyn_bias_a'] == n['init_dyn_bias_a']       # no accelerometer claim
    assert a['calib_cam_timeoffset'] == n['calib_cam_timeoffset']


def test_improper_candidate_rotation_is_rejected(tmp_path):
    for bad in (np.diag([1., 1., -1.]), 2 * np.eye(3)):
        with pytest.raises(ValueError):
            materialize(RIG, tmp_path / str(id(bad)), {},
                        dict(R_CtoI=bad.tolist(), bg=[0., 0., 0.]))


def test_state_injection_only_at_bootstrap():
    source = (ROOT / 'native/openvins_binding.cpp').read_text()
    assert source.count('initialize_with_gt(') == 1 and '.def("initialize"' in source
    assert 'initialized_time() >= 0' in source                 # the refusal guard
    assert 'set_external_seed' not in source and 'ExternalSeed' not in source
    from davio.backends.openvins import OpenVinsBackend
    public = set(k for k in vars(OpenVinsBackend) if not k.startswith('_'))
    assert public == {'feed_imu', 'feed_camera', 'initialized', 'initialize', 'state',
                      'diagnostics', 'close'}


def test_native_packet_never_injects_state():
    """The native packet path must not bootstrap; only a shadow candidate may, once."""
    class Native:
        def feed_imu(self, *a): pass
        def feed_camera(self, *a): pass
        def initialized(self): return False
        def initialize(self, *a): raise AssertionError('feed_packet must never bootstrap by default')
    packet = dict(t=1., image=np.zeros((8, 8), np.uint8), imu=[(1.01, np.zeros(3), np.zeros(3))])
    assert feed_packet(Native(), packet) is None


def test_repeated_state_is_not_a_fresh_update():
    """A filter echoing an old state must not be counted as tracking the current frame."""
    class Native:
        def feed_imu(self, *a): pass
        def feed_camera(self, *a): pass
        def initialized(self): return True
        def state(self): return dict(t=0., q_GtoI=[0, 0, 0, 1], p=[0, 0, 0], v=[0, 0, 0],
                                     bg=[0, 0, 0], ba=[0, 0, 0], q_ItoC=[0, 0, 0, 1],
                                     p_IinC=[0, 0, 0])
    assert feed_packet(Native(), dict(t=1., imu=[], image=np.zeros((8, 8), np.uint8))) is None


def test_alternating_arm_runs_no_joint_iterations():
    """The ablation must differ in the SOLVER, not in the data budget or the gate."""
    from davio.init.joint_calibration import solve_joint_calibration
    from davio.init.preintegration import ImuSample
    times = np.linspace(0., 1., 5)
    imu = [ImuSample(float(t), np.array([.4, .2, .1]), np.array([0., 0., 9.81]))
           for t in np.linspace(-.1, 1.2, 300)]
    windows = [SimpleNamespace(t0=times[0], frame_times=times,
                               camera_rotations=np.repeat(np.eye(3)[None], 5, axis=0))
               for _ in range(2)]
    result = solve_joint_calibration(imu, windows, alternating_only=True)
    assert result.n_iterations == 0
    assert result.termination_reason.startswith(('alternating', 'handeye_init_failed'))


# --- geometry and output ------------------------------------------------------------

def test_camera_pose_accounts_for_lever_arm():
    state = dict(q_GtoI=[0, 0, 0, 1], p=[1, 2, 3], q_ItoC=[0, 0, 0, 1], p_IinC=[-.2, 0, 0])
    np.testing.assert_allclose(camera_pose(state)[:3, 3], [1.2, 2, 3])


def test_optional_nonfinite_diagnostic_cannot_break_json():
    assert json.loads(json.dumps(json_safe(dict(cost=float('inf'))),
                                 allow_nan=False)) == dict(cost=None)


def test_sim3_exp_log_round_trip_and_validation():
    x = np.array([.3, -.2, .1, .05, -.4, .2, .3])
    np.testing.assert_allclose(sim3.log(sim3.exp(x)), x, atol=1e-10)
    with pytest.raises(ValueError):
        sim3.validate(np.diag([1., 2., 3., 1.]))              # affine is not Sim(3)
    with pytest.raises(ValueError):
        sim3.validate(np.diag([-1., 1., 1., 1.]))             # negative scale


# --- mapping ------------------------------------------------------------------------

def test_stationary_mapping_does_not_invent_metric_scale(tmp_path):
    from davio_mapper.online import OnlineMapper
    mapper = OnlineMapper(dict(orb_features=50, min_baseline_m=.05), tmp_path)
    prediction = SimpleNamespace(depth=np.ones((5, 32, 32)),
                                 processed_images=np.zeros((5, 32, 32, 3), np.uint8),
                                 extrinsics=np.repeat(np.eye(4)[None], 5, axis=0),
                                 intrinsics=np.repeat(np.eye(3)[None], 5, axis=0))
    result = mapper.add(dict(camera_poses=np.repeat(np.eye(4)[None], 5, axis=0)), prediction)
    assert result['status'] == 'deferred'


def test_metric_seed_never_claims_scale_it_did_not_observe():
    """A negative signed fit is an initialization incompatibility, not a metric anchor."""
    local = np.repeat(np.eye(4)[None], 3, axis=0)
    local[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    world = np.repeat(np.eye(4)[None], 3, axis=0)
    world[:, :3, 3] = [[0, 0, 0], [-1, 0, 0], [-2, 0, 0]]     # translations disagree
    _t, observable, info = initialize_metric_submap(local, world, return_info=True)
    assert observable and not info['positive_interior_fit']
    assert not info['metric_scale_anchored'] and info['seed_source'] == 'rms_baseline_ratio'


def test_each_frame_contributes_to_the_map_once():
    from davio_mapper.mapping import fuse_map
    sm = dict(id='0', frame_ids=['a', 'b'], depth=np.ones((2, 4, 4)),
              rgb=np.zeros((2, 4, 4, 3), np.uint8),
              intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
              poses=np.repeat(np.eye(4)[None], 2, axis=0))
    shared = dict(sm, id='1')                                  # same frame_ids, second submap
    _xyz, _rgb, contributed = fuse_map([sm, shared], {'s:0': np.eye(4), 's:1': np.eye(4)},
                                       voxel_size=.1, pixel_step=1)
    assert contributed == 2 * 4 * 4                            # two frames, not four


def test_graph_requires_a_gauge_and_descends():
    graph = Graph()
    graph.add_node('a', np.eye(4), dimensions=0)
    pose = np.eye(4)
    pose[:3, 3] = [1., 0., 0.]
    graph.add_node('b', pose, dimensions=6)
    disconnected = Graph()
    disconnected.add_node('a', np.eye(4), dimensions=0)
    disconnected.add_node('b', pose, dimensions=6)
    with pytest.raises(ValueError, match='disconnected'):
        disconnected.optimize()
    truth = np.eye(4)
    truth[:3, 3] = [1.2, 0., 0.]
    graph.add(Factor('a', 'b', 'temporal', measurement=truth, projected=True,
                     sigmas=np.r_[np.full(3, .05), np.full(3, .02)]))
    info = graph.optimize(max_iterations=20)
    assert info['final_cost'] < info['initial_cost']
    np.testing.assert_allclose(graph.nodes['b'][:3, 3], [1.2, 0., 0.], atol=1e-6)


# --- transport ----------------------------------------------------------------------

class FakeDataset:
    """20 Hz camera, 200 Hz IMU, nothing on disk."""
    def __init__(self, n=10):
        self.stamps = np.arange(n) * 50_000_000 + 1_000_000_000
        self.samples = [SimpleNamespace(t=1. + i * .005, gyro=np.zeros(3), accel=np.zeros(3))
                        for i in range(n * 10 + 10)]

    def imu(self): return self.samples
    def image_stamps(self): return self.stamps
    def load_filter_image(self, stamp): return np.zeros((4, 4), np.uint8)


def test_replay_packets_bracket_every_camera_time():
    from davio.data.replay import interval, replay
    ds = FakeDataset()
    packets = []
    engine = SimpleNamespace(step=packets.append)
    start, end = interval(ds)
    replay(engine, ds, start, end, rate=0.)
    assert packets
    for packet in packets:
        assert packet['imu'] and packet['imu'][-1][0] > packet['t']
    assert [p['t'] for p in packets] == sorted(p['t'] for p in packets)


def test_replay_charges_arrival_at_the_bracketing_sample():
    """Publication age must include the IMU the filter had to wait for, not just the image."""
    from davio.data.replay import interval, replay
    ds = FakeDataset(n=4)
    packets = []
    replay(SimpleNamespace(step=packets.append), ds, *interval(ds), rate=1.)
    spans = np.diff([p['arrival_wall'] for p in packets])
    np.testing.assert_allclose(spans, .05, atol=5e-3)


def test_truncated_imu_tail_ends_the_replay_instead_of_feeding_unbracketed_images():
    from davio.data.replay import interval, replay
    ds = FakeDataset(n=6)
    ds.samples = ds.samples[:41]                               # IMU stops mid-sequence
    packets = []
    replay(SimpleNamespace(step=packets.append), ds, *interval(ds), rate=0.)
    assert 0 < len(packets) < 6


# --- provenance and evaluation ------------------------------------------------------

def test_perturbation_moves_the_rotation_prior_only(tmp_path):
    perturbed = perturb_rig(RIG, tmp_path / 'rig', 10., seed=3)
    base = read_relative(materialize(RIG, tmp_path / 'base', {}), 'relative_config_imucam')
    after = read_relative(perturbed, 'relative_config_imucam')
    t0 = np.asarray(base['cam0']['T_imu_cam'], float)
    t1 = np.asarray(after['cam0']['T_imu_cam'], float)
    np.testing.assert_allclose(t1[:3, 3], t0[:3, 3])
    assert after['cam0']['intrinsics'] == base['cam0']['intrinsics']
    angle = np.degrees(np.arccos(np.clip((np.trace(t0[:3, :3] @ t1[:3, :3].T) - 1) / 2, -1, 1)))
    assert abs(angle - 10.) < 1e-6


def test_rigid_alignment_fits_no_scale():
    """A scaled estimate must show up as error, not be absorbed by the alignment."""
    from davio.eval.metrics import se3_align
    reference = np.random.default_rng(0).normal(size=(3, 30))
    _r, _t, err = se3_align(2. * reference, reference)
    assert err > .1


def test_rpe_reports_an_impossible_interval_as_unavailable():
    from davio.eval import metrics
    gt = SimpleNamespace(t=np.linspace(0., 5., 51), p=np.zeros((51, 3)),
                         R=np.repeat(np.eye(3)[None], 51, axis=0), valid=None,
                         orientation_reliable=True)
    ds = SimpleNamespace(groundtruth=lambda: gt)
    trajectory = [(float(t), np.zeros(3), np.array([0., 0., 0., 1.]))
                  for t in np.linspace(0., 5., 51)]
    out = metrics.rpe(ds, trajectory, 0.)
    assert out[1.]['position_m'] == 0. and out[1.]['n_pairs'] > 0
    assert out[10.]['position_m'] is None and out[10.]['n_pairs'] == 0


def test_config_override_rejects_an_unknown_key(tmp_path):
    import run_davio
    with pytest.raises(SystemExit):
        run_davio.main(['--dataset', 'euroc', '--data', str(tmp_path), '--sequence', 'x',
                        '--out', str(tmp_path / 'run'), '--set', 'assistance.typo=1'])


class FakeVio:
    """Enough of ov_msckf::VioManager to drive the engine end to end, offline."""
    def __init__(self, config_path):
        self.t = None
        self.n = 0

    def feed_imu(self, t, wm, am):
        pass

    def feed_camera(self, t, image, cam_id=0):
        self.t = t
        self.n += 1

    def initialized(self):
        return self.n > 2

    def get_state(self):
        return dict(t=self.t, q_GtoI=[0., 0., 0., 1.], p=[self.n * .1, 0., 0.],
                    v=[.1, 0., 0.], bg=[0.] * 3, ba=[0.] * 3,
                    q_ItoC=[0., 0., 0., 1.], p_IinC=[0., 0., 0.])


def test_engine_drives_a_full_offline_run(tmp_path, monkeypatch):
    """Wiring check for the whole online path with assistance and mapping off."""
    import types
    import yaml
    from davio.data.replay import interval, replay
    from davio.runtime.engine import Engine
    monkeypatch.setitem(sys.modules, 'openvins_ext',
                        types.SimpleNamespace(VioManager=FakeVio))
    cfg = yaml.safe_load((ROOT / 'config/system.yaml').read_text())
    cfg['seed'] = 0
    cfg['assistance']['enabled'] = False
    cfg['mapping']['enabled'] = False
    ds = FakeDataset(n=8)
    ds.load_filter_image = lambda stamp: np.zeros((480, 752), np.uint8)
    out = tmp_path / 'run'
    engine = Engine(cfg, RIG, out, meta=dict(dataset='euroc', sequence='fake'))
    start, end = interval(ds)
    replay(engine, ds, start, end, rate=0.)
    engine.close()

    meta = json.loads((out / 'run.json').read_text())
    assert meta['state'] == 'completed' and meta['selected'] == 'native'
    assert meta['dataset'] == 'euroc' and meta['n_states'] > 0
    assert meta['publication_age_p50_s'] is not None
    poses = [line for line in (out / 'trajectory.tum').read_text().splitlines() if line]
    assert len(poses) == meta['n_states'] and all(len(p.split()) == 8 for p in poses)
    kinds = {json.loads(line)['kind']
             for line in (out / 'events.jsonl').read_text().splitlines() if line}
    assert {'selected', 'pose'} <= kinds
    # With no back-end running, the map frame is the odometry frame, stated explicitly
    # as an identity correction rather than by omitting the stream.
    assert (out / 'map_trajectory.tum').read_text() == (out / 'trajectory.tum').read_text()
    # The materialized native config is kept with the run, and is not the shared rig.
    assert (out / 'native_config' / 'estimator.yaml').exists()


def test_mapper_commits_a_scaled_submap_and_an_exportable_map(tmp_path):
    """Scale fit, rescale, graph, persistence and PLY export, on synthetic geometry."""
    # The default feature backend is XFeat, which needs torch and its checkout: present in
    # the image (./davio test), not on a bare host or in CI.
    pytest.importorskip('torch')
    if not (ROOT / 'thirdparty/accelerated_features/modules/xfeat.py').is_file():
        pytest.skip('XFeat checkout missing: make xfeat')
    import yaml
    from davio_mapper.online import OnlineMapper
    settings = yaml.safe_load((ROOT / 'config/system.yaml').read_text())['mapping']
    mapper = OnlineMapper(settings, tmp_path)
    n, size, scale = 5, 48, 4.
    # Visual prediction in its own units; the metric trajectory is `scale` times larger.
    local = np.repeat(np.eye(4)[None], n, axis=0)
    local[:, 0, 3] = np.linspace(0., .25, n)
    metric = np.repeat(np.eye(4)[None], n, axis=0)
    metric[:, 0, 3] = local[:, 0, 3] * scale
    k = np.array([[60., 0., size / 2], [0., 60., size / 2], [0., 0., 1.]])
    rng = np.random.default_rng(0)
    prediction = SimpleNamespace(
        depth=np.full((n, size, size), 2.),
        processed_images=rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
        extrinsics=np.linalg.inv(local), intrinsics=np.repeat(k[None], n, axis=0))
    stamps = [int(i * 5e7) for i in range(n)]
    result = mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps],
                             camera_poses=metric), prediction)
    assert result['status'] == 'mapped'
    assert result['scale'] == pytest.approx(scale, rel=1e-6)
    assert result['alignment_rmse_m'] < 1e-9
    index = json.loads((tmp_path / 'map/map_index.json').read_text())
    key = result['submap']
    assert index['submaps'][key]['scale'] == pytest.approx(scale, rel=1e-6)
    with np.load(tmp_path / 'map' / index['submaps'][key]['file']) as stored:
        # Depth and local translations are both in metres once the fit is applied.
        assert stored['depth'].max() == pytest.approx(2. * scale, rel=1e-6)
        # Poses are re-centred on the middle camera, so the last is half the span.
        assert stored['poses'][-1][0, 3] == pytest.approx(.125 * scale, rel=1e-6)
    assert (tmp_path / 'map/active.ply').stat().st_size > 0


# --- pose-graph back-end ------------------------------------------------------------

def _backend_config(**overrides):
    import yaml
    settings = yaml.safe_load((ROOT / 'config/system.yaml').read_text())['mapping']
    # These back-end contract tests describe the v1 Sim(3) chart without per-frame scales.
    settings.update(feature_backend='orb', sparse_alignment=False, depth_filter=False, loop_min_inlier_ratio=.5,
                    graph_nodes='sim3', frame_scale_refinement=False, loop_min_drift_ratio=0.,
                    odometry_drift_m_per_m=.02)      # v1 drift model, which the C3 predeclaration froze
    settings.update(overrides)
    return settings


def _keyframe(rgb, depth, centres, scale):
    """A five-frame window whose DA3 prediction is the metric truth divided by `scale`."""
    n = len(centres)
    metric = np.repeat(np.eye(4)[None], n, axis=0)
    metric[:, :3, 3] = centres
    local = np.repeat(np.eye(4)[None], n, axis=0)
    local[:, :3, 3] = centres / scale
    k = np.array([[80., 0., depth.shape[1] / 2], [0., 80., depth.shape[0] / 2], [0., 0., 1.]])
    prediction = SimpleNamespace(depth=np.repeat(depth[None], n, axis=0),
                                 processed_images=np.repeat(rgb[None], n, axis=0),
                                 extrinsics=np.linalg.inv(local),
                                 intrinsics=np.repeat(k[None], n, axis=0))
    return metric, prediction


def _drive(mapper, waypoints, images, scale=2.):
    """One submap per waypoint, 0.1 m of in-window motion, one second apart."""
    size = images[0].shape[0]
    grid = np.linspace(0, 2 * np.pi, size)
    depth = 2. + .6 * np.outer(np.sin(3 * grid), np.cos(3 * grid))
    results = []
    for i, (centre, rgb) in enumerate(zip(waypoints, images)):
        offsets = np.zeros((5, 3))
        offsets[:, 0] = np.linspace(0., .1, 5)
        metric, prediction = _keyframe(rgb, depth, np.asarray(centre) + offsets, scale)
        stamps = [int((i + j * .05) * 1e9) for j in range(5)]
        results.append(mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps],
                                       camera_poses=metric), prediction))
    return results


def test_odometry_edge_is_the_vio_relative_pose_not_the_visual_fit():
    """The chain must carry OpenVINS's own relative pose; the fit residual is the submap's."""
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]
    mapper = OnlineMapper(_backend_config(loops_enabled=False), pathlib_tmp())
    _drive(mapper, [[0, 0, 0], [1., 0, 0], [2., 0, 0]], images)
    keys = list(mapper.nodes)
    pose_edges = [e for e in mapper.edges if e['kind'] != 'submap_scale']
    assert [e['kind'] for e in pose_edges] == ['odometry', 'odometry']
    for edge in pose_edges:
        expected = np.linalg.inv(mapper.nodes[edge['a']]['odom']) @ mapper.nodes[edge['b']]['odom']
        np.testing.assert_allclose(edge['measurement'], expected, atol=1e-12)
    # One metre between waypoints, so the first edge is exactly one metre of odometry.
    np.testing.assert_allclose(mapper.edges[0]['measurement'][:3, 3], [1., 0., 0.], atol=1e-9)
    assert len(keys) == 3


def test_odometry_only_chain_is_not_optimized_and_leaves_the_frames_aligned():
    """With no loop there is nothing to correct: cost is zero and map frame == odom frame."""
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
    mapper = OnlineMapper(_backend_config(loops_enabled=False, scale_coupling=False),
                          pathlib_tmp())
    results = _drive(mapper, [[i * 1., 0, 0] for i in range(4)], images)
    assert all(r['status'] == 'mapped' for r in results)
    assert all(r['accepted_loops'] == 0 and r['optimizer'] == {} for r in results)
    np.testing.assert_allclose(results[-1]['T_map_odom'], np.eye(4), atol=1e-9)


def test_overlap_coupling_pulls_back_a_submap_whose_window_fit_is_wrong():
    from davio_mapper import sim3
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)   # same scene throughout
    grid = np.linspace(0, 2 * np.pi, 128)
    depth = 2. + .6 * np.outer(np.sin(3 * grid), np.cos(3 * grid))
    mapper = OnlineMapper(_backend_config(loops_enabled=False), pathlib_tmp())
    keys = []
    for i, fit in enumerate([2., 2., 2.4, 2., 2.]):
        offsets = np.zeros((5, 3))
        offsets[:, 0] = np.linspace(0., .1, 5)
        metric, prediction = _keyframe(rgb, depth, np.array([i * .15, 0., 0.]) + offsets, fit)
        stamps = [int((i + j * .05) * 1e9) for j in range(5)]
        result = mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps],
                                 camera_poses=metric), prediction)
        assert result['status'] == 'mapped'
        keys.append(result['submap'])
    assert [e['kind'] for e in mapper.edges].count('submap_scale') >= 2
    corrections = [sim3.scale(mapper.nodes[k]['pose']) for k in keys]
    # Submap 2 is pulled towards 1/1.2; its correctly fitted neighbours stay put.
    assert .78 < corrections[2] < .95, corrections
    assert all(.95 < c < 1.05 for i, c in enumerate(corrections) if i != 2), corrections
    # The trajectory must not inherit any of it: T_map_odom stays a rigid transform.
    correction = np.asarray(result['T_map_odom'])
    np.testing.assert_allclose(sim3.scale(correction), 1., atol=1e-9)


def test_a_loop_corrects_nodes_whose_dense_geometry_is_long_gone():
    """The point of the split bound: the graph outlives the dense memory budget."""
    from davio_mapper.online import OnlineMapper
    rng = np.random.default_rng(3)
    revisit = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    others = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    # Out and back: the last waypoint revisits the first, seeing the same scene.
    waypoints = [[0, 0, 0], [.8, 0, 0], [1.6, 0, 0], [2.4, 0, 0],
                 [1.6, .6, 0], [.8, .6, 0], [.4, .2, 0]]
    # The joint (C1) graph is what this pins; the frozen default is scale-only nodes.
    mapper = OnlineMapper(_backend_config(max_active_submaps=3, loop_min_separation_s=2., graph_nodes='sim3'),
                          pathlib_tmp())
    # The last keyframe sees the first keyframe's scene again -- that is the revisit.
    results = _drive(mapper, waypoints, [revisit] + others + [revisit])
    final = results[-1]
    assert final['accepted_loops'] >= 1, 'no loop verified against the revisited keyframe'
    # The graph kept every node; only the dense geometry was bounded.
    assert final['nodes'] == len(waypoints) and final['resident_submaps'] == 3
    assert final['optimizer']['final_cost'] < final['optimizer']['initial_cost']
    # The correction is real, and it reached a node evicted from dense memory.
    assert not np.allclose(final['T_map_odom'], np.eye(4), atol=1e-6)
    evicted = list(mapper.nodes)[1]
    assert evicted not in mapper.geometry
    assert not np.allclose(mapper.nodes[evicted]['pose'], mapper.nodes[evicted]['odom'], atol=1e-6)
    index = json.loads((mapper.root / 'map_index.json').read_text())
    np.testing.assert_allclose(index['submaps'][evicted]['T_map_submap'],
                               mapper.nodes[evicted]['pose'], atol=1e-12)


def test_engine_applies_the_backend_correction_at_odometry_rate(tmp_path, monkeypatch):
    """The odom->map transform must reach the pose stream, not just the map index."""
    import types
    import yaml
    from davio.data.replay import interval, replay
    from davio.runtime.engine import Engine
    monkeypatch.setitem(sys.modules, 'openvins_ext',
                        types.SimpleNamespace(VioManager=FakeVio))
    cfg = yaml.safe_load((ROOT / 'config/system.yaml').read_text())
    cfg['seed'] = 0
    cfg['assistance']['enabled'] = False
    cfg['mapping']['enabled'] = False
    cfg['mapping']['correction_ramp_s'] = 0.   # this test pins the unramped, immediate application
    ds = FakeDataset(n=8)
    ds.load_filter_image = lambda stamp: np.zeros((480, 752), np.uint8)
    engine = Engine(cfg, RIG, tmp_path / 'run')
    correction = np.eye(4)
    correction[:3, 3] = [5., 0., 0.]
    delivered = [dict(kind='map', payload=dict(status='mapped', submap='000007',
                                               T_map_odom=correction.tolist()))]

    # A back-end result arriving mid-run on the vision outbox, without a real worker.
    polls = []

    def drain():
        polls.append(1)
        return [delivered.pop()] if len(polls) == 5 and delivered else []

    engine.vision = types.SimpleNamespace(
        drain=drain, close=lambda: None,
        process=types.SimpleNamespace(is_alive=lambda: True))
    start, end = interval(ds)
    replay(engine, ds, start, end, rate=0.)
    engine.close()

    odom = [line.split() for line in (tmp_path / 'run/trajectory.tum').read_text().splitlines()]
    mapped = [line.split() for line in
              (tmp_path / 'run/map_trajectory.tum').read_text().splitlines()]
    assert len(odom) == len(mapped) > 1
    shifts = [float(m[1]) - float(o[1]) for o, m in zip(odom, mapped)]
    # Odometry is untouched; the map stream picks the correction up when it arrives
    # and keeps it, rather than waiting for the next submap to republish a pose.
    assert shifts[0] == 0. and shifts[-1] == pytest.approx(5.)
    assert shifts == sorted(shifts) and set(shifts) == {0., 5.}
    # Frame consistency: each published pose names the commit whose correction it used,
    # and the activation itself is logged once with its sensor time.
    events = [json.loads(s) for s in (tmp_path / 'run/events.jsonl').read_text().splitlines()]
    versions = [e['correction_version'] for e in events if e['kind'] == 'pose']
    assert versions == [None if s == 0. else '000007' for s in shifts]
    activated = [e for e in events if e['kind'] == 'correction_activated']
    assert len(activated) == 1 and activated[0]['version'] == '000007'
    assert activated[0]['t'] <= float(mapped[shifts.index(5.)][0])


def _fake_euroc(root, n=10):
    """The smallest tree EurocDataset accepts: cam0 frames, IMU and reference CSVs."""
    import cv2
    mav0 = root / 'seq' / 'mav0'
    (mav0 / 'cam0/data').mkdir(parents=True)
    (mav0 / 'imu0').mkdir(parents=True)
    (mav0 / 'state_groundtruth_estimate0').mkdir(parents=True)
    rng = np.random.default_rng(7)
    stamps = np.arange(n) * 50_000_000 + 1_000_000_000
    for stamp in stamps:
        cv2.imwrite(str(mav0 / 'cam0/data' / f'{stamp}.png'),
                    rng.integers(0, 255, (480, 752), dtype=np.uint8))
    imu = ['#timestamp,wx,wy,wz,ax,ay,az']
    for i in range(n * 10 + 20):
        t = 1_000_000_000 + i * 5_000_000
        imu.append(f'{t},0,0,0,0,0,9.81')
    (mav0 / 'imu0/data.csv').write_text('\n'.join(imu) + '\n')
    truth = ['#timestamp,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz']
    for i in range(n * 4):
        t = 1_000_000_000 + i * 12_500_000
        x = .1 * i
        truth.append(f'{t},{x},0,0,1,0,0,0,1,0,0,0,0,0,0,0,0')
    (mav0 / 'state_groundtruth_estimate0/data.csv').write_text('\n'.join(truth) + '\n')
    return root


def test_rerun_export_writes_a_debuggable_recording(tmp_path, monkeypatch):
    """The debug artefact must survive a real run directory, not just import cleanly."""
    import types
    import yaml
    import rerun_export
    from davio.data.replay import interval, replay
    from davio.runtime.engine import Engine
    from davio.runtime.geometry import json_safe
    from davio_mapper.online import OnlineMapper
    monkeypatch.setitem(sys.modules, 'openvins_ext',
                        types.SimpleNamespace(VioManager=FakeVio))
    data = _fake_euroc(tmp_path / 'data')
    cfg = yaml.safe_load((ROOT / 'config/system.yaml').read_text())
    cfg['seed'] = 0
    cfg['assistance']['enabled'] = False
    cfg['mapping']['enabled'] = False
    from davio.data import open_dataset
    ds = open_dataset('euroc', data, seq='seq')
    run = tmp_path / 'run'
    engine = Engine(cfg, RIG, run, meta=dict(dataset='euroc', sequence='seq', mode='native',
                                             interval_start=0., interval_end=0.))
    start, end = interval(ds)
    engine.meta.update(interval_start=start, interval_end=end)
    replay(engine, ds, start, end, rate=0.)
    engine.close()

    # A back-end result written into the same run directory, without a GPU worker.
    mapper = OnlineMapper(_backend_config(max_active_submaps=3, loop_min_separation_s=2.),
                          run)
    rng = np.random.default_rng(8)
    revisit = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    others = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    results = _drive(mapper, [[0, 0, 0], [.8, 0, 0], [1.6, 0, 0], [2.4, 0, 0],
                              [1.6, .6, 0], [.8, .6, 0], [.4, .2, 0]],
                     [revisit] + others + [revisit])
    with (run / 'events.jsonl').open('a') as stream:
        for i, payload in enumerate(results):
            stream.write(json.dumps(json_safe(dict(
                kind='vision', wall=float(i),
                result=dict(kind='map', payload=payload, started_wall=float(i),
                            completed_wall=float(i) + .2, sensor_time=1. + i * .05))),
                allow_nan=False) + '\n')

    out = tmp_path / 'debug.rrd'
    sys.argv = ['rerun_export', '--run', str(run), '--data', str(data), '--out', str(out)]
    rerun_export.main()
    assert out.stat().st_size > 1000


def test_diagnose_names_the_gate_that_blocked(tmp_path):
    """Tuning needs the blocking gate, not a pass/fail count."""
    import diagnose
    run = tmp_path / 'seq_0_000000_assist_p0'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps(dict(
        state='completed', selected='native', dropped_optional_tasks=2,
        submitted_optional_tasks=30)))
    events = [
        dict(kind='vision', result=dict(kind='assist', payload=dict(
            status='rejected',
            reason='rotation/bias ambiguity: sigma_min(H_X|b) = 1.2e-09 < 1e-06'))),
        dict(kind='vision', result=dict(kind='assist', payload=dict(
            status='rejected',
            reason='rotation/bias ambiguity: sigma_min(H_X|b) = 4.0e-09 < 1e-06'))),
        dict(kind='vision', result=dict(kind='assist', payload=dict(
            status='rejected', reason='single-axis motion: axis coverage 0.004 < 0.02'))),
        dict(kind='vision', result=dict(kind='map', payload=dict(
            status='mapped', accepted_loops=2))),
    ]
    (run / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    counts, example, loops = diagnose.scan([run])
    # The two ambiguity rejections group into one gate despite differing numbers.
    assert counts['proposal_gate']['rotation/bias ambiguity'] == 2
    assert counts['proposal_gate']['single-axis motion'] == 1
    assert '1e-06' in example['rotation/bias ambiguity']
    assert counts['proposal']['rejected'] == 3 and loops['accepted'] == 2


def test_assistance_gates_are_configurable_and_typo_proof():
    """Tuning sweeps these; a silently inert name would waste a run."""
    from davio.runtime.assistance import CalibrationAssistant
    import yaml
    settings = yaml.safe_load((ROOT / 'config/system.yaml').read_text())['assistance']
    settings['gates']['joint_min_sigma_x_given_b'] = 1e-9
    assistant = CalibrationAssistant(settings)
    assert assistant.config.joint_min_sigma_x_given_b == 1e-9
    assert assistant.config.joint_min_axis_coverage == .02
    with pytest.raises(ValueError, match='unknown assistance gate'):
        CalibrationAssistant(dict(settings, gates={'min_sigma': 1e-9}))


def test_shadow_calibration_refuses_a_schedule_that_can_never_validate():
    """A diagnostic that defers every epoch looks like it is running and is not."""
    from davio.runtime.assistance import CalibrationAssistant
    base = dict(shadow_continuous=True, imu_history_s=10., window_frames=5,
                sample_period_s=.1)
    ok, reason = CalibrationAssistant.shadow_schedule_ok(dict(base, shadow_period_s=5.))
    assert not ok and 'defer' in reason
    assert CalibrationAssistant.shadow_schedule_ok(dict(base, shadow_period_s=2.))[0]
    # Off by default: the check must not constrain a run that never uses it.
    assert CalibrationAssistant.shadow_schedule_ok(
        dict(base, shadow_continuous=False, shadow_period_s=5.))[0]


def test_conditioned_window_with_a_wrong_metric_fit_is_rejected(tmp_path):
    from davio_mapper.online import OnlineMapper
    import yaml
    settings = yaml.safe_load((ROOT / 'config/system.yaml').read_text())['mapping']
    settings.update(feature_backend='orb', sparse_alignment=False, depth_filter=False,
                    frame_scale_refinement=False, da3_pose_conditioning=True)
    mapper = OnlineMapper(settings, tmp_path)
    n, size = 5, 48
    local = np.repeat(np.eye(4)[None], n, axis=0)
    local[:, 0, 3] = np.linspace(0., .25, n)
    metric = local.copy()
    metric[:, 0, 3] *= 4.
    k = np.array([[60., 0., size / 2], [0., 60., size / 2], [0., 0., 1.]])
    prediction = SimpleNamespace(depth=np.full((n, size, size), 2.),
                                 processed_images=np.zeros((n, size, size, 3), np.uint8),
                                 extrinsics=np.linalg.inv(local), intrinsics=np.repeat(k[None], n, axis=0))
    stamps = [int(i * 5e7) for i in range(n)]
    out = mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps], camera_poses=metric,
                          conditioned=True), prediction)
    assert out['status'] == 'rejected' and 'conditioned' in out['reason']
    out = mapper.add(dict(stamps=stamps, times=[s * 1e-9 for s in stamps], camera_poses=metric,
                          conditioned=False), prediction)
    assert out['status'] == 'mapped' and out['conditioned'] is False


def test_overlap_scale_prefers_the_pixel_aligned_shared_frame_ratio():
    from davio_mapper.online import OnlineMapper
    mapper = OnlineMapper(_backend_config(loops_enabled=False, scale_overlap_min_points=10), pathlib_tmp())
    older = dict(depth=np.full((5, 8, 8), 2.), frame_ids=['a', 'b', 'c', 'd', 'e'], valid=np.ones((5, 8, 8), bool))
    newer = dict(depth=np.full((5, 8, 8), 2.4), frame_ids=['d', 'e', 'f', 'g', 'h'], valid=np.ones((5, 8, 8), bool))
    ratio, sigma, n = mapper.shared_frame_ratio(older, newer)
    assert ratio == pytest.approx(1.2) and n == 2 and sigma >= mapper.cfg['scale_sigma_log']
    assert mapper.shared_frame_ratio(older, dict(newer, frame_ids=['x', 'y', 'z', 'w', 'v'])) == (None, None, 0)


def test_loop_moves_poses_only_where_odometry_drift_exceeds_its_own_error():
    from davio_mapper.online import OnlineMapper
    cfg = dict(loop_sigma_m=.25, loop_sigma_deg=5., loop_scale_relative_error=.10,
               odometry_sigma_floor_m=.02, odometry_drift_m_per_m=.02, loop_min_drift_ratio=1.)
    mapper = type('M', (), {'cfg': cfg, 'loop_sigmas': OnlineMapper.loop_sigmas, 'loop_useful': OnlineMapper.loop_useful})()
    near, drift, sigma = mapper.loop_useful(dict(path_m=0.), dict(path_m=5.), 1.7)
    far, drift_far, _ = mapper.loop_useful(dict(path_m=0.), dict(path_m=40.), 1.7)
    assert not near and far and drift == pytest.approx(.1) and drift_far == pytest.approx(.8)
    assert sigma == pytest.approx(np.hypot(.25, .17))
    cfg['loop_min_drift_ratio'] = 0.                                        # v1 behaviour
    assert mapper.loop_useful(dict(path_m=0.), dict(path_m=5.), 1.7)[0]
