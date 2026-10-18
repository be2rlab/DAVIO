import numpy as np
from davio_mapper.graph import Graph, PointFactor
from davio_mapper import sim3


def test_sparse_factor_jacobian_and_metric_recovery():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(60, 3)) + [0, 0, 3]
    true = sim3.exp(np.array([.2, -.1, .05, .03, -.04, .02, .12]))
    target = (points - true[:3, 3]) @ np.linalg.inv(true[:3, :3]).T
    nodes = {'a': np.eye(4), 'b': true.copy()}
    # Isotropic and the anisotropic ray-aligned model the configuration ships.
    for uncertainty in ({}, dict(sigma_radial=.10, sigma_lateral=.01)):
        factor = PointFactor('a', 'b', points, target, **uncertainty)
        _, residual, blocks = factor.linearize(nodes)
        for key in nodes:
            for k in range(7):
                step = np.zeros(7); step[k] = 1e-7
                shifted = dict(nodes); shifted[key] = nodes[key] @ sim3.exp(step)
                numeric = (factor.linearize(shifted, False)[1] - residual) / step[k]
                np.testing.assert_allclose(numeric, blocks[key][:, k], atol=2e-6)
        graph = Graph()
        graph.add_node('a', np.eye(4), dimensions=0)
        graph.add_node('b', np.eye(4))
        graph.add(factor)
        report = graph.optimize()
        assert report['final_cost'] < 1e-10
        np.testing.assert_allclose(graph.nodes['b'], true, atol=1e-5)


def test_loop_sigma_grows_with_the_loop_baseline():
    from davio_mapper.online import OnlineMapper
    cfg = dict(loop_sigma_m=.15, loop_sigma_deg=3., loop_scale_relative_error=.10)
    sigmas = OnlineMapper.loop_sigmas.__get__(type('C', (), {'cfg': cfg})())
    near, far = sigmas(0.), sigmas(3.)
    np.testing.assert_allclose(near[:3], .15)
    np.testing.assert_allclose(far[:3], np.hypot(.15, .3))
    np.testing.assert_allclose(near[3:], far[3:])   # rotation does not ride on depth scale


def test_map_correction_uses_optimized_node(tmp_path, monkeypatch):
    from test_system import _backend_config, _keyframe
    from davio_mapper.online import OnlineMapper
    cfg = _backend_config(loops_enabled=False)
    mapper = OnlineMapper(cfg, tmp_path)
    rgb = np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    def solve():
        for node in mapper.nodes.values():
            node['pose'] = node['pose'].copy()
            node['pose'][0, 3] += .3
        return {}
    monkeypatch.setattr(mapper, 'optimize', solve)
    monkeypatch.setattr(mapper, 'overlap_scale', lambda *args: 1.)
    for offset in (0., 1.):
        centres = np.array([[offset + i*.1, 0, 0] for i in range(5)])
        metric, pred = _keyframe(rgb, np.full((128,128), 3.), centres, 1.)
        task = dict(camera_poses=metric, stamps=list(range(5)), times=list(range(5)))
        result = mapper.add(task, pred)
    np.testing.assert_allclose(result['T_map_odom'][0][3], .3)


def test_multiview_filter_rejects_unsupported_floating_depth():
    from davio_mapper.filtering import filter_depth
    z = np.ones((2, 32, 32), np.float32) * 3
    z[0, 10:20, 10:20] = 7
    sm = dict(depth=z, poses=np.repeat(np.eye(4)[None],2,axis=0),
              intrinsics=np.repeat(np.array([[30.,0,16],[0,30.,16],[0,0,1]])[None],2,axis=0))
    retained = filter_depth(sm, {})
    assert 0 < retained < 1
    assert sm['valid'][0, 5, 5]
    assert not sm['valid'][0, 15, 15]


def test_final_trajectory_uses_final_corrections_and_keeps_raw(tmp_path):
    import json
    from export_trajectory import final_trajectory
    (tmp_path/'map').mkdir()
    entries={}
    for i in range(2):
        odom=np.eye(4);odom[0,3]=i
        mapped=odom.copy();mapped[1,3]=i*2
        entries[str(i)]=dict(timestamp=float(i), T_odom_submap=odom.tolist(), T_map_submap=mapped.tolist())
    (tmp_path/'map/map_index.json').write_text(json.dumps(dict(submaps=entries)))
    raw='0.0 0 0 0 0 0 0 1\n0.5 0.5 0 0 0 0 0 1\n1.0 1 0 0 0 0 0 1\n'
    (tmp_path/'trajectory.tum').write_text(raw)
    out=list(final_trajectory(tmp_path))
    np.testing.assert_allclose(out[1][1][:3,3],[.5,1,0])
    assert (tmp_path/'trajectory.tum').read_text()==raw


def test_selected_estimator_backpressure_drains_before_retry(tmp_path, monkeypatch):
    import sys, types, time, yaml
    from test_system import ROOT, RIG, FakeVio
    from davio.runtime.engine import Engine
    monkeypatch.setitem(sys.modules,'openvins_ext',types.SimpleNamespace(VioManager=FakeVio))
    cfg=yaml.safe_load((ROOT/'config/system.yaml').read_text())
    cfg['seed']=0;cfg['assistance']['enabled']=False;cfg['mapping']['enabled']=False
    cfg['shutdown_drain_s']=0
    engine=Engine(cfg,RIG,tmp_path/'run')
    engine.native.close();engine.native=None;engine.selected='assisted'
    fake=FakeVio('');fake.t=1.;state=fake.get_state()
    calls=[]
    def submit(packet):
        calls.append('submit')
        return calls.count('submit')>1
    def drain():
        calls.append('drain')
        return [dict(kind='state',state=state,warm_frames=5)]
    engine.shadow=types.SimpleNamespace(submit=submit,drain=drain,close=lambda:None,
                                       process=types.SimpleNamespace(is_alive=lambda:True))
    engine.step(dict(t=1., image=np.zeros((480,752),np.uint8),
                     imu=[(1.001,np.zeros(3),np.zeros(3))],arrival_wall=time.monotonic()))
    engine.close()
    assert calls[:3]==['submit','drain','submit']
    assert engine.count==1 and engine.status=='completed'


def test_sparse_nonzero_residual_cost_gradient_has_fixed_weights():
    points = np.array([[.2, .3, 2.], [1., -.2, 4.]])
    nodes = {'a': np.eye(4), 'b': sim3.exp(
        np.array([.05, .02, 0., .1, .03, 0., .1]))}
    for huber in (100., .2):
        factor = PointFactor('a', 'b', points, points + .03,
                             sigma_radial=.1, sigma_lateral=.01, huber_delta=huber)
        initial, residual, blocks = factor.linearize(nodes)
        reference = factor._reference_whitener.copy()
        for key in nodes:
            for k in range(7):
                step = np.zeros(7)
                step[k] = 1e-6
                plus, minus = dict(nodes), dict(nodes)
                plus[key] = nodes[key] @ sim3.exp(step)
                minus[key] = nodes[key] @ sim3.exp(-step)
                numeric = (factor.linearize(plus, False)[0]
                           - factor.linearize(minus, False)[0]) / (2e-6)
                np.testing.assert_allclose(numeric, blocks[key][:, k] @ residual,
                                           atol=2e-7, rtol=2e-6)
        np.testing.assert_array_equal(reference, factor._reference_whitener)
        assert factor.linearize(nodes, False)[0] == initial
