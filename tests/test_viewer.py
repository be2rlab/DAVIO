"""scripts/view_run.py: a run drawn in Rerun from the files it writes, while it is written or after."""
import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from davio.geometry import tum_line
from davio.mapper import sim3
from davio.mapper.fusion import world_points

rr = pytest.importorskip('rerun')

# Images are named by their stamp in nanoseconds, as text, as the mapper writes them; the two submaps share the one at 1.2 s.
IDS = {'000000': ['1000000000', '1100000000', '1200000000'], '000001': ['1200000000', '1300000000', '1400000000']}


def submap(n, size=6, colour=False):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * .1
    K = np.array([[5., 0., 3.], [0., 5., 3.], [0., 0., 1.]])
    rgb = np.full((n, size, size, 3), 90, np.uint8)
    if colour:
        rgb[..., 0] = 200
    return dict(depth=np.full((n, size, size), 2.), conf=np.ones((n, size, size)), intrinsics=np.tile(K, (n, 1, 1)),
                rgb=rgb, poses=poses)


def pose(i):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = Rotation.from_rotvec([0., 0., .1 * i]).as_matrix(), [.01 * i, 0., 0.]
    return out


def archive(run, key, colour=False):
    np.savez_compressed(run / f'map/submaps/{key}.npz', **submap(len(IDS[key]), colour=colour))


def write_index(run, nodes):
    index = {key: dict(file=f'submaps/{key}.npz', frame_ids=IDS[key], center=1, timestamp=int(IDS[key][-1]) * 1e-9,
                       T_map_submap=node.tolist()) for key, node in nodes.items()}
    path = run / 'map/map_index.json'
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(json.dumps(dict(submaps=index, last_sensor_time=2.45)))
    os.utime(path, ns=(before + 10 ** 9, before + 10 ** 9))       # a new version, however coarse the clock


def fake_run(run, state='completed', archives=True, colour=False):
    """What the engine and the mapper write: poses, two overlapping submaps and a loop between them."""
    (run / 'map/submaps').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps(dict(sequence='seq', state=state, vision_failed=False)))
    for name in ('trajectory', 'camera_trajectory', 'map_trajectory'):
        (run / f'{name}.tum').write_text(''.join(tum_line(1. + .05 * i, pose(i)) for i in range(30)))
    nodes = {'000000': sim3.exp(np.array([0., 0., 0., 0., 0., 0., np.log(1.2)])),
             '000001': sim3.exp(np.array([.5, .1, 0., 0., 0., .2, np.log(.9)]))}
    for key in nodes if archives else ():
        archive(run, key, colour)
    write_index(run, nodes)
    events = [dict(kind='window', wall=10., arrival_wall=9.5, sensor_time=1.2, report=dict(status='mapped', submap='000000', scale=1.2)),
              dict(kind='window', wall=11., arrival_wall=10.6, sensor_time=1.4,
                   report=dict(status='mapped', submap='000001', scale=.9, accepted_loops=1, loops=[dict(a='000000', b='000001')])),
              dict(kind='correction', wall=11., t=1.45, submap='000001', jump_m=.02, jump_deg=.1, ramped=False)]
    (run / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    return nodes


@pytest.fixture
def viewer(script, tmp_path):
    rr.init('davio-test')
    rr.memory_recording()
    return lambda **kw: script('view_run').Viewer(rr, tmp_path / 'run', pixel_step=1, **kw)


def test_the_points_of_an_image_land_where_the_exported_map_puts_them(script):
    # The scale of a node multiplies depth only: on the submap's transform it would move every image
    # off the centre by (s - 1) times its baseline.
    view, sm = script('view_run'), submap(3)
    node = sim3.exp(np.array([.4, -.3, .2, .05, -.02, .03, np.log(1.15)]))
    placed = view.placement(node)
    for i in range(3):
        local, rgb = view.image_points(sm, i, node, pixel_step=1)
        expected, colors = world_points(sm, i, node, pixel_step=1)
        np.testing.assert_allclose(local @ placed[:3, :3].T + placed[:3, 3], expected, atol=1e-9)
        assert (rgb == colors).all()


@pytest.mark.parametrize('state, label', [(None, 'AWAITING'), ('running', 'LIVE'), ('completed', 'RECORDED'), ('failed', 'FAILED')])
def test_what_is_shown_is_labelled_by_the_run_s_own_state(script, tmp_path, state, label):
    if state:
        (tmp_path / 'run.json').write_text(json.dumps(dict(sequence='seq', state=state)))
    assert script('view_run').provenance(tmp_path)[0] == label


def test_a_run_whose_mapping_process_failed_says_there_is_no_map(script, tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps(dict(sequence='seq', state='completed', vision_failed=True)))
    assert 'no dense map' in script('view_run').provenance(tmp_path)[1]
    assert 'no dense map' in script('view_run').run_card(tmp_path)


def test_a_line_still_being_written_waits_for_its_end(script, tmp_path):
    path = tmp_path / 'trajectory.tum'
    tail = script('view_run').Tail(path)
    assert tail.lines() == []
    path.write_text('1 2\n3 ')
    assert tail.lines() == ['1 2']
    with path.open('a') as f:
        f.write('4\n')
    assert tail.lines() == ['3 4']


def test_a_run_is_drawn_with_every_pose_each_image_once_and_its_loop(viewer, tmp_path):
    fake_run(tmp_path / 'run')
    v = viewer()
    v.update()
    assert v.poses == 30 and set(v.drawn) == {'000000', '000001'} and not v.waiting
    assert len(v.owner) == 5 and v.points == 5 * 36         # the image both submaps hold is drawn once
    assert v.loops == {('000000', '000001')}


def test_one_image_per_submap_is_drawn_when_asked(viewer, tmp_path):
    fake_run(tmp_path / 'run')
    v = viewer(frames='centre')
    v.update()
    assert v.points == 2 * 36


def test_a_submap_rescaled_by_the_graph_is_redrawn_and_one_only_moved_is_not(viewer, tmp_path):
    nodes = fake_run(tmp_path / 'run')
    v = viewer()
    v.update()
    first = dict(v.drawn)
    write_index(tmp_path / 'run', {'000000': nodes['000000'] @ sim3.exp(np.array([.3, 0., 0., 0., 0., 0., 0.])),
                                   '000001': nodes['000001'] @ sim3.exp(np.array([0., 0., 0., 0., 0., 0., np.log(1.05)]))})
    v.update()
    assert v.drawn['000000'] == first['000000'] and v.drawn['000001'] == pytest.approx(first['000001'] * 1.05)


def test_a_submap_listed_before_its_archive_is_on_disk_is_drawn_once_it_is(viewer, tmp_path):
    run = tmp_path / 'run'
    fake_run(run, state='running', archives=False)
    v = viewer()
    v.update()
    assert v.drawn == {} and v.waiting == {'000000', '000001'}
    (run / 'map/submaps/000000.npz').write_bytes(b'PK\x03\x04 half written')
    v.update()
    assert v.drawn == {} and v.waiting == {'000000', '000001'}
    for key in IDS:
        archive(run, key)
    v.update()                  # the index has not changed: the archives alone
    assert set(v.drawn) == {'000000', '000001'} and not v.waiting


def test_a_run_is_written_to_a_file_to_open_later(script, tmp_path):
    fake_run(tmp_path / 'run')
    out = tmp_path / 'run.rrd'
    assert script('view_run').main(['--run', str(tmp_path / 'run'), '--save', str(out)]) == 0
    assert out.stat().st_size > 1000


def test_a_directory_that_is_not_there_is_refused_unless_followed(script, tmp_path):
    with pytest.raises(SystemExit):
        script('view_run').main(['--run', str(tmp_path / 'nothing'), '--save', str(tmp_path / 'x.rrd')])


def test_the_address_printed_names_the_server_in_a_form_the_page_reads(script):
    # Left as it is, the '+' of rerun+http reads as a space and the page shows Rerun's welcome screen instead of the run.
    url = script('view_run').viewer_url(9090, 'rerun+http://127.0.0.1:9876/proxy')
    assert url == 'http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy'


def test_a_height_colour_depends_on_the_point_alone_and_its_shade_on_its_pixel(script):
    view = script('view_run')
    z, grey = np.array([0., 1., 2.5, 1.]), np.array([200., 200., 200., 60.])
    colors = view.height_colors(z, 0., grey)
    assert colors.dtype == np.uint8 and colors.shape == (4, 3)
    assert len({tuple(c) for c in colors[:3]}) == 3             # height reads as colour
    assert (colors[3] < colors[1]).all()                          # a darker pixel, a darker point
    # Whatever else is drawn beside it: a growing map does not recolour what is already on screen.
    np.testing.assert_array_equal(view.height_colors(z[1:2], 0., grey[1:2]), colors[1:2])


def test_a_grey_camera_is_told_from_a_colour_one(script):
    view, rgb = script('view_run'), np.full((2, 4, 4, 3), 90, np.uint8)
    rgb[..., 1] += 1                            # the depth model's resampling of a grey image
    assert view.is_grey(rgb)
    rgb[0, 1, 1] = (200, 40, 40)
    assert not view.is_grey(rgb)


@pytest.mark.parametrize('color, colour_camera, mode', [('auto', False, 'height'), ('auto', True, 'rgb'),
                                                        ('rgb', False, 'rgb'), ('height', True, 'height')])
def test_the_map_is_coloured_by_height_unless_the_camera_sees_colour(viewer, tmp_path, color, colour_camera, mode):
    fake_run(tmp_path / 'run', colour=colour_camera)
    v = viewer(color=color)
    v.update()
    assert v.mode == mode
    assert v.floor == pytest.approx(2.4)        # the first submap's images, depth 2 at scale 1.2, all at one height


def test_the_run_card_says_how_far_the_run_is_and_how_good_once_it_is_scored(script, tmp_path):
    view = script('view_run')
    assert 'AWAITING' in view.run_card(tmp_path)
    (tmp_path / 'run.json').write_text(json.dumps(dict(
        sequence='V1_01_easy', state='running', rate=1., interval_start=10., interval_end=110., last_sensor_time=40., poses=600,
        windows=dict(submitted=12, mapped=10, dropped=2, deferred=0, rejected=0), loops_closed=1, pose_age_p50_s=.016, pose_age_p95_s=.022)))
    card = view.run_card(tmp_path)
    assert all(text in card for text in ('LIVE', 'V1_01_easy', '30 s of 100 s', '10 of 12', '2 dropped', '16 ms'))
    assert 'ATE' not in card
    (tmp_path / 'evaluation.json').write_text(json.dumps(dict(ate=dict(
        odometry=dict(position_m=.0402), live=dict(position_m=.0401), final=dict(position_m=.0421)))))
    (tmp_path / 'surface.json').write_text(json.dumps(dict(thresholds={'0.05': dict(f1=.5354)}, chamfer_mean_m=.0761)))
    card = view.run_card(tmp_path)
    assert all(text in card for text in ('ATE', '4.0 cm', '4.2 cm', '0.54', '7.6 cm'))


def test_the_floor_grid_lies_under_the_map_and_only_grows(viewer, tmp_path):
    # Rerun frames the view on everything drawn: a grid much larger than the map would shrink the map.
    nodes = fake_run(tmp_path / 'run')
    v = viewer()
    v.update()
    centres = np.array([node[:2, 3] for node in nodes.values()])
    lo, hi = v.grid
    assert (lo <= centres.min(axis=0) - 3).all() and (lo > centres.min(axis=0) - 4).all()
    assert (hi >= centres.max(axis=0) + 3).all() and (hi < centres.max(axis=0) + 4).all()
    write_index(tmp_path / 'run', {'000000': nodes['000000'] @ sim3.exp(np.array([-5., 0., 0., 0., 0., 0., 0.])), '000001': nodes['000001']})
    v.update()
    assert v.grid[0][0] <= -5 * 1.2 - 3 and (v.grid[1] == hi).all()
