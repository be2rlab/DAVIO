"""The Open3D renders' schedule and colouring; Open3D itself is not needed."""
import numpy as np


def test_every_panel_is_on_the_master_clock_and_holds_its_own_ending(script):
    grid = script('compose_grid')
    master = dict(t0=10., t1=20., fps=30, total_frames=101, hold_frames=30)
    short = dict(t0=10., t1=20., fps=30, total_frames=51, hold_frames=60)
    assert grid.sensor_time(0, master) == 10. and grid.sensor_time(100, master) == 20.
    assert grid.sensor_time(130, master) == 20.                         # held on the finished map
    assert grid.panel_frame(50, master, short) == 25                    # the same sensor time in a shorter render
    assert grid.panel_frame(101, master, short) == 51                   # then its own hold, frame by frame
    assert grid.panel_frame(130, master, short) == 80
    assert grid.panel_frame(10_000, master, short) == 110               # never past its last frame


def test_the_crane_leaves_the_walk_and_lands_on_the_three_quarter_shot(script):
    build = script('render_build')
    start = (np.array([0., 0., 2.]), np.array([1., 0., 1.]))
    end = (np.array([10., 10., 10.]), np.array([5., 5., 0.]))
    for f, expected in ((0., start), (1., end)):
        eye, target = build.crane(start, end, f)
        np.testing.assert_allclose(eye, expected[0])
        np.testing.assert_allclose(target, expected[1])
    eye, _ = build.crane(start, end, .5)
    np.testing.assert_allclose(eye, [5., 5., 6.])                       # eased, symmetric about the middle


def test_a_grey_map_is_coloured_by_height_and_a_colour_one_is_left_alone(script):
    build = script('render_build')
    grey = np.full((4, 3), 128, np.uint8)
    z = np.array([0., 1., 2., 3.])
    coloured = build.colour_points(np.column_stack((np.zeros(4), np.zeros(4), z)), grey, floor=0., grey_map=True)
    assert coloured.shape == (4, 3) and not (coloured[0] == coloured[-1]).all()
    rgb = np.array([[200, 10, 10]] * 4, np.uint8)
    assert (build.colour_points(np.zeros((4, 3)), rgb, floor=0., grey_map=False) == rgb).all()


def test_the_floor_is_the_densest_layer_not_what_reflections_put_below_it(script):
    view = script('view_run')
    rng = np.random.default_rng(1)
    floor = rng.normal(-1.3, .01, 4000)
    walls = rng.uniform(-1.3, 2., 6000)
    reflections = rng.uniform(-6.5, -2., 900)               # a glossy floor, seen as a room below it
    z = np.concatenate((floor, walls, reflections))
    assert np.percentile(z, 2) < -4.                         # what the percentile took for the floor
    assert abs(view.floor_height(z) + 1.3) < .05
    assert abs(view.floor_height(np.concatenate((rng.normal(0., .01, 3000), rng.uniform(0., 2.5, 5000)))) - 0.) < .05


def test_a_render_keeps_what_lies_between_the_floor_cut_and_the_ceiling_cut(script):
    build = script('render_build')
    z = np.array([-3., -1.3, 0., 2.9])
    assert build.between(z, low=-1.6, high=2.5).tolist() == [1, 2]
    assert build.between(z, low=None, high=None).tolist() == [0, 1, 2, 3]


def test_a_height_cut_follows_the_space_the_map_fills_not_how_densely_it_was_sampled(script):
    build = script('render_build')
    rng = np.random.default_rng(2)
    walls = np.column_stack((rng.uniform(0., 4., 3000), np.zeros(3000), rng.uniform(0., 2., 3000)))
    ceiling = np.column_stack((rng.uniform(0., .5, 30000), rng.uniform(0., .5, 30000), np.full(30000, 2.)))   # seen close
    points = np.concatenate((walls, ceiling))
    assert np.quantile(points[:, 2], .85) == 2.                  # the per-point quantile sits in the ceiling
    assert build.level(points, .85) < 1.9                       # the occupied space's is below it
    sparse = np.concatenate((walls, ceiling[::30]))
    assert abs(build.level(points, .5) - build.level(sparse, .5)) < .1


def test_a_mirrored_ceiling_below_a_glossy_floor_is_not_the_floor_when_the_camera_height_is_known(script):
    view = script('view_run')
    rng = np.random.default_rng(3)
    floor = rng.normal(-1.3, .01, 3000)
    walls = rng.uniform(-1.3, 2., 3000)
    mirrored_ceiling = rng.normal(-4.6, .01, 6000)          # the ceiling, 3.3 m up, seen in the floor
    z = np.concatenate((floor, walls, mirrored_ceiling))
    assert abs(view.floor_height(z) + 4.6) < .05             # densest below the median: the reflection
    assert abs(view.floor_height(z, camera_z=0.) + 1.3) < .05
    assert abs(view.floor_height(np.concatenate((rng.normal(-.1, .01, 2000), rng.uniform(-.1, 3., 4000))),
                                 camera_z=0.) + .1) < .05    # a drone that starts on the ground


def test_a_map_sorted_by_height_keeps_each_points_colour_and_a_prefix_is_everything_below_a_height(script):
    build = script('render_build')
    points = np.array([[0., 0., 3.], [1., 0., -1.], [2., 0., 1.], [3., 0., 0.]])
    colours = np.array([[.3, 0., 0.], [-.1, 0., 0.], [.1, 0., 0.], [0., 0., 0.]])     # red channel = z / 10
    z, sorted_points, sorted_colours = build.by_height(points, colours)
    assert z.tolist() == [-1., 0., 1., 3.] and (sorted_points[:, 2] == z).all()
    np.testing.assert_allclose(sorted_colours[:, 0], z / 10)
    assert sorted_points[:np.searchsorted(z, 1., side='right'), 0].tolist() == [1., 3., 2.]
