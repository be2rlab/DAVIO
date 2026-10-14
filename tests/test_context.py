import numpy as np

from davio import context


def pose(x=0., turn_deg=0.):
    """A camera at x that looks along z turned about its y axis."""
    a = np.radians(turn_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    T[0, 3] = x
    return T


def ns(seconds):
    return int(seconds * 1e9)


def select(poses, first_s, mapping, **settings):
    return context.select({ns(t): p for t, p in poses.items()}, ns(first_s), pose(), dict(mapping, **settings))


def test_a_context_keyframe_is_old_enough_in_range_and_looks_the_same_way(mapping):
    poses = {1.: pose(1.),              # the one that qualifies
             9.: pose(1.),              # taken less than 2 s before the window
             2.: pose(.1),              # too close to the centre camera
             3.: pose(5.),              # too far
             4.: pose(1.5, 45.)}        # looks elsewhere
    assert select(poses, 10., mapping) == [ns(1.)]


def test_keyframes_older_than_the_limit_are_left_out(mapping):
    poses = {1.: pose(1., 5.), 60.: pose(-1., 10.)}
    assert select(poses, 70., mapping) == [ns(60.)]
    assert select(poses, 70., mapping, context_max_age_s=100.) == [ns(1.), ns(60.)]


def test_the_most_similar_directions_come_first_and_no_two_from_one_place(mapping):
    poses = {1.: pose(1., 20.), 2.: pose(2., 5.),
             3.: pose(2.1, 10.),        # 10 cm from the one before: skipped
             4.: pose(-1., 15.)}
    assert select(poses, 10., mapping) == [ns(2.), ns(4.)]
    assert select(poses, 10., mapping, context_frames=5) == [ns(2.), ns(4.), ns(1.)]
    assert select(poses, 10., mapping, context_frames=0) == []
    assert select({}, 10., mapping) == []
