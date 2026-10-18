from collections import deque
from types import SimpleNamespace

from davio.runtime.engine import Engine


def engine(preroll, times):
    history = deque((dict(t=t, image=None, imu=[]) for t in times), maxlen=160)
    return SimpleNamespace(history=history,
                           cfg=dict(assistance=dict(candidate_preroll_s=preroll)))


TIMES = [i * 0.05 for i in range(160)]         # 8 s of history at 20 Hz


def replay(preroll, bootstrap):
    return Engine.candidate_history(engine(preroll, TIMES), bootstrap)


def test_the_frozen_default_replays_everything():
    """null is the behaviour every archived run was produced with."""
    assert len(replay(None, dict(t=6.0))) == 160


def test_a_pre_roll_keeps_only_what_the_tracker_needs():
    kept = replay(1.0, dict(t=6.0))
    assert kept[0]['t'] >= 5.0 - 1e-9, 'nothing older than one second before the bootstrap'
    assert kept[-1]['t'] == TIMES[-1], 'and everything up to now'
    assert len(kept) == 60


def test_the_bootstrap_frame_itself_is_always_replayed():
    kept = replay(0.0, dict(t=6.0))
    assert any(abs(p['t'] - 6.0) < 1e-9 for p in kept), 'the state is injected on this frame'


def test_a_candidate_without_a_bootstrap_keeps_the_full_history():
    """A rotation/bias-only candidate has no state to start from; it initialises itself."""
    assert len(replay(1.0, None)) == 160


def test_a_bootstrap_newer_than_the_history_still_leaves_something_to_replay():
    kept = replay(1.0, dict(t=100.0))
    assert len(kept) == 1 and kept[0]['t'] == TIMES[-1]
