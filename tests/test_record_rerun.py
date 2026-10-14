"""The Rerun video: its layout and the schedule it is recorded on; Chrome is not needed."""
import numpy as np
import pytest

rr = pytest.importorskip('rerun')


def test_the_video_layout_is_the_map_beside_the_camera_and_its_depth(script):
    import rerun.blueprint as rrb
    view = script('view_run')
    for layout in ('inspect', 'video'):
        assert isinstance(view.blueprint(rrb, view.PALETTE['dark'], layout), rrb.Blueprint)
    with pytest.raises(ValueError):
        view.blueprint(rrb, view.PALETTE['dark'], 'poster')


def test_frames_cover_the_recording_evenly_then_hold_on_its_end(script):
    times = script('record_rerun').frame_times(100., 200., total=11, hold=3)
    assert times[0] == 100. and times[10] == 200. and times[5] == 150. and times[11:] == [200.] * 3


def test_a_uniform_frame_is_blank_and_a_rendered_one_is_not(script):
    blank = script('record_rerun').blank
    assert blank(np.full((90, 160, 3), 13, np.uint8))
    frame = np.full((90, 160, 3), 13, np.uint8)
    frame[30:60, 40:120] = (0, 210, 255)
    assert not blank(frame)


def test_a_light_recording_keeps_the_camera_as_jpeg_and_the_depth_at_half_size(script):
    view = script('view_run')
    rgb = np.random.default_rng(0).integers(0, 255, (48, 64, 3), np.uint8)
    assert isinstance(view.camera_image(rr, rgb, light=True), rr.EncodedImage)
    assert isinstance(view.camera_image(rr, rgb, light=False), rr.Image)
    depth = np.ones((48, 64), np.float32)
    assert view.depth_for_viewer(depth, light=True).shape == (24, 32) and view.depth_for_viewer(depth, False).shape == (48, 64)


def test_a_portrait_phones_frames_are_stood_up_for_display_only(script):
    view = script('view_run')
    image = np.zeros((480, 640, 3), np.uint8)
    image[0, 0] = 255                                         # top left of the sideways frame
    upright = view.upright(image, 90)                         # a quarter turn clockwise
    assert upright.shape == (640, 480, 3) and upright[0, -1, 0] == 255
    assert view.upright(image, 0) is image
    assert view.upright(np.zeros((48, 64), np.float32), 270).shape == (64, 48)


@pytest.mark.parametrize('cuts, drawn', [(dict(), 5 * 36), (dict(floor_cut=100., ceiling_cut=100.), 5 * 36),
                                         (dict(floor_cut=-100.), 0), (dict(ceiling_cut=-100.), 0)])
def test_what_lies_beyond_the_floor_and_ceiling_cuts_is_not_drawn(script, tmp_path, cuts, drawn):
    from test_viewer import fake_run
    rr.init('davio-test')
    rr.memory_recording()
    fake_run(tmp_path / 'run')
    v = script('view_run').Viewer(rr, tmp_path / 'run', pixel_step=1, **cuts)
    v.update()
    assert v.floor is not None and v.points == drawn


def test_a_floor_given_is_the_floor_drawn_on(script, tmp_path):
    from test_viewer import fake_run
    rr.init('davio-test')
    rr.memory_recording()
    fake_run(tmp_path / 'run')
    v = script('view_run').Viewer(rr, tmp_path / 'run', pixel_step=1, floor=-5.)
    v.update()
    assert v.floor == -5. and v.grid is not None
