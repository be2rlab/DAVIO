import numpy as np
import pytest

from davio.filter import Filter, materialize, read_camera, read_yaml


def packet(t):
    return dict(t=t, image=np.zeros((4, 4), np.uint8), imu=[(t - .005, np.zeros(3), np.zeros(3)), (t + .001, np.zeros(3), np.zeros(3))])


@pytest.fixture
def rig(root):
    return root / 'config/euroc/estimator_config.yaml'


def test_the_filter_gives_no_state_until_it_is_initialized_then_the_state_of_the_frame(openvins):
    filter = Filter('estimator.yaml')
    assert filter.vio.config_path == 'estimator.yaml'
    assert filter.feed(packet(1.)) is None and filter.feed(packet(1.05)) is None
    state = filter.feed(packet(1.1))
    assert state['t'] == 1.1 and state['p'][0] == pytest.approx(.3)
    assert filter.vio.imu == pytest.approx([.995, 1.001, 1.045, 1.051, 1.095, 1.101])
    openvins.silent_after = 3
    assert filter.feed(packet(1.15)) is None            # the state of an earlier frame is not an update


def test_a_run_gets_its_own_copy_of_the_rig_with_the_settings_written_over_it(rig, tmp_path, cfg):
    estimator = materialize(rig, tmp_path / 'filter', cfg['openvins'])
    assert estimator == tmp_path / 'filter/estimator.yaml' and estimator.read_text().startswith('%YAML:1.0\n')
    written, source = read_yaml(estimator), read_yaml(rig)
    assert source['calib_cam_intrinsics'] is True and written['calib_cam_intrinsics'] is False
    assert written['init_dyn_bias_g'] == [0., 0., 0.] and written['verbosity'] == 'WARNING'
    assert written['max_clones'] == source['max_clones']
    assert not any(written[key] for key in ('record_timing_information', 'save_total_state', 'record_init_pose'))
    for key in ('relative_config_imu', 'relative_config_imucam'):
        assert written[key] == key + '.yaml'
        assert read_yaml(tmp_path / 'filter' / written[key]) == read_yaml(rig.parent / source[key])
    assert read_camera(estimator) == read_camera(rig) and read_camera(rig)['resolution'] == [752, 480]
    with pytest.raises(FileExistsError):                # a run's directory is never written twice
        materialize(rig, tmp_path / 'filter', {})
