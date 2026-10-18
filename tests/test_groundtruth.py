"""The reference a score is computed against is part of the score."""
import numpy as np
import pytest

from davio.data import groundtruth as gt


def _write(path, rows, header='#t,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz'):
    path.write_text(header + '\n' + '\n'.join(','.join(str(v) for v in r) for r in rows) + '\n')
    return path


def _sample(t_ns):
    return [t_ns, 1., 2., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]


def test_reads_both_reference_layouts_identically(tmp_path):
    rows = [_sample(1000000000), _sample(1050000000)]
    asl = _write(tmp_path / 'asl.csv', rows, '#timestamp, p_RS_R_x [m], ...')
    ov = _write(tmp_path / 'ov.csv', rows)
    for path in (asl, ov):
        t, p, q, v, bg, ba = gt.read_asl_state_csv(path)
        np.testing.assert_allclose(t, [1.0, 1.05])
        np.testing.assert_allclose(p[0], [1., 2., 3.])
        np.testing.assert_allclose(q[0], [1., 0., 0., 0.])
    assert gt.sha256(asl) == gt.sha256(ov) or asl.read_bytes() != ov.read_bytes()


def test_variant_resolution_and_documented_suspect_flag(tmp_path):
    mav0 = tmp_path / 'V1_01_easy' / 'mav0' / 'state_groundtruth_estimate0'
    mav0.mkdir(parents=True)
    _write(mav0 / 'data.csv', [_sample(1000000000)])
    repo = tmp_path / 'repo'
    ovdir = repo / 'thirdparty/open_vins/ov_data/euroc_mav'
    ovdir.mkdir(parents=True)
    _write(ovdir / 'V1_01_easy.csv', [_sample(1000000000)])

    path, variant, reliable = gt.resolve('euroc', 'V1_01_easy', mav0.parent, repo, 'dataset')
    assert variant == 'dataset' and not reliable      # documented inaccurate orientation
    path, variant, reliable = gt.resolve('euroc', 'V1_01_easy', mav0.parent, repo, 'openvins')
    assert variant == 'openvins' and reliable and path.name == 'V1_01_easy.csv'
    _, variant, _ = gt.resolve('euroc', 'V1_01_easy', mav0.parent, repo, 'auto')
    assert variant == 'openvins'                      # auto repairs a suspect sequence
    _, variant, reliable = gt.resolve('euroc', 'V1_02_medium', mav0.parent, repo, 'auto')
    assert variant == 'dataset' and reliable          # nothing to repair, no corrected file
    with pytest.raises(FileNotFoundError):
        gt.resolve('euroc', 'V1_02_medium', mav0.parent, repo, 'openvins')
    with pytest.raises(ValueError):
        gt.resolve('euroc', 'V1_01_easy', mav0.parent, repo, 'whatever')


def test_provenance_identifies_the_file(tmp_path):
    path = _write(tmp_path / 'a.csv', [_sample(0), _sample(50000000)])
    record = gt.provenance(path, 'dataset', False, [0., .05])
    assert record['sha256'] == gt.sha256(path) and record['samples'] == 2
    assert record['rate_hz'] == pytest.approx(20.)
    assert record['orientation_reliable'] is False
