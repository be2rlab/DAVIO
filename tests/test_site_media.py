"""Turning a finished run into the page's assets: what it refuses, and the facts it carries."""
import json

import pytest


def run_dir(tmp_path, **meta):
    run = tmp_path / 'run'
    run.mkdir(parents=True)
    base = dict(dataset='euroc', sequence='V1_01_easy', state='completed', vision_failed=False, poses=3, loops_closed=2,
                interval_start=0., interval_end=90.)
    (run / 'run.json').write_text(json.dumps({**base, **meta}))
    (run / 'map_trajectory_final.tum').write_text('0 0 0 0 0 0 0 1\n1 3 4 0 0 0 0 1\n2 3 4 12 0 0 0 1\n')
    return run


def test_a_run_that_stopped_early_or_lost_its_map_is_refused(script, tmp_path):
    media = script('site_media')
    with pytest.raises(SystemExit, match='failed'):
        media.check_run(run_dir(tmp_path, state='failed'))
    with pytest.raises(SystemExit, match='dense map'):
        media.check_run(run_dir(tmp_path / 'b', vision_failed=True))


def test_the_facts_are_the_runs_own_numbers(script, tmp_path):
    run = run_dir(tmp_path)
    (run / 'evaluation.json').write_text(json.dumps(dict(ate=dict(final=dict(position_m=.042)))))
    (run / 'surface.json').write_text(json.dumps(dict(thresholds={'0.05': dict(f1=.61)})))
    (run / 'map').mkdir()
    (run / 'map/map_index.json').write_text(json.dumps(dict(submaps={'0': {}, '1': {}})))
    facts = script('site_media').facts(run, dict(title='EuRoC V1_01', note='Vicon room'))
    assert facts == dict(title='EuRoC V1_01', note='Vicon room', duration_s=90, path_m=17.0, submaps=2, loops=2,
                         ate_final_m=.042, f1_5cm=.61)


def test_an_asset_over_the_ceiling_is_refused(script, tmp_path):
    big = tmp_path / 'work/x.rrd'
    big.parent.mkdir()
    big.write_bytes(b'0' * 2_000_000)
    script('site_media').publish(big, tmp_path / 'site/x.rrd', limit_mb=3)
    assert (tmp_path / 'site/x.rrd').stat().st_size == 2_000_000
    big.write_bytes(b'0' * 2_000_000)
    with pytest.raises(SystemExit, match='MB'):
        script('site_media').publish(big, tmp_path / 'site/y.rrd', limit_mb=1)
    assert not (tmp_path / 'site/y.rrd').exists()


def test_a_video_over_the_ceiling_is_encoded_again_at_lower_quality_until_it_fits(script, tmp_path, monkeypatch):
    import subprocess
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=30:duration=2',
                    '-c:v', 'libx264', '-crf', '10', str(source)], check=True)
    media = script('site_media')
    media.web(source, tmp_path / 'a.mp4', tmp_path / 'a.jpg', 1.)
    first = (tmp_path / 'a.mp4').stat().st_size / 1e6
    monkeypatch.setattr(media, 'VIDEO_LIMIT_MB', first * .8)
    media.web(source, tmp_path / 'b.mp4', tmp_path / 'b.jpg', 1.)
    assert (tmp_path / 'b.mp4').stat().st_size / 1e6 <= first * .8 and (tmp_path / 'b.jpg').is_file()
    monkeypatch.setattr(media, 'VIDEO_LIMIT_MB', first * .01)
    with pytest.raises(SystemExit, match='MB'):
        media.web(source, tmp_path / 'c.mp4', tmp_path / 'c.jpg', 1.)
    assert not (tmp_path / 'c.mp4').exists() and not (tmp_path / 'c.jpg').exists()      # nothing the page could take


def test_the_floor_of_a_run_is_the_whole_maps_under_its_cameras_not_a_reflection_below_it(script, tmp_path):
    import numpy as np
    from davio.mapper.fusion import write_ply
    run = run_dir(tmp_path)
    rng = np.random.default_rng(4)
    z = np.concatenate((rng.normal(-1.3, .01, 3000), rng.uniform(-1.3, 2., 3000), rng.normal(-4.6, .01, 6000)))
    points = np.column_stack((rng.uniform(0., 5., len(z)), rng.uniform(0., 5., len(z)), z))
    write_ply(run / 'map.ply', points, np.zeros((len(z), 3), np.uint8))
    (run / 'map_trajectory_final.tum').write_text(''.join(f'{i} 1 1 0 0 0 0 1\n' for i in range(5)))
    assert abs(script('site_media').map_floor(run) + 1.3) < .05
