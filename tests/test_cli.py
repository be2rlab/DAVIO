import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    # The entry point has no .py suffix, so the loader has to be named explicitly.
    spec = importlib.util.spec_from_loader(
        'davio_cli', importlib.machinery.SourceFileLoader('davio_cli', str(ROOT / 'davio')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, 'containerized',
                        lambda argv, camera=False: calls.append(argv) or 0)
    monkeypatch.setattr(module, 'on_host', lambda argv: calls.append(argv) or 0)
    module.calls = calls
    return module


def test_offline_mapping_replays_the_runs_own_submaps_and_keeps_them(cli):
    assert cli.main(['map', 'runs/V1_01_easy']) == 0
    assert cli.calls[0] == ['python3', 'scripts/replay_map.py',
                            '--source', 'runs/V1_01_easy', '--out', 'runs/V1_01_easy_remap',
                            '--keep-dense']


def test_offline_mapping_can_drop_the_dense_result_for_an_ablation(cli):
    cli.main(['map', 'runs/r01', '--out', 'runs/r01_noloops', '--no-keep-dense',
              '--set', 'loops_enabled=false'])
    assert '--keep-dense' not in cli.calls[0] and '--no-keep-dense' not in cli.calls[0]
    assert cli.calls[0][-2:] == ['--set', 'loops_enabled=false']


def test_export_names_the_map_directory_not_the_run(cli):
    cli.main(['export', 'runs/r01/'])
    assert cli.calls[0] == ['python3', 'scripts/export_map.py',
                            '--map', 'runs/r01/map', '--out', 'runs/r01/map.ply']


def test_render_refuses_before_a_map_is_exported(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/r01').mkdir(parents=True)
    assert cli.main(['render', 'runs/r01']) == 2
    assert cli.calls == []


def test_render_runs_on_the_host_because_the_image_has_no_graphics(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/r01').mkdir(parents=True)
    (tmp_path / 'runs/r01/map.ply').write_text('ply')
    assert cli.main(['render', 'runs/r01', '--cut-ceiling', '0.9']) == 0
    assert cli.calls[0][1:] == ['scripts/render_map.py', '--run', 'runs/r01',
                                '--out', 'renders/r01', '--cut-ceiling', '0.9']


def test_build_video_needs_no_exported_cloud_because_it_reads_the_archives(cli, tmp_path,
                                                                          monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/r01').mkdir(parents=True)          # no map.ply
    assert cli.main(['render', 'runs/r01', '--build', 'fly', '--duration', '30']) == 0
    assert cli.calls[0][1:] == ['scripts/render_build.py', '--run', 'runs/r01',
                                '--out', 'renders/r01', '--camera', 'fly',
                                '--duration', '30']


def test_compose_hands_the_video_and_its_run_to_the_compositor(cli):
    cli.main(['compose', 'renders/x/map_build_fly.mp4', '--run', 'runs/beach_sky',
              '--rotate', '90'])
    assert cli.calls[0][1:] == ['scripts/compose_video.py',
                                '--video', 'renders/x/map_build_fly.mp4',
                                '--run', 'runs/beach_sky', '--rotate', '90']


def test_grid_passes_every_panel_through_in_order(cli):
    cli.main(['grid', '--run', 'runs/r01', '--panel', 'camera', '--panel', 'fly=a.mp4',
              '--out', 'g.mp4'])
    assert cli.calls[0][1:] == ['scripts/compose_grid.py', '--run', 'runs/r01',
                                '--panel', 'camera', '--panel', 'fly=a.mp4', '--out', 'g.mp4']


def test_run_scores_the_run_against_the_same_dataset(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    cli.main(['run', 'r01', '--dataset', 'ori'])
    assert cli.calls[0][:8] == ['python3', 'scripts/run_davio.py', '--dataset', 'ori',
                                '--data', 'data/ori', '--sequence', 'r01']
    assert cli.calls[1] == ['python3', 'scripts/evaluate_run.py',
                            '--run', 'runs/r01', '--data', 'data/ori']


def test_a_custom_run_finds_its_rig_and_is_not_scored(cli, tmp_path, monkeypatch):
    """Own recordings ship no reference, and their rig comes from `./davio rig`."""
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    rig = tmp_path / 'config/custom/walk/estimator_config.yaml'
    rig.parent.mkdir(parents=True)
    rig.write_text('')
    assert cli.main(['run', 'walk', '--dataset', 'custom']) == 0
    assert len(cli.calls) == 1                                  # no evaluate_run call
    argv = cli.calls[0]
    assert argv[argv.index('--rig') + 1] == 'config/custom/walk/estimator_config.yaml'
    assert argv[argv.index('--data') + 1] == 'data/custom'


def test_a_custom_run_without_a_rig_says_how_to_make_one(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    assert cli.main(['run', 'walk', '--dataset', 'custom']) == 2
    assert cli.calls == []
    assert './davio rig data/custom/walk' in capsys.readouterr().err


def test_data_outside_the_checkout_is_refused_because_the_container_cannot_see_it(
        cli, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'ROOT', tmp_path / 'repo')
    (tmp_path / 'repo').mkdir()
    assert cli.main(['run', 'V1_01_easy', '--data', str(tmp_path / 'elsewhere')]) == 2
    assert cli.calls == [] and 'data/' in capsys.readouterr().err


def test_data_under_the_checkout_is_passed_relative(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    cli.main(['run', 'r01', '--dataset', 'ori', '--data', str(tmp_path / 'data/mine')])
    argv = cli.calls[0]
    assert argv[argv.index('--data') + 1] == 'data/mine'


def test_evaluate_reads_the_dataset_folder_from_the_run(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/r01').mkdir(parents=True)
    (tmp_path / 'runs/r01/run.json').write_text(
        '{"dataset": "ori", "data_root": "/workspace/data/ori"}')
    assert cli.main(['evaluate', 'runs/r01/']) == 0
    assert cli.calls[0] == ['python3', 'scripts/evaluate_run.py', '--run', 'runs/r01',
                            '--data', 'data/ori']


def test_evaluate_refuses_a_run_that_has_no_reference(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/walk').mkdir(parents=True)
    (tmp_path / 'runs/walk/run.json').write_text(
        '{"dataset": "custom", "data_root": "/workspace/data/custom"}')
    assert cli.main(['evaluate', 'runs/walk']) == 2 and cli.calls == []


def test_a_remap_reports_the_scores_it_made_itself(cli, tmp_path, monkeypatch, capsys):
    """replay.json already holds the re-map's ATE; evaluate_run cannot score a re-map."""
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    run = tmp_path / 'runs/r01_remap'
    run.mkdir(parents=True)
    (run / 'run.json').write_text('{"dataset": "ori", "data_root": "/workspace/data/ori"}')
    (run / 'replay.json').write_text('{"ate_raw_m": 0.207, "ate_map_final_m": 0.109}')
    assert cli.main(['evaluate', 'runs/r01_remap']) == 0
    assert cli.calls == []
    assert '0.109' in capsys.readouterr().out


def test_surface_scoring_exports_the_map_first_when_it_is_missing(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    run = tmp_path / 'runs/r01_remap'
    run.mkdir(parents=True)
    (run / 'run.json').write_text('{"dataset": "ori", "data_root": "/workspace/data/ori"}')
    (run / 'replay.json').write_text('{}')
    assert cli.main(['evaluate', 'runs/r01_remap', '--surface', '--force']) == 0
    assert [c[1] for c in cli.calls] == ['scripts/export_map.py', 'scripts/evaluate_surface.py']
    assert cli.calls[1][2:] == ['--run', 'runs/r01_remap', '--data', 'data/ori', '--force']


def test_rig_writes_to_the_folder_run_looks_in(cli):
    cli.main(['rig', 'data/custom/walk/'])
    assert cli.calls[0][1:] == ['scripts/phone_rig.py', '--session', 'data/custom/walk',
                                '--out', 'config/custom/walk']


def test_run_refuses_to_write_into_an_existing_run_directory(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path / 'runs/V1_01_easy').mkdir(parents=True)
    assert cli.main(['run', 'V1_01_easy']) == 2
    assert cli.calls == []
