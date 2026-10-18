"""The viewer must say whether it is showing a live run, a recording, or a replay."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from view_run import run_provenance         # noqa: E402


def _run(tmp_path, **meta):
    (tmp_path / 'run.json').write_text(json.dumps(meta))
    return tmp_path


def test_live_run_is_labelled_live(tmp_path):
    label, detail = run_provenance(_run(tmp_path, state='running', sequence='V1_01_easy'))
    assert label == 'LIVE' and 'V1_01_easy' in detail


def test_completed_run_is_not_called_live(tmp_path):
    label, detail = run_provenance(_run(tmp_path, state='completed', sequence='V1_02_medium'))
    assert label == 'RECORDED'
    assert 'not a measurement being taken now' in detail


def test_cached_replay_says_no_scheduler_took_part(tmp_path):
    label, detail = run_provenance(_run(
        tmp_path, kind='cached-submap replay', state='completed', sequence='V1_01_easy',
        source_run={'path': 'runs/v1_01_fixed'}))
    assert label == 'CACHED REPLAY'
    assert 'scheduler' in detail and 'runs/v1_01_fixed' in detail


def test_missing_or_broken_descriptor_is_not_silently_live(tmp_path):
    assert run_provenance(tmp_path)[0] == 'AWAITING'
    (tmp_path / 'run.json').write_text('{not json')
    assert run_provenance(tmp_path)[0] == 'UNREADABLE'
