"""Fetch lifecycle checks using tiny offline ZIP fixtures (no network or SLAM)."""
import importlib.util
import json
from pathlib import Path
import stat
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('fetch_ori', Path(__file__).parents[1] / 'scripts/fetch_ori.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def fixture_zip(path, prefix='r01/'):
    with zipfile.ZipFile(path, 'w') as z:
        for name in ['r01_bag/metadata.yaml', 'r01_bag/data.mcap',
                     'r01_gt/cloud_gt.pcd', 'r01_gt/poses_gt.txt', 'r01_gt/cloud_gt_fov/1_0.pcd']:
            z.writestr(prefix + name, 'fixture')


@pytest.mark.parametrize('prefix', ['r01/', ''])
def test_extract_validates_layout_and_refuses_overwrite(tmp_path, prefix):
    archive = tmp_path / 'r01.zip'
    fixture_zip(archive, prefix)
    dest = tmp_path / 'r01'
    fetch.extract(archive, dest, 'r01', {'archive_sha256': fetch.digest(archive)})
    fetch.validate_layout(dest, 'r01')
    record = json.loads((dest / 'download_manifest.json').read_text())
    assert record['status'] == 'downloaded_and_extracted_not_run'
    assert len(record['files']) == 5
    with pytest.raises(FileExistsError):
        fetch.extract(archive, dest, 'r01', {})


def test_incomplete_archive_is_not_published(tmp_path):
    archive = tmp_path / 'incomplete.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('r01/r01_bag/metadata.yaml', 'only metadata')
    with pytest.raises(ValueError):
        fetch.extract(archive, tmp_path / 'r01', 'r01', {})
    assert not (tmp_path / 'r01').exists()


@pytest.mark.parametrize('name,mode', [('../escape', 0), ('/absolute', 0),
                                      ('r01/link', stat.S_IFLNK | 0o777)])
def test_unsafe_zip_not_extracted(tmp_path, name, mode):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        member = zipfile.ZipInfo(name)
        member.external_attr = mode << 16
        z.writestr(member, 'target')
    with pytest.raises(ValueError):
        fetch.extract(archive, tmp_path / 'r01', 'r01', {})
    assert not (tmp_path / 'r01').exists()


def test_disk_budget_checked_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / 'r01.zip'
    fixture_zip(archive)
    monkeypatch.setattr(fetch.shutil, 'disk_usage', lambda p: type('Usage', (), {'free': 0})())
    with pytest.raises(RuntimeError, match='Insufficient disk'):
        fetch.extract(archive, tmp_path / 'r01', 'r01', {})
    assert not (tmp_path / 'r01').exists()
