#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 2 * 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require_space(path, size):
    if shutil.disk_usage(path).free < size + RESERVE:
        raise RuntimeError(f'Insufficient disk: need {size + RESERVE:,} free bytes at {path}')


def safe_members(archive):
    members = archive.infolist()
    for item in members:
        p = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        if (p.is_absolute() or '..' in p.parts or '\\' in item.filename
                or (p.parts and ':' in p.parts[0]) or stat.S_ISLNK(mode)):
            raise ValueError(f'Unsafe ZIP member: {item.filename}')
    return members


def validate_layout(path, seq):
    required = [path / f'{seq}_bag/metadata.yaml', path / f'{seq}_gt/cloud_gt.pcd',
                path / f'{seq}_gt/poses_gt.txt']
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise ValueError(f'Missing/nonempty ORI metadata or ground truth under {path}')
    if not any(p.stat().st_size for p in (path / f'{seq}_bag').glob('*.mcap')):
        raise ValueError('Missing MCAP sensor bag')
    if not any(p.stat().st_size for p in (path / f'{seq}_gt/cloud_gt_fov').glob('*.pcd')):
        raise ValueError('Missing camera-FoV reference clouds')


def extract(archive_path, destination, seq, provenance):
    # Caller never overwrites an existing sequence, including an incomplete one.
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    with zipfile.ZipFile(archive_path) as z:
        members = safe_members(z)
        require_space(destination.parent, sum(m.file_size for m in members))
        with tempfile.TemporaryDirectory(prefix=f'.{seq}-extract-', dir=destination.parent) as tmp:
            stage = Path(tmp)
            z.extractall(stage)  # zipfile verifies CRC while reading each file.
            payload = stage / seq if (stage / seq).is_dir() else stage
            validate_layout(payload, seq)
            files = [{"path": str(p.relative_to(payload)), "bytes": p.stat().st_size}
                     for p in sorted(payload.rglob('*')) if p.is_file()]
            record = dict(provenance, files=files, status='downloaded_and_extracted_not_run',
                          extracted_at=datetime.now(timezone.utc).isoformat())
            (payload / 'download_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
            if payload == stage:
                # Rename a child so TemporaryDirectory always retains its own root.
                ready = stage / '.ready'
                ready.mkdir()
                for child in list(stage.iterdir()):
                    if child != ready:
                        child.rename(ready / child.name)
                payload = ready
            payload.rename(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--sequences', nargs='+', default=['r01'], help='r01..r05 or all; default r01')
    p.add_argument('--plan', action='store_true', help='Print URLs/sizes only; no writes or network')
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--extract-only', action='store_true', help='Use already downloaded cache archives')
    a = p.parse_args()
    if a.download_only and a.extract_only:
        p.error('--download-only and --extract-only are mutually exclusive')
    catalog = json.loads((ROOT / 'config/ori/downloads.json').read_text())
    seqs = [x.lower() for x in a.sequences]
    if seqs == ['all']:
        seqs = list(catalog['sequences'])
    if len(seqs) != len(set(seqs)) or any(s not in catalog['sequences'] for s in seqs):
        p.error('Select unique sequences r01..r05, or all by itself')
    root = a.data_root.resolve() / 'ori'
    total = sum(catalog['sequences'][s]['bytes'] for s in seqs)
    print(f'{len(seqs)} archive(s), {total:,} compressed bytes; destination {root}', flush=True)
    for seq in seqs:
        entry = catalog['sequences'][seq]
        print(f"{seq}: {entry['bytes']:,} bytes https://drive.google.com/file/d/{entry['file_id']}/view", flush=True)
    if a.plan:
        return
    cache = root / '.download_cache'
    cache.mkdir(parents=True, exist_ok=True)
    # One invocation owns the download/extraction lifecycle. Never auto-remove a stale lock.
    lock = root / '.fetch.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        for seq in seqs:
            entry = catalog['sequences'][seq]
            dest = root / seq
            if dest.exists():
                validate_layout(dest, seq)
                manifest = json.loads((dest / 'download_manifest.json').read_text())
                if manifest['file_id'] != entry['file_id'] or manifest['archive_bytes'] != entry['bytes']:
                    raise ValueError(f'Existing provenance differs: {dest}')
                for item in manifest['files']:
                    f = dest / item['path']
                    if not f.is_file() or f.stat().st_size != item['bytes']:
                        raise ValueError(f'Existing data incomplete: {f}')
                print(f'Already extracted, layout/size inventory checked: {seq}', flush=True)
                continue
            archive = cache / f'{seq}.zip'
            partial = cache / f'{seq}.zip.part'
            if not archive.exists():
                if a.extract_only:
                    raise FileNotFoundError(archive)
                if importlib.util.find_spec('gdown') is None:
                    raise RuntimeError('Install download helper in your environment: python3 -m pip install gdown')
                require_space(cache, max(0, entry['bytes'] - (partial.stat().st_size if partial.exists() else 0)))
                subprocess.run([sys.executable, '-m', 'gdown', entry['file_id'], '-O', str(partial), '--continue'], check=True)
                if partial.stat().st_size != entry['bytes']:
                    raise ValueError(f'Archive size mismatch for {seq}; keep partial for inspection; do not extract')
                if not zipfile.is_zipfile(partial):
                    raise ValueError('Download is not a ZIP archive')
                partial.rename(archive)
            if archive.stat().st_size != entry['bytes']:
                raise ValueError(f'Cached archive size mismatch: {archive}')
            provenance = dict(source_page=catalog['source_page'], source_folder=catalog['source_folder'],
                              file_id=entry['file_id'], archive_bytes=entry['bytes'],
                              archive_sha256=digest(archive), checksum_origin='locally computed; not author supplied')
            sidecar = cache / f'{seq}.json'
            if sidecar.exists() and json.loads(sidecar.read_text()) != provenance:
                raise ValueError(f'Cached archive hash/provenance changed: {archive}')
            sidecar.write_text(json.dumps(provenance, indent=2)+'\n')
            if not a.download_only:
                extract(archive, dest, seq, provenance)
                print(f'Extracted {seq}; adapter/calibration checks still required before DAVIO runs', flush=True)
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        sys.exit(f'ORI fetch failed: {exc}')
