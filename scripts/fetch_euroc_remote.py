#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BUNDLES = {
    'machine_hall': 'https://www.research-collection.ethz.ch/server/api/core/bitstreams/'
                    '7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content',
    'vicon_room1': 'https://www.research-collection.ethz.ch/server/api/core/bitstreams/'
                   '02ecda9a-298f-498b-970c-b7c44334d880/content',
    'vicon_room2': 'https://www.research-collection.ethz.ch/server/api/core/bitstreams/'
                   'ea12bc01-3677-4b4c-853d-87c7870b8c44/content',
}
ALL_SEQUENCES = ('MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult',
                 'MH_05_difficult', 'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
                 'V2_01_easy', 'V2_02_medium', 'V2_03_difficult')
# Monocular + IMU + evaluation references. cam1, leica0 and vicon0 are deliberately absent.
KEEP = ('mav0/cam0/', 'mav0/imu0/', 'mav0/state_groundtruth_estimate0/',
        'mav0/pointcloud0/', 'mav0/body.yaml')
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0'


def category(sequence):
    if sequence.startswith('MH_'):
        return 'machine_hall'
    if sequence.startswith('V1_'):
        return 'vicon_room1'
    if sequence.startswith('V2_'):
        return 'vicon_room2'
    raise ValueError(f'unknown EuRoC sequence {sequence!r}')


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of an HTTP resource, with block read-ahead and backoff."""

    def __init__(self, url, block=32 << 20, attempts=8):
        self.url, self.block, self.attempts = url, block, attempts
        self.position, self.cache_start, self.cache = 0, 0, b''
        self.requests = self.bytes_fetched = 0
        self.size = self._probe_size()

    def _get(self, start, end):
        delay = 5.
        for attempt in range(self.attempts):
            request = urllib.request.Request(self.url, headers={
                'User-Agent': UA, 'Range': f'bytes={start}-{end}'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f'server ignored the range request ({response.status})')
                    data = response.read()
                self.requests += 1
                self.bytes_fetched += len(data)
                return data
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if attempt == self.attempts - 1:
                    raise
                code = getattr(exc, 'code', None)
                print(f'  range {start}-{end} failed ({code or exc}); retry in {delay:.0f}s',
                      file=sys.stderr, flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300.)
        raise OSError('unreachable')

    def _probe_size(self):
        request = urllib.request.Request(self.url, headers={'User-Agent': UA,
                                                            'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            header = response.headers.get('Content-Range', '')
        if '/' not in header:
            raise OSError(f'no Content-Range from {self.url}; range requests unsupported')
        return int(header.rsplit('/', 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.position
        n = max(0, min(n, self.size - self.position))
        out = bytearray()
        while len(out) < n:
            offset = self.position - self.cache_start
            if not (0 <= offset < len(self.cache)):
                # Directory reads jump around near the end of the file; small blocks there
                # avoid refetching 32 MB for every central-directory probe.
                length = self.block if n - len(out) > (1 << 16) else (1 << 20)
                length = max(length, n - len(out))
                end = min(self.size, self.position + length) - 1
                self.cache_start, self.cache = self.position, self._get(self.position, end)
                offset = 0
            take = self.cache[offset:offset + n - len(out)]
            out += take
            self.position += len(take)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class Window(io.RawIOBase):
    """A byte range of another seekable file, presented as a file of its own."""

    def __init__(self, parent, start, length):
        self.parent, self.start, self.length, self.position = parent, start, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.length}[whence]
        self.position = max(0, min(self.length, base + offset))
        return self.position

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.length - self.position
        n = max(0, min(n, self.length - self.position))
        self.parent.seek(self.start + self.position)
        data = self.parent.read(n)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def inner_zip(outer_file, outer, member, spool_dir):
    info = outer.getinfo(member)
    if info.compress_type == zipfile.ZIP_STORED:
        outer_file.seek(info.header_offset)
        header = outer_file.read(30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError(f'bad local header for {member}')
        name_length = int.from_bytes(header[26:28], 'little')
        extra_length = int.from_bytes(header[28:30], 'little')
        start = info.header_offset + 30 + name_length + extra_length
        return zipfile.ZipFile(Window(outer_file, start, info.compress_size)), None
    spool_dir = Path(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool = spool_dir / (Path(member).name + '.spool')
    started, written = time.monotonic(), 0
    with outer.open(info) as source, spool.open('wb') as sink:
        while True:
            chunk = source.read(8 << 20)
            if not chunk:
                break
            sink.write(chunk)
            written += len(chunk)
            if written % (512 << 20) < len(chunk):
                rate = written / max(time.monotonic() - started, 1e-6) / 1e6
                print(f'  inflating {Path(member).name}: {written / 1e9:.2f} / '
                      f'{info.file_size / 1e9:.2f} GB, {rate:.1f} MB/s', flush=True)
    return zipfile.ZipFile(spool), spool


def fetch(sequence, destination, dry_run=False, spool_dir=None):
    root = Path(destination) / sequence
    manifest_path = root / 'fetch_manifest.json'
    if manifest_path.is_file():
        print(f'{sequence}: already fetched ({manifest_path})')
        return json.loads(manifest_path.read_text())
    remote = RemoteFile(BUNDLES[category(sequence)])
    outer = zipfile.ZipFile(remote)
    candidates = [n for n in outer.namelist()
                  if n.endswith(f'/{sequence}/{sequence}.zip') or n.endswith(f'{sequence}.zip')]
    if not candidates:
        raise FileNotFoundError(f'{sequence}.zip not in bundle {category(sequence)}')
    member_info = outer.getinfo(candidates[0])
    if dry_run:
        print(f'{sequence}: {candidates[0]} is '
              f'{"stored" if member_info.compress_type == zipfile.ZIP_STORED else "deflated"}, '
              f'{member_info.compress_size / 1e9:.2f} GB in bundle, '
              f'{member_info.file_size / 1e9:.2f} GB inflated')
        return dict(sequence=sequence, member=candidates[0],
                    compressed=member_info.compress_size, inflated=member_info.file_size)
    inner, spool = inner_zip(remote, outer, candidates[0],
                             spool_dir or Path(destination) / '.spool')
    members = [i for i in inner.infolist()
               if not i.is_dir() and i.filename.startswith(KEEP)]
    total = sum(i.file_size for i in members)
    print(f'{sequence}: {len(members)} members, {total / 1e9:.2f} GB to extract '
          f'(of {sum(i.file_size for i in inner.infolist()) / 1e9:.2f} GB in the ASL zip)',
          flush=True)
    files, started = {}, time.monotonic()
    # Archive order keeps reads sequential, which is what makes the read-ahead block pay.
    for count, info in enumerate(sorted(members, key=lambda i: i.header_offset)):
        target = root / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        partial = target.with_name(target.name + '.part')
        with inner.open(info) as source, partial.open('wb') as sink:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                sink.write(chunk)
        partial.replace(target)
        files[info.filename] = dict(bytes=info.file_size, sha256=digest.hexdigest())
        if count % 500 == 0:
            rate = remote.bytes_fetched / max(time.monotonic() - started, 1e-6) / 1e6
            print(f'  {sequence}: {count}/{len(members)} files, '
                  f'{remote.bytes_fetched / 1e9:.2f} GB fetched, {rate:.1f} MB/s', flush=True)
    manifest = dict(sequence=sequence, source=BUNDLES[category(sequence)],
                    inner_member=candidates[0], kept_prefixes=list(KEEP),
                    files=len(files), bytes=sum(f['bytes'] for f in files.values()),
                    http_requests=remote.requests, bytes_fetched=remote.bytes_fetched,
                    wall_s=time.monotonic() - started, contents=files)
    inner.close()
    if spool is not None:
        spool.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f'{sequence}: done, {manifest["bytes"] / 1e9:.2f} GB in {manifest["wall_s"]:.0f}s '
          f'with {remote.requests} range requests', flush=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/euroc'))
    p.add_argument('--sequences', nargs='+', default=list(ALL_SEQUENCES))
    p.add_argument('--dry-run', action='store_true', help='Report bundle member sizes only')
    p.add_argument('--spool', type=Path, help='Where a deflated ASL zip is inflated '
                   '(default DATA/.spool); deleted after each sequence')
    a = p.parse_args(argv)
    for sequence in a.sequences:
        fetch(sequence, a.data, a.dry_run, a.spool)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
