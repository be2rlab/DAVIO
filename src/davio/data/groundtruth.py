import csv
import hashlib
from pathlib import Path

import numpy as np

# Sequences whose shipped ASL orientation reference is documented as inaccurate.
# Position stays usable, which is why this is a per-field flag and not a rejection.
ASL_ORIENTATION_SUSPECT = ('V1_01_easy',)

VARIANTS = ('dataset', 'openvins', 'openvins_original', 'auto')


def _openvins_dir(repo_root):
    return Path(repo_root) / 'thirdparty/open_vins/ov_data/euroc_mav'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_asl_state_csv(path):
    """(t_s, p, q_wxyz, v, b_gyro, b_accel) from either reference layout."""
    rows = []
    with Path(path).open() as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith('#'):
                continue
            values = [float(x) for x in row[:17]]
            if len(values) < 8:
                raise ValueError(f'{path}: reference row needs at least time, p and q')
            rows.append(values + [0.] * (17 - len(values)))
    if not rows:
        raise ValueError(f'{path}: no reference samples')
    table = np.asarray(rows, float)
    return (table[:, 0] * 1e-9, table[:, 1:4], table[:, 4:8],
            table[:, 8:11], table[:, 11:14], table[:, 14:17])


def resolve(dataset, sequence, mav0_dir, repo_root, choice='dataset'):
    if choice not in VARIANTS:
        raise ValueError(f'Unknown ground-truth variant {choice!r}; expected {VARIANTS}')
    shipped = Path(mav0_dir) / 'state_groundtruth_estimate0/data.csv'
    suspect = dataset == 'euroc' and sequence in ASL_ORIENTATION_SUSPECT

    if choice in ('openvins', 'openvins_original') or (choice == 'auto' and suspect):
        if dataset != 'euroc':
            raise ValueError(f'No OpenVINS reference distributed for dataset {dataset!r}')
        name = sequence + ('_original' if choice == 'openvins_original' else '')
        path = _openvins_dir(repo_root) / f'{name}.csv'
        if not path.is_file():
            if choice == 'auto':
                return shipped, 'dataset', not suspect
            raise FileNotFoundError(
                f'{path} is missing; run `git submodule update --init thirdparty/open_vins`')
        # The _original file is a copy of the shipped reference, so it inherits the
        # same caveat; the corrected file is exactly what repairs it.
        return path, ('openvins_original' if choice == 'openvins_original' else 'openvins'), \
            not (choice == 'openvins_original' and suspect)
    return shipped, 'dataset', not suspect


def provenance(path, variant, orientation_reliable, times):
    """Identity of the reference behind a score, small enough to embed in any output."""
    times = np.asarray(times, float)
    steps = np.diff(times)
    return dict(
        variant=variant, path=str(path), sha256=sha256(path), samples=int(times.size),
        first_time=float(times[0]) if times.size else None,
        last_time=float(times[-1]) if times.size else None,
        rate_hz=float(1. / np.median(steps)) if steps.size else None,
        orientation_reliable=bool(orientation_reliable),
        note=('shipped ASL reference; OpenVINS documents this sequence\'s orientation as '
              'inaccurate' if variant in ('dataset', 'openvins_original') and not orientation_reliable
              else 'selected reference'))
