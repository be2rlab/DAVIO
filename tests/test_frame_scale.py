"""Per-frame depth scale refinement with poses fixed (davio_mapper/frame_scale.py)."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from davio_mapper.frame_scale import refine_frame_scales          # noqa: E402


class Matcher:
    """Descriptor-free stand-in: keypoints carry their point id in the descriptor."""

    def match(self, a, b):
        ida, idb = a['descriptors'][:, 0].astype(int), b['descriptors'][:, 0].astype(int)
        common = np.intersect1d(ida, idb)
        return np.column_stack((np.searchsorted(ida, common), np.searchsorted(idb, common)))


CFG = dict(frame_scale_pairs=2, frame_scale_min_matches=10, frame_scale_prior_sigma=1.)


def _scene(truth, noise=0.):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, -1, 2], [1, 1, 4], size=(200, 3))
    k = np.array([[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]])
    kits, poses = [], []
    for i, s in enumerate(truth):
        pose = np.eye(4)
        pose[:3, 3] = [0.1 * i, 0., 0.]                                 # T_{centre<-i}
        pc = (pts - pose[:3, 3]) @ pose[:3, :3]
        uv = (pc @ k.T)[:, :2] / pc[:, 2:3]
        depth = pc[:, 2] * s * (1 + noise * rng.normal(size=len(pc)))  # depth WRONG by s
        kits.append(dict(pixels=uv.astype(np.float32), descriptors=np.arange(200)[:, None].astype(np.float32),
                         depth=depth, K=k, shape=(96, 128)))
        poses.append(pose)
    return kits, np.array(poses)


def test_frame_scales_are_recovered_with_poses_fixed():
    truth = np.array([1.0, 1.15, 0.9, 1.05, 1.2])
    kits, poses = _scene(truth)
    scales, info = refine_frame_scales(kits, poses, CFG, matcher=Matcher())
    # Relative corrections only (geometric mean one): every frame ends on one common scale.
    common = np.exp(np.mean(np.log(truth)))
    np.testing.assert_allclose(scales * truth, common, atol=2e-3)
    assert np.exp(np.mean(np.log(scales))) == pytest.approx(1.)
    assert info['pairs_used'] == 7 and info['rms_after_rel'] < info['rms_before_rel']


def test_outliers_do_not_move_the_scales_much():
    truth = np.array([1.0, 1.15, 0.9, 1.05, 1.2])
    kits, poses = _scene(truth, noise=.02)                                  # 2 % depth noise
    rng = np.random.default_rng(1)
    for kit in kits:                                                    # 20 % gross depth outliers
        bad = rng.random(200) < .2
        kit['depth'][bad] *= rng.uniform(1.5, 3., size=bad.sum())
    scales, _info = refine_frame_scales(kits, poses, CFG, matcher=Matcher())
    np.testing.assert_allclose(scales * truth, np.exp(np.mean(np.log(truth))), atol=3e-2)


def test_unmatched_frames_keep_unit_scale():
    k = np.eye(3)
    kits = [dict(pixels=np.zeros((0, 2)), descriptors=None, depth=np.zeros(0), K=k, shape=(4, 4))
            for _ in range(3)]
    scales, info = refine_frame_scales(kits, np.repeat(np.eye(4)[None], 3, axis=0), CFG, matcher=Matcher())
    np.testing.assert_allclose(scales, 1.)
    assert info['pairs_used'] == 0


def _anchors(offset_m=.4, n=5):
    """A previous, already-metric window: correct depth, displaced along x by ``offset_m``."""
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, -1, 2], [1, 1, 4], size=(200, 3))            # the same scene points
    k = np.array([[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]])
    out = []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, 3] = [-offset_m - 0.1 * i, 0., 0.]
        pc = (pts - pose[:3, 3]) @ pose[:3, :3]
        uv = (pc @ k.T)[:, :2] / pc[:, 2:3]
        out.append((dict(pixels=uv.astype(np.float32), descriptors=np.arange(200)[:, None].astype(np.float32),
                         depth=pc[:, 2], K=k, shape=(96, 128)), pose))
    return out


def test_absolute_mode_is_pinned_by_anchor_matches_not_by_the_window():
    truth = np.array([1.1, 1.25, 1.0, 1.15, 1.3])                     # every frame too deep
    kits, poses = _scene(truth, noise=.02)
    rng = np.random.default_rng(2)
    for kit in kits:                                                    # 20 % gross outliers
        bad = rng.random(200) < .2
        kit['depth'][bad] *= rng.uniform(1.5, 3., size=bad.sum())
    cfg = dict(CFG, frame_scale_absolute=True, frame_scale_abs_prior_sigma=.3)
    alone, info0 = refine_frame_scales(kits, poses, cfg, matcher=Matcher())
    assert info0['absolute'] and info0['anchor_matches'] == 0
    assert abs(np.mean(alone * truth) - 1.) > .05                       # the window alone cannot pin it
    scales, info = refine_frame_scales(kits, poses, cfg, matcher=Matcher(), anchors=_anchors())
    assert info['anchor_matches'] > 0
    np.testing.assert_allclose(scales * truth, 1., atol=2e-2)          # absolute, within 2 %


def test_absolute_mode_keeps_unit_scale_without_baseline():
    kits, poses = _scene(np.array([1.2]))
    kits, poses = [dict(kits[0]) for _ in range(3)], np.repeat(np.eye(4)[None], 3, axis=0)   # one camera, three copies
    cfg = dict(CFG, frame_scale_absolute=True)
    scales, _ = refine_frame_scales(kits, poses, cfg, matcher=Matcher())
    np.testing.assert_allclose(scales, 1., atol=1e-6)                   # prior wins; nothing observable
