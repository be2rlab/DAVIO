"""Explicit feature backends; learned matching never silently falls back to ORB."""
from pathlib import Path
import sys
import cv2
import numpy as np

class Features:
    def __init__(self, cfg):
        self.backend = cfg.get('feature_backend', 'orb')
        self.top_k = int(cfg.get('xfeat_features', 2048))
        if self.backend == 'orb':
            self.orb = cv2.ORB_create(nfeatures=int(cfg['orb_features']))
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.backend in ('xfeat_lighterglue', 'xfeat_mnn'):
            root = Path(cfg.get('xfeat_path', 'thirdparty/accelerated_features')).resolve()
            if not (root / 'modules/xfeat.py').is_file():
                raise RuntimeError('Missing official XFeat checkout; run scripts/fetch_xfeat.sh')
            sys.path.insert(0, str(root))
            import torch
            from modules.xfeat import XFeat
            self.torch = torch
            self.model = XFeat(top_k=self.top_k)
            # xfeat_mnn keeps the detector, descriptors and keypoint budget and replaces only
            # the learned matcher with mutual nearest neighbours, which is the controlled
            # comparison for LighterGlue that an ORB arm is not.
            if self.backend == 'xfeat_lighterglue':
                from modules.lighterglue import LighterGlue
                self.model.lighterglue = LighterGlue().eval()
            self.min_cossim = float(cfg.get('xfeat_mnn_min_cossim', .82))
        else:
            raise ValueError('Unknown feature_backend: ' + self.backend)

    def extract(self, rgb):
        if self.backend == 'orb':
            keys, desc = self.orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
            return np.array([k.pt for k in keys], np.float32).reshape(-1, 2), desc
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.
        out = self.model.detectAndCompute(tensor)[0]
        return out['keypoints'].cpu().numpy(), out['descriptors'].cpu().numpy()

    def match(self, a, b):
        if a['descriptors'] is None or b['descriptors'] is None:
            return np.empty((0, 2), int)
        if min(len(a['pixels']), len(b['pixels'])) < 2:
            return np.empty((0, 2), int)
        if self.backend == 'xfeat_mnn':
            first = self.torch.as_tensor(a['descriptors'], device=self.model.dev)
            second = self.torch.as_tensor(b['descriptors'], device=self.model.dev)
            idx0, idx1 = self.model.match(first, second, min_cossim=self.min_cossim)
            return np.column_stack([idx0.cpu().numpy(), idx1.cpu().numpy()]).astype(int).reshape(-1, 2)
        if self.backend == 'xfeat_lighterglue':
            def data(kit):
                return dict(keypoints=self.torch.as_tensor(kit['pixels'], device=self.model.dev),
                            descriptors=self.torch.as_tensor(kit['descriptors'], device=self.model.dev),
                            image_size=(kit['shape'][1], kit['shape'][0]))
            return self.model.match_lighterglue(data(a), data(b))[2].astype(int)
        pairs = [m[0] for m in self.matcher.knnMatch(a['descriptors'], b['descriptors'], k=2)
                 if len(m) == 2 and m[0].distance < .7 * m[1].distance]
        used, unique = set(), []
        for m in sorted(pairs, key=lambda m: m.distance):
            if m.trainIdx not in used:
                used.add(m.trainIdx)
                unique.append((m.queryIdx, m.trainIdx))
        return np.asarray(unique, int).reshape(-1, 2)

    def retrieval_score(self, a, b, budget=512, min_cossim=.82):
        if self.backend not in ('xfeat_lighterglue', 'xfeat_mnn'):
            raise ValueError('appearance retrieval needs XFeat descriptors')
        if a.get('descriptors') is None or b.get('descriptors') is None:
            return 0
        first = self.torch.as_tensor(a['descriptors'][:budget], device=self.model.dev)
        second = self.torch.as_tensor(b['descriptors'][:budget], device=self.model.dev)
        if min(len(first), len(second)) < 2:
            return 0
        idx0, _idx1 = self.model.match(first, second, min_cossim=min_cossim)
        return int(len(idx0))
