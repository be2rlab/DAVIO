"""XFeat keypoints, matched by LighterGlue."""
from pathlib import Path
import sys

import numpy as np


class Features:
    def __init__(self, cfg):
        root = Path(__file__).resolve().parents[3] / cfg['xfeat_path']
        if not (root / 'modules/xfeat.py').is_file():
            raise RuntimeError('Missing XFeat checkout; run scripts/fetch_xfeat.sh')
        sys.path.insert(0, str(root))
        import torch
        from modules.xfeat import XFeat
        from modules.lighterglue import LighterGlue
        self.torch = torch
        self.model = XFeat(top_k=int(cfg['xfeat_features']))
        self.model.lighterglue = LighterGlue().eval()

    def extract(self, rgb):
        """Keypoints (N, 2) in pixels and their descriptors of one RGB image."""
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.
        out = self.model.detectAndCompute(tensor)[0]
        return out['keypoints'].cpu().numpy(), out['descriptors'].cpu().numpy()

    def match(self, a, b):
        """Index pairs (M, 2) between two sets of keypoints (dicts with pixels, descriptors, shape)."""
        if a['descriptors'] is None or b['descriptors'] is None or min(len(a['pixels']), len(b['pixels'])) < 2:
            return np.empty((0, 2), int)

        def data(kit):
            return dict(keypoints=self.torch.as_tensor(kit['pixels'], device=self.model.dev),
                        descriptors=self.torch.as_tensor(kit['descriptors'], device=self.model.dev),
                        image_size=(kit['shape'][1], kit['shape'][0]))
        return self.model.match_lighterglue(data(a), data(b))[2].astype(int)
