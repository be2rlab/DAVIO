import numpy as np

_P_LO, _P_HI = 1.0, 99.0


def _to_gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img

_MAX_SATURATED_FRAC = 0.5


def to_u8(images, mode="window_stretch", max_saturated_frac=_MAX_SATURATED_FRAC):
    if mode not in ("window_stretch", "shift8"):
        raise ValueError("unknown tone mode: %r" % (mode,))
    grays = [_to_gray(im) for im in images]
    if all(g.dtype == np.uint8 for g in grays):
        return grays                              # EuRoC path: untouched

    if mode == "shift8":
        out = [(g >> 8).astype(np.uint8) if g.dtype != np.uint8 else g for g in grays]
    else:
        pooled = np.concatenate([g.reshape(-1) for g in grays])
        lo, hi = np.percentile(pooled, [_P_LO, _P_HI])
        span = max(float(hi) - float(lo), 1.0)    # flat window -> no divide by zero
        out = [np.clip((g.astype(np.float32) - lo) * (255.0 / span), 0, 255).astype(np.uint8)
               for g in grays]

    pooled_out = np.concatenate([o.reshape(-1) for o in out])
    sat_frac = float(np.count_nonzero(pooled_out == 255)) / pooled_out.size
    if sat_frac > max_saturated_frac:
        raise ValueError(
            "%.1f%% of pixels saturated to 255 after %r conversion (threshold %.0f%%) -- "
            "this is the historical 16-bit-to-white failure (Sec. VI-A): a bad cast turns "
            "TUM VI frames pure white and hands the backbone a blank image."
            % (100.0 * sat_frac, mode, 100.0 * max_saturated_frac))
    return out


def to_u8_color(images, mode="window_stretch"):
    if mode not in ("window_stretch", "shift8"):
        raise ValueError("unknown tone mode: %r" % (mode,))
    out = []
    for image in images:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype == np.uint8:
            out.append(image)
            continue
        if mode == "shift8":
            out.append((image >> 8).astype(np.uint8))
        else:
            lo, hi = np.percentile(image.reshape(-1), [_P_LO, _P_HI])
            span = max(float(hi) - float(lo), 1.0)
            out.append(np.clip((image.astype(np.float32) - lo) * (255.0 / span),
                               0, 255).astype(np.uint8))
    return out


def read_color_u8(path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("could not read image: %s" % (path,))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray_u8(path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("could not read image: %s" % (path,))
    return to_u8([img], mode="shift8")[0]


class Rectifier:

    __slots__ = ("K_new", "fov_deg", "_map1", "_map2", "_tone")

    def __init__(self, K_new, fov_deg, map1, map2, tone="window_stretch"):
        self.K_new = K_new
        self.fov_deg = fov_deg
        self._map1, self._map2 = map1, map2
        self._tone = tone

    def __call__(self, images):
        import cv2

        u8 = to_u8(images, mode=self._tone)
        return [cv2.remap(im, self._map1, self._map2, cv2.INTER_LINEAR) for im in u8]

    def color(self, images):
        """The same geometry, with colour kept. Used for the dense map, never the filter."""
        import cv2

        u8 = to_u8_color(images, mode=self._tone)
        return [cv2.remap(im, self._map1, self._map2, cv2.INTER_LINEAR) for im in u8]

    @property
    def output_resolution(self):
        h, w = self._map1.shape[:2]
        return int(w), int(h)


class PassThrough:
    """free calibration mode: no rectification; DA3 predicts the intrinsics itself."""

    __slots__ = ("K_new", "fov_deg", "_tone")

    def __init__(self, tone="shift8"):
        self.K_new, self.fov_deg, self._tone = None, None, tone

    def __call__(self, images):
        return to_u8(images, mode=self._tone)

    def color(self, images):
        return to_u8_color(images, mode=self._tone)


def _half_fov_deg(K, D, size):
    """Half field of view, in degrees, to the left/right/top/bottom image edges."""
    import cv2

    w, h = size
    cx, cy = K[0, 2], K[1, 2]
    edges = np.float32([[[0, cy]], [[w - 1, cy]], [[cx, 0]], [[cx, h - 1]]])
    rays = cv2.fisheye.undistortPoints(edges, K, D.reshape(4, 1)).reshape(-1, 2)
    return np.degrees(np.arctan(np.linalg.norm(rays, axis=1)))


def build_rectifier(camera, fov_deg=None, out_size=None, tone="window_stretch"):
    import cv2

    K = np.asarray(camera.K, float)
    D = np.asarray(camera.D, float)
    w, h = int(camera.resolution[0]), int(camera.resolution[1])

    if camera.model == "equidistant":
        if fov_deg is None:
            raise ValueError("equidistant rectification needs an explicit fov_deg")
        # The square fov_deg view is for a wide fisheye (ORI reaches ~88 deg on every
        # side). A narrow Kannala-Brandt lens --- a phone camera calibrated as kb4 reaches
        # ~28 deg vertically --- cannot fill a 90 deg square, which would leave most of the
        # backbone's input black. Such a camera is undistorted to a same-size pinhole, the
        # way the radtan branch does, with balance 0 so no invalid border survives.
        if min(_half_fov_deg(K, D, (w, h))) < fov_deg / 2.0:
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D.reshape(4, 1), (w, h), np.eye(3), balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D.reshape(4, 1), np.eye(3), K_new, (w, h), cv2.CV_16SC2)
            return Rectifier(K_new, None, map1, map2, tone)
        n = int(out_size or w)
        f = (n / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        K_new = np.array([[f, 0.0, n / 2.0 - 0.5],
                          [0.0, f, n / 2.0 - 0.5],
                          [0.0, 0.0, 1.0]])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D.reshape(4, 1), np.eye(3), K_new, (n, n), cv2.CV_16SC2)
        return Rectifier(K_new, float(fov_deg), map1, map2, tone)

    if camera.model == "radtan":
        K_new, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv2.CV_16SC2)
        return Rectifier(K_new, None, map1, map2, tone)

    raise ValueError("unknown camera model: %r" % (camera.model,))
