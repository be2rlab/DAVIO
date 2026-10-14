"""The view given to the depth model: the camera's frames undistorted to a pinhole image."""
import cv2
import numpy as np


class View:
    """Remaps raw frames, grey or colour, to a pinhole view with intrinsics K."""

    def __init__(self, K, map1, map2):
        self.K, self.map1, self.map2 = K, map1, map2

    def __call__(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    @property
    def resolution(self):
        """Width, height."""
        h, w = self.map1.shape[:2]
        return int(w), int(h)


def depth_view(camera, fov_deg=90., size=512):
    K, D = np.asarray(camera.K, float), np.asarray(camera.D, float)
    w, h = int(camera.resolution[0]), int(camera.resolution[1])
    if camera.model == 'equidistant':
        edges = np.float32([[[0, K[1, 2]]], [[w - 1, K[1, 2]]], [[K[0, 2], 0]], [[K[0, 2], h - 1]]])
        rays = cv2.fisheye.undistortPoints(edges, K, D.reshape(4, 1)).reshape(-1, 2)
        if np.degrees(np.arctan(np.linalg.norm(rays, axis=1))).min() < fov_deg / 2.:
            raise ValueError(f'the lens does not fill a {fov_deg} degree view')
        f = (size / 2.) / np.tan(np.radians(fov_deg / 2.))
        K_new = np.array([[f, 0., size / 2. - .5], [0., f, size / 2. - .5], [0., 0., 1.]])
        return View(K_new, *cv2.fisheye.initUndistortRectifyMap(K, D.reshape(4, 1), np.eye(3), K_new, (size, size), cv2.CV_16SC2))
    if camera.model == 'radtan':
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0., (w, h))
        return View(K_new, *cv2.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv2.CV_16SC2))
    raise ValueError(f'unknown camera model: {camera.model!r}')
