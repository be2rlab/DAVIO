import cv2
import numpy as np
import pytest

from davio.datasets import Camera
from davio.filter import read_camera
from davio.images import depth_view

RAYS = np.array([[0., 0., 1.], [.3, -.2, 1.], [-.5, .4, 1.]])


def rig_camera(root, rig):
    cam = read_camera(root / f'config/{rig}/estimator_config.yaml')
    fx, fy, cx, cy = cam['intrinsics']
    return Camera(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), D=np.array(cam['distortion_coeffs']),
                  model=cam['distortion_model'], resolution=tuple(cam['resolution']))


def raw_pixel(camera, ray):
    """Where the camera's lens puts a ray."""
    if camera.model == 'equidistant':
        return cv2.fisheye.projectPoints(ray.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.D.reshape(4, 1))[0].ravel()
    return cv2.projectPoints(ray.reshape(1, 3), np.zeros(3), np.zeros(3), camera.K, camera.D)[0].ravel()


@pytest.mark.parametrize('rig, resolution', [('euroc', (752, 480)), ('ori', (512, 512))])
def test_a_ray_lands_where_the_pinhole_intrinsics_of_the_view_say(root, rig, resolution):
    camera = rig_camera(root, rig)
    view = depth_view(camera)
    assert view.resolution == resolution and view(np.zeros((*camera.resolution[::-1], 3), np.uint8)).shape == (*resolution[::-1], 3)
    for ray in RAYS:
        raw = np.zeros(camera.resolution[::-1], np.uint8)
        cv2.circle(raw, tuple(int(x) for x in np.rint(raw_pixel(camera, ray))), 3, 255, -1)
        seen = cv2.GaussianBlur(view(raw), (9, 9), 0)
        v, u = np.unravel_index(np.argmax(seen), seen.shape)
        np.testing.assert_allclose([u, v], (view.K @ ray)[:2], atol=1.5)


def test_a_fisheye_gets_a_square_view_of_ninety_degrees(root):
    view = depth_view(rig_camera(root, 'ori'))
    np.testing.assert_allclose(view.K, [[256., 0., 255.5], [0., 256., 255.5], [0., 0., 1.]])
    assert depth_view(rig_camera(root, 'ori'), fov_deg=60., size=256).K[0, 0] == pytest.approx(128. / np.tan(np.radians(30.)))


def test_a_lens_narrower_than_the_view_or_of_an_unknown_model_is_refused():
    narrow = Camera(K=np.array([[436., 0., 320.], [0., 438., 240.], [0., 0., 1.]]), D=np.array([.01, 0., 0., 0.]),
                    model='equidistant', resolution=(640, 480))
    with pytest.raises(ValueError, match='does not fill'):
        depth_view(narrow)
    with pytest.raises(ValueError, match='unknown camera model'):
        depth_view(Camera(K=narrow.K, D=narrow.D, model='omni', resolution=(640, 480)))
