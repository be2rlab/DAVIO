"""Ramped causal correction (runtime/engine.py::ramp_correction)."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from davio.runtime.engine import ramp_correction     # noqa: E402


def test_ramp_is_monotone_and_reaches_the_target():
    a = np.eye(4)
    b = np.eye(4)
    b[:3, 3] = [0.3, 0., 0.]
    b[:3, :3] = Rotation.from_euler('z', 10, degrees=True).as_matrix()
    last = None
    for f in np.linspace(0., 1., 11):
        c = ramp_correction(a, b, f)
        x = c[:3, 3][0]
        angle = Rotation.from_matrix(c[:3, :3]).magnitude()
        assert last is None or (x >= last[0] - 1e-12 and angle >= last[1] - 1e-12)
        last = (x, angle)
    np.testing.assert_allclose(ramp_correction(a, b, 1.), b, atol=1e-12)
    np.testing.assert_allclose(ramp_correction(a, b, 0.), a, atol=1e-12)
    np.testing.assert_allclose(ramp_correction(a, b, 7.), b, atol=1e-12)    # clamped
