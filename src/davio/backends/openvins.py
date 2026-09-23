"""Native OpenVINS adapter. The only write path is the one-shot bootstrap ``initialize``."""
import numpy as np


class OpenVinsBackend:
    def __init__(self, config_path):
        import openvins_ext
        self._vio = openvins_ext.VioManager(str(config_path))
        self._has_diagnostics = hasattr(self._vio, 'get_diagnostics')

    def feed_imu(self, t, wm, am):
        self._vio.feed_imu(t, wm, am)

    def feed_camera(self, t, image, cam_id=0):
        self._vio.feed_camera(t, image, cam_id)

    def initialized(self):
        return self._vio.initialized()

    def initialize(self, state):
        as_f = lambda k, n: np.ascontiguousarray(np.asarray(state[k], float).reshape(n))
        self._vio.initialize(float(state['t']), as_f('q_GtoI', 4), as_f('p', 3), as_f('v', 3),
                             as_f('bg', 3), as_f('ba', 3), as_f('sigmas', 15))

    def state(self):
        return self._vio.get_state()

    def diagnostics(self):
        if not self._has_diagnostics or not self._vio.initialized():
            return None
        try:
            return self._vio.get_diagnostics()
        except Exception:                      # noqa: BLE001 - a shim fault must not stop tracking
            self._has_diagnostics = False
            return None

    def close(self):
        self._vio = None
