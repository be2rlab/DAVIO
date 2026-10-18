#!/usr/bin/env python3
"""Reader for the vi-recorder session format."""

import json
from pathlib import Path

import numpy as np


class Session:
    """One recording directory."""

    def __init__(self, root):
        self.root = Path(root)
        for name in ("frames.bin", "frames.jsonl", "imu.csv", "metadata.json"):
            if not (self.root / name).is_file():
                raise FileNotFoundError(f"{self.root} is missing {name}")

        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.frames = [
            json.loads(line)
            for line in (self.root / "frames.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self._blob = self.root / "frames.bin"

    def __len__(self):
        return len(self.frames)

    @property
    def is_complete(self):
        """metadata.json is written last, so its presence marks a clean finish."""
        return (self.root / "metadata.json").is_file()

    @property
    def extension(self):
        return self.metadata.get("config", {}).get("encoding", "JPEG").lower().replace("jpeg", "jpg")

    def frame_bytes(self, i):
        """Encoded bytes for frame i, exactly as the device wrote them."""
        rec = self.frames[i]
        with open(self._blob, "rb") as fh:
            fh.seek(rec["offset"])
            return fh.read(rec["length"])

    def iter_frame_bytes(self):
        """Streams every frame with one open handle - use this in converters."""
        with open(self._blob, "rb") as fh:
            for rec in self.frames:
                fh.seek(rec["offset"])
                yield rec, fh.read(rec["length"])

    def frame_timestamps(self):
        """(N,) int64 on the IMU clock base."""
        return np.array([r["ts_ns"] for r in self.frames], dtype=np.int64)

    def frame_timestamps_raw(self):
        """(N,) int64 as the camera reported them, before correction."""
        return np.array([r["ts_raw_ns"] for r in self.frames], dtype=np.int64)

    def imu(self):
        """(M,7) float64: timestamp_ns, gx, gy, gz, ax, ay, az."""
        return np.loadtxt(self.root / "imu.csv", delimiter=",", comments="#", ndmin=2)

    @property
    def is_color(self):
        """True when the recording kept its chroma."""
        cfg = self.metadata.get("config", {})
        if "color_mode" in cfg:
            return cfg["color_mode"] == "COLOR"
        return not cfg.get("grayscale", True)

    def frame_image(self, i, color=None):
        """Decoded frame (RGB HxWx3 or grey HxW). Requires opencv or pillow."""
        data = self.frame_bytes(i)
        want_color = self.is_color if color is None else color
        try:
            import cv2
            if want_color:
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                return bgr[:, :, ::-1]
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        except ImportError:
            from io import BytesIO

            from PIL import Image
            return np.asarray(Image.open(BytesIO(data)).convert("RGB" if want_color else "L"))


if __name__ == "__main__":
    import sys

    s = Session(sys.argv[1])
    ts = s.frame_timestamps()
    dt = np.diff(ts) / 1e9
    imu = s.imu()
    print(f"frames       {len(s)}")
    print(f"duration     {(ts[-1] - ts[0]) / 1e9:.2f} s")
    print(f"rate         {(len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9):.2f} Hz")
    print(f"sorted       {bool(np.all(dt > 0))}")
    print(f"imu samples  {len(imu)}")
    print(f"blob         {(s.root / 'frames.bin').stat().st_size / 1e6:.1f} MB")
