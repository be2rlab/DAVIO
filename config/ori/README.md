# ORI (ScaRF-SLAM release) configuration

Restored 2026-09-13 after `config/` was deleted from the working tree; the OpenVINS files
were recovered verbatim from `runs/ori/r01__davio__s0/native_config`, `downloads.json` from
`data/ori/*/download_manifest.json`.

- `estimator_config.yaml`, `kalibr_imu_chain.yaml`, `kalibr_imucam_chain.yaml`: the ScaRF
  release's own OpenVINS calibration (commit e62a4a3d): equidistant intrinsics, `T_imu_cam`,
  IMU noise, `timeshift_cam_imu -4.0 ms`. Never EuRoC values.
- `downloads.json`: official pinned Google Drive ids and archive sizes for R01..R05.
- Acquire: `python3 scripts/fetch_ori.py --sequences all --plan`, then `--sequences r01`.
  Convert once: `python3 scripts/convert_ori.py`. Runs are declared unpaced (`--rate 0`).
- Ground truth (`rXX_gt/poses_gt.txt`, TUM camera poses) is read by evaluation only.
