from pathlib import Path

def open_dataset(name, root, seq, groundtruth='dataset'):
    """`groundtruth` selects the reference variant; see data/groundtruth.py."""
    if name == 'euroc':
        from .euroc import EurocDataset
        return EurocDataset(Path(root)/seq/'mav0', seq=seq, groundtruth=groundtruth)
    if name == 'ori':
        from .ori import OriDataset
        return OriDataset(Path(root)/seq, seq=seq, groundtruth=groundtruth)
    if name in ('custom', 'phone'):
        # Any EuRoC/ASL-layout recording with its calibration beside it (Kalibr, Basalt,
        # or the recorder's own); 'phone' is the older name for the same adapter.
        from .phone import PhoneDataset
        return PhoneDataset(Path(root)/seq, seq=seq, groundtruth=groundtruth)
    if name == 'vcu_rvi':
        from .vcu_rvi import VcuRviDataset
        return VcuRviDataset(Path(root)/seq, seq=seq, groundtruth=groundtruth)
    if name == 'realsense':
        from .realsense import RealSenseDataset
        return RealSenseDataset(Path(root)/seq, seq=seq, groundtruth=groundtruth)
    if name == 'tumvi':
        from .tumvi import TumViDataset
        if groundtruth not in ('dataset', 'auto'):
            raise ValueError('Only the shipped reference exists for TUM-VI')
        return TumViDataset(Path(root)/seq, seq=seq)
    raise ValueError('Unknown dataset')
