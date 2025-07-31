# path: data/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NpySegmentationDataset(Dataset):
    """
    PyTorch Dataset for .npy-based segmentation data.
    Expects:
      root/<dataset>/<split>/images/*.npy  → images in CHW float32 (normalized) or HWC uint8
      root/<dataset>/<split>/masks/*.npy   → masks in H×W uint8 (0 or 1)
    Returns:
      img:  torch.FloatTensor of shape [3, H, W]
      mask: torch.LongTensor of shape [H, W]
    """
    def __init__(self, cfg, split='train'):
        self.root    = Path(cfg['data']['root'])
        self.ds_name = cfg['data']['dataset']
        self.split   = split

        imgs_dir  = self.root / self.ds_name / split / 'images'
        masks_dir = self.root / self.ds_name / split / 'masks'

        self.img_paths  = sorted(imgs_dir.glob('*.npy'))
        self.mask_paths = sorted(masks_dir.glob('*.npy'))

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.img_paths)} images vs {len(self.mask_paths)} masks"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image array
        arr = np.load(self.img_paths[idx])
        if arr.ndim == 3 and arr.shape[0] == 3:
            # Already CHW
            img = torch.from_numpy(arr).float()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # HWC → CHW
            img = torch.from_numpy(arr).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape {arr.shape} at index {idx}")

        # Load mask array (H×W, values 0/1)
        m_arr = np.load(self.mask_paths[idx])
        if m_arr.ndim != 2:
            raise ValueError(f"Unexpected mask shape {m_arr.shape} at index {idx}")
        mask = torch.from_numpy(m_arr).long()

        return img, mask
