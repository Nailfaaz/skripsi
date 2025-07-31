# path: data/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NpySegmentationDataset(Dataset):
    """
    Expects:
      root/<dataset>/<split>/{images,masks}/*.npy
    Returns (imgTensor, maskTensor), where
      imgTensor: float32, [3,H,W] in [0,1]
      maskTensor: int64, [H,W] with values {0,1}
    """
    def __init__(self, cfg, split='train'):
        self.root    = Path(cfg['data']['root'])
        self.ds_name = cfg['data']['dataset']
        self.split   = split
        imgs_dir = self.root/self.ds_name/split/'images'
        masks_dir= self.root/self.ds_name/split/'masks'
        self.imgs  = sorted(imgs_dir.glob('*.npy'))
        self.masks = sorted(masks_dir.glob('*.npy'))
        assert len(self.imgs)==len(self.masks), "Images/masks count mismatch"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        arr = np.load(self.imgs[idx])         # H×W×3 uint8
        img = torch.from_numpy(arr).permute(2,0,1).float().div(255.0)
        m   = np.load(self.masks[idx])        # H×W uint8 {0,1}
        mask= torch.from_numpy(m).long()
        return img, mask
