import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NpySegDataset(Dataset):
    """
    Loads preprocessed .npy image/mask pairs for segmentation,
    from a specific split (train, val, or test).
    """
    def __init__(self, root_dir: str, split: str = "train"):
        """
        root_dir: path to e.g. "data/processed/shenzhen"
        split:    one of "train", "val", or "test"
        """
        # point at the right subfolders
        self.img_dir  = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        # collect all base names (without .npy)
        self.ids = [fname[:-4] 
                    for fname in os.listdir(self.img_dir) 
                    if fname.endswith(".npy")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base = self.ids[idx]
        img_path  = os.path.join(self.img_dir,  base + ".npy")
        mask_path = os.path.join(self.mask_dir, base + "_mask.npy")

        # load
        img = np.load(img_path).astype(np.float32)
        m   = np.load(mask_path).astype(np.uint8)

        # to torch tensors with channel dim
        img_t  = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        mask_t = torch.from_numpy(m).unsqueeze(0)    # [1, H, W]
        return img_t, mask_t
