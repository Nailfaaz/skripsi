#!/usr/bin/env python3
import os, shutil, random

def split_dataset(base: str, splits=(0.7,0.15,0.15), seed=42):
    """
    base: e.g. "data/processed/shenzhen"
    splits: fractions for train/val/test (must sum to 1)
    """
    random.seed(seed)
    imgs = sorted(os.listdir(os.path.join(base, "images")))
    total = len(imgs)

    # compute cut indices
    n_train = int(total * splits[0])
    n_val   = int(total * (splits[0] + splits[1]))
    perm    = random.sample(imgs, total)

    assignments = {
        "train": perm[:n_train],
        "val":   perm[n_train:n_val],
        "test":  perm[n_val:],
    }

    for split, files in assignments.items():
        for kind in ("images","masks"):
            out_dir = os.path.join(base, split, kind)
            os.makedirs(out_dir, exist_ok=True)
        for img in files:
            base_name = os.path.splitext(img)[0]
            # move image
            src_img = os.path.join(base, "images", img)
            dst_img = os.path.join(base, split, "images", img)
            shutil.move(src_img, dst_img)
            # move mask (with _mask suffix)
            mask_name = f"{base_name}_mask.npy"
            src_mask = os.path.join(base, "masks", mask_name)
            dst_mask = os.path.join(base, split, "masks", mask_name)
            shutil.move(src_mask, dst_mask)

    # remove now-empty folders
    for d in ("images","masks"):
        os.rmdir(os.path.join(base, d))

if __name__ == "__main__":
    for ds in ("shenzhen","montgomery"):
        split_dataset(os.path.join("data","processed",ds))
        print(f"Split done for {ds}")
