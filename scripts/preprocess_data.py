#!/usr/bin/env python3

import os, sys
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.data.transforms import preprocess_pair

def preprocess_dataset(
    raw_root: str,
    proc_root: str,
    size: tuple[int, int] = (224, 224)
):
    img_dir   = os.path.join(raw_root,   "images")
    mask_dir  = os.path.join(raw_root,   "masks")
    out_img   = os.path.join(proc_root,  "images")
    out_mask  = os.path.join(proc_root,  "masks")

    os.makedirs(out_img,  exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base     = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir,  fname)
        mask_path= os.path.join(mask_dir, f"{base}_mask.png")

        img_np, mask_np = preprocess_pair(img_path, mask_path, size)

        np.save(os.path.join(out_img,  f"{base}.npy"),       img_np)
        np.save(os.path.join(out_mask, f"{base}_mask.npy"),  mask_np)

    print(f"✔ Processed {raw_root} → {proc_root} ({len(os.listdir(img_dir))} samples)")

if __name__ == "__main__":
    for ds in ("shenzhen", "montgomery"):
        raw  = os.path.join("data/raw",       ds)
        proc = os.path.join("data/processed", ds)
        preprocess_dataset(raw, proc, size=(224,224))
