# path: scripts/process_data.py

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# ImageNet normalization stats
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def process_dataset(raw_root: Path, processed_root: Path, size: tuple):
    """
    Process segmentation datasets:
      - Resize images to `size` with BILINEAR
      - Normalize images using ImageNet mean/std
      - Resize masks to `size` with NEAREST and binarize (0/1)
      - Save numpy arrays (.npy) under processed_root:
          images → float32, shape (3, H, W)
          masks  → uint8,  shape (H, W)
      - Match masks named exactly or with '_mask' suffix
    """
    for ds in raw_root.iterdir():
        if not ds.is_dir():
            continue
        raw_images = ds / 'images'
        raw_masks  = ds / 'masks'
        if not raw_images.exists() or not raw_masks.exists():
            print(f"Skipping {ds.name}: missing images or masks.")
            continue

        out_img_dir = processed_root / ds.name / 'images'
        out_msk_dir = processed_root / ds.name / 'masks'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_msk_dir.mkdir(parents=True, exist_ok=True)

        for img_path in raw_images.iterdir():
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue

            # --- process image ---
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0       # H×W×3 in [0,1]
            # normalize per-channel
            arr = (arr - MEAN[None, None, :]) / STD[None, None, :]
            # to CHW
            arr = arr.transpose(2, 0, 1)                       # 3×H×W
            out_img_path = out_img_dir / f"{img_path.stem}.npy"
            np.save(out_img_path, arr.astype(np.float32))

            # --- process mask ---
            mask_path = raw_masks / img_path.name
            if not mask_path.exists():
                cand = raw_masks / f"{img_path.stem}_mask{img_path.suffix}"
                if cand.exists():
                    mask_path = cand
                else:
                    print(f"Warning: no mask for {img_path.name}, skipping.")
                    continue

            m = Image.open(mask_path).convert('L')
            m = m.resize(size, Image.NEAREST)
            m_arr = np.array(m, dtype=np.uint8)
            m_bin = (m_arr > 0).astype(np.uint8)                # 0 or 1
            out_msk_path = out_msk_dir / f"{Path(mask_path).stem.replace('_mask','')}.npy"
            np.save(out_msk_path, m_bin)

    print(f"Processing complete! Data saved to {processed_root}")

def main():
    parser = argparse.ArgumentParser(
        description="Process and resize segmentation datasets to normalized .npy format."
    )
    parser.add_argument(
        '--raw_root', type=Path, required=True,
        help='Path to raw datasets (e.g., datasets/raw)'
    )
    parser.add_argument(
        '--processed_root', type=Path, required=True,
        help='Path for processed output (e.g., datasets/processed)'
    )
    parser.add_argument(
        '--height', type=int, default=224,
        help='Output image & mask height (default: 224)'
    )
    parser.add_argument(
        '--width', type=int, default=224,
        help='Output image & mask width  (default: 224)'
    )
    args = parser.parse_args()

    size = (args.width, args.height)
    process_dataset(args.raw_root, args.processed_root, size)

if __name__ == '__main__':
    main()
