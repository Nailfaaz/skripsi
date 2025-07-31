# path: scripts/split_data.py
import argparse
import random
import shutil
from pathlib import Path


def split_dataset(processed_root: Path, output_root: Path, train_ratio: float, val_ratio: float, seed: int = 42):
    """
    Splits each dataset under `processed_root` containing .npy images and masks into train/val/test.
    Expects structure:
      processed_root/<dataset>/{images, masks}/*.npy
    Copies to:
      output_root/<dataset>/<split>/{images, masks}/*.npy
    """
    random.seed(seed)

    for ds in processed_root.iterdir():
        if not ds.is_dir():
            continue
        imgs_dir = ds / "images"
        masks_dir = ds / "masks"
        if not imgs_dir.exists() or not masks_dir.exists():
            print(f"Skipping {ds.name}: missing images or masks directory.")
            continue

        # list all .npy image files
        files = [p.name for p in imgs_dir.glob('*.npy')]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        for split, fnames in splits.items():
            out_img_dir = output_root / ds.name / split / 'images'
            out_msk_dir = output_root / ds.name / split / 'masks'
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_msk_dir.mkdir(parents=True, exist_ok=True)

            for fname in fnames:
                img_src = imgs_dir / fname
                mask_src = masks_dir / fname
                if not mask_src.exists():
                    stem = Path(fname).stem
                    candidate = masks_dir / f"{stem}_mask.npy"
                    if candidate.exists():
                        mask_src = candidate
                    else:
                        raise FileNotFoundError(f"Mask for {fname} not found in {masks_dir}")

                shutil.copy(img_src, out_img_dir / fname)
                shutil.copy(mask_src, out_msk_dir / mask_src.name)

    print(f"Split complete: data saved to {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Split processed .npy segmentation data into train/val/test.")
    parser.add_argument('--processed_root', required=True, type=Path,
                        help='Path to processed .npy data root')
    parser.add_argument('--output_root', required=True, type=Path,
                        help='Path where split data will be saved')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    split_dataset(
        processed_root=args.processed_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
