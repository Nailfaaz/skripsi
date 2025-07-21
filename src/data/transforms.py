# preprocessing / augmentation functions go here
from PIL import Image
import numpy as np
from typing import Tuple

def preprocess_image(
    img: Image.Image,
    size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    - convert to grayscale
    - resize with bilinear interpolation
    - normalize pixel values to [0,1]
    """
    img = img.convert("L").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def preprocess_mask(
    mask: Image.Image,
    size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    - convert to grayscale
    - resize with nearest‐neighbor (to preserve labels)
    - binarize to {0,1}
    """
    mask = mask.convert("L").resize(size, Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)

def preprocess_pair(
    img_path: str,
    mask_path: str,
    size: Tuple[int, int] = (224, 224)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image & mask from disk and apply the above transforms.
    Returns (image_array, mask_array).
    """
    img  = Image.open(img_path)
    mask = Image.open(mask_path)
    return (
        preprocess_image(img, size),
        preprocess_mask(mask, size),
    )
