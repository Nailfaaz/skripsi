# Multi-Organ Chest X-Ray Segmentation (undergraduate thesis, UGM)

Segmenting the **left lung, right lung and heart** in chest X-rays, comparing encoder-decoder
architectures and lightweight encoders. This work led to the paper *"Systematic Evaluation of CRF-as-RNN
for Multi-Organ Chest X-Ray Segmentation"* (ICTIS 2026, Springer LNNS, accepted).

> Work in progress: this repository holds the baseline stage. The CRF-as-RNN refinement experiments are
> not uploaded yet.

## Baseline experiment

Notebook: [`notebooks/01_baseline_unet_deeplabv3plus.ipynb`](notebooks/01_baseline_unet_deeplabv3plus.ipynb)
(run on Kaggle GPU, training logs kept in the outputs).

| Setting | Value |
|---|---|
| Data | JSRT chest X-rays with organ masks, resized to 224 × 224; fixed train/val split (`splits.json`) |
| Classes | 4: background, left lung, right lung, heart |
| Architectures | U-Net, DeepLabV3+ (`segmentation_models_pytorch`) |
| Encoders | MobileNetV3 small / small-minimal / large / large-minimal, ImageNet weights |
| Loss | 0.5 × Dice + 0.5 × cross-entropy |
| Training | AdamW, up to 50 epochs, ReduceLROnPlateau, early stopping |
| Repeats | 3 seeds (42, 1337, 2025) per configuration, 24 runs in total |

### Result: best validation loss (mean ± std over 3 seeds, lower is better)

| Encoder | U-Net | DeepLabV3+ |
|---|---:|---:|
| MobileNetV3-small-minimal | 0.0716 ± 0.0042 | 0.0682 ± 0.0011 |
| MobileNetV3-small | 0.0717 ± 0.0015 | 0.0653 ± 0.0017 |
| MobileNetV3-large-minimal | 0.0685 ± 0.0013 | 0.0643 ± 0.0008 |
| MobileNetV3-large | 0.0654 ± 0.0018 | **0.0614 ± 0.0012** |

DeepLabV3+ beats U-Net with every encoder, and the larger encoders help both. Each run takes about
1.2–1.9 minutes on a Kaggle GPU. Dice/IoU per organ on the test set are the next step.

## Earlier code (`models/`, `engines/`, `scripts/`, `configs/`)

A first, script-based version of the pipeline for binary lung segmentation (Shenzhen dataset), with U-Net,
Swin-UNet and Mobile-UNETR model definitions, preprocessing (`scripts/process_data.py`) and data
splitting (`scripts/split_data.py`). The notebook above supersedes it.

## Data

JSRT images are not redistributed here; request them from the Japanese Society of Radiological Technology.
