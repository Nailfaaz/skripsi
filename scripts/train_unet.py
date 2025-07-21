#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─── PATH HACK ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from ptflops import get_model_complexity_info  # requires `pip install ptflops`

from src.data.loader import NpySegDataset
from src.models.unet   import UNet

def dice_coef(pred, target, eps=1e-6):
    p = (pred > 0.5).float()
    t = target.float()
    inter = (p * t).sum(dim=[1,2,3])
    union = p.sum(dim=[1,2,3]) + t.sum(dim=[1,2,3])
    return ((2*inter + eps)/(union + eps)).mean().item()

def iou_coef(pred, target, eps=1e-6):
    p = (pred > 0.5).float()
    t = target.float()
    inter = (p * t).sum(dim=[1,2,3])
    union = (p + t - p*t).sum(dim=[1,2,3])
    return (inter/(union + eps)).mean().item()

def hausdorff(pred, target):
    p = (pred > 0.5).cpu().numpy().astype(np.uint8)[0,0]
    t = target.cpu().numpy().astype(np.uint8)[0,0]
    coords_p = np.column_stack(np.where(p))
    coords_t = np.column_stack(np.where(t))
    if coords_p.size and coords_t.size:
        d1 = directed_hausdorff(coords_p, coords_t)[0]
        d2 = directed_hausdorff(coords_t, coords_p)[0]
        return max(d1, d2)
    return float("nan")

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train", unit="batch"):
        imgs, masks = imgs.to(device), masks.to(device).float()
        preds = model(imgs)
        loss  = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    total_hd   = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=" Val ", unit="batch"):
            imgs, masks = imgs.to(device), masks.to(device).float()
            preds = torch.sigmoid(model(imgs))
            total_loss += loss_fn(torch.logit(preds, eps=1e-6), masks).item()
            total_dice += dice_coef(preds, masks)
            total_iou  += iou_coef(preds, masks)
            total_hd   += hausdorff(preds, masks)
    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n, total_hd/n

def print_model_stats(model, input_size=(1,224,224)):
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {params:,}")
    print(f"Trainable params: {trainable:,}")

    # on‑disk size
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name)/(1024*1024)
    tmp.close(); os.remove(tmp.name)
    print(f"Model size:       {size_mb:.2f} MB")

    # FLOPs
    flops, params_str = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=False
    )
    print(f"FLOPs (MACs):     {flops}")
    print(f"Params (human):   {params_str}")

def parse_args():
    p = argparse.ArgumentParser(description="Train & eval U‑Net segmentation")
    p.add_argument("--data_root",  required=True,
                   help="path to data/processed/<dataset> (with train/val/test subfolders)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--out_dir",    type=str, default="results/models")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # prepare output
    os.makedirs(args.out_dir, exist_ok=True)

    # datasets & loaders
    train_ds = NpySegDataset(args.data_root, split="train")
    val_ds   = NpySegDataset(args.data_root, split="val")
    test_ds  = NpySegDataset(args.data_root, split="test")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model, loss, optim
    model     = UNet(n_channels=1, n_classes=1).to(device)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\n====== Model Complexity ======")
    print_model_stats(model, input_size=(1,224,224))

    best_dice = 0.0
    best_ckpt = None

    for epoch in range(1, args.epochs+1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_dice, val_iou, val_hd = validate_epoch(model, val_dl, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | HD: {val_hd:.2f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_ckpt = os.path.join(args.out_dir, f"unet_best_epoch{epoch}.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"--> Saved best model to {best_ckpt}")

    # final test
    print("\n====== Testing on split 'test' ======")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_dice, test_iou, test_hd = validate_epoch(model, test_dl, loss_fn, device)
    print(f"Test    Dice: {test_dice:.4f} | IoU: {test_iou:.4f} | HD: {test_hd:.2f}")

if __name__ == "__main__":
    main()
