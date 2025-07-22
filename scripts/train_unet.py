#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from ptflops import get_model_complexity_info

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import NpySegDataset
from src.models.unet import UNet

def dice_coef(pred, target, eps=1e-6):
    p, t = (pred > 0.5).float(), target.float()
    inter = (p * t).sum(dim=[1,2,3])
    union = p.sum(dim=[1,2,3]) + t.sum(dim=[1,2,3])
    return ((2*inter + eps)/(union + eps)).mean().item()

def iou_coef(pred, target, eps=1e-6):
    p, t = (pred > 0.5).float(), target.float()
    inter = (p * t).sum(dim=[1,2,3])
    union = (p + t - p * t).sum(dim=[1,2,3])
    return (inter/(union + eps)).mean().item()

def hausdorff(pred, target):
    p = (pred > 0.5).cpu().numpy().astype(np.uint8)[0,0]
    t = target.cpu().numpy().astype(np.uint8)[0,0]
    coords_p, coords_t = np.column_stack(np.where(p)), np.column_stack(np.where(t))
    if coords_p.size and coords_t.size:
        return max(directed_hausdorff(coords_p, coords_t)[0], directed_hausdorff(coords_t, coords_p)[0])
    return float("nan")

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(device), masks.to(device).float()
        loss = loss_fn(model(imgs), masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    metrics = {'loss': [], 'dice': [], 'iou': [], 'hd': []}
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=" Val "):
            imgs, masks = imgs.to(device), masks.to(device).float()
            logits = model(imgs)
            preds = torch.sigmoid(logits)
            
            metrics['loss'].append(loss_fn(logits, masks).item())
            metrics['dice'].append(dice_coef(preds, masks))
            metrics['iou'].append(iou_coef(preds, masks))
            metrics['hd'].append(hausdorff(preds, masks))
    
    return [np.mean(metrics[k]) for k in ['loss', 'dice', 'iou', 'hd']]

def measure_inference_time(model, loader, device, n_samples=100):
    model.eval()
    times = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            if len(times) >= n_samples: break
            imgs = imgs.to(device)
            
            # Warmup on first batch
            if not times:
                [model(imgs) for _ in range(10)]
                torch.cuda.synchronize() if device.type == 'cuda' else None
            
            start = time.time()
            model(imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.extend([(time.time() - start) / imgs.size(0)] * imgs.size(0))
    
    return np.mean(times[:n_samples]) * 1000

def get_model_stats(model, input_size=(1,224,224)):
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name) / (1024*1024)
    tmp.close(); os.remove(tmp.name)
    
    # FLOPs
    flops, params_str = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    
    return {'params': params, 'trainable': trainable, 'size_mb': size_mb, 'flops': flops, 'params_str': params_str}

def print_recap(stats, metrics, inference_ms):
    print(f"\n{'='*60}\n                    FINAL EVALUATION RECAP\n{'='*60}")
    
    print(f"\n📊 MODEL COMPLEXITY:\n   Parameters:    {stats['params']:,} ({stats['trainable']:,} trainable)")
    print(f"   Model Size:    {stats['size_mb']:.2f} MB\n   FLOPs:         {stats['flops']}\n   Human Params:  {stats['params_str']}")
    
    print(f"\n🎯 PERFORMANCE:\n   Dice:          {metrics['dice']:.4f}")
    print(f"   IoU:           {metrics['iou']:.4f}\n   Hausdorff:     {metrics['hausdorff']:.2f} pixels")
    
    print(f"\n⚡ INFERENCE:\n   Time/Sample:   {inference_ms:.2f} ms")
    print(f"   Throughput:    {1000/inference_ms:.1f} samples/sec")
    
    # Performance grades
    grades = {
        'dice': 'Excellent' if metrics['dice'] > 0.8 else 'Good' if metrics['dice'] > 0.7 else 'Fair' if metrics['dice'] > 0.6 else 'Poor',
        'iou': 'Excellent' if metrics['iou'] > 0.7 else 'Good' if metrics['iou'] > 0.6 else 'Fair' if metrics['iou'] > 0.5 else 'Poor',
        'efficiency': 'Lightweight' if stats['params'] < 1000000 else 'Standard',
        'speed': 'Fast' if inference_ms < 50 else 'Standard' if inference_ms < 200 else 'Slow'
    }
    
    print(f"\n📈 GRADES:\n   Dice: {grades['dice']} | IoU: {grades['iou']} | Size: {grades['efficiency']} | Speed: {grades['speed']}")
    print("="*60)

def save_summary(out_dir, stats, metrics, inference_ms):
    summary = f"""U-Net Segmentation - Final Evaluation
{'='*50}

MODEL STATS:
- Parameters: {stats['params']:,} ({stats['trainable']:,} trainable)
- Size: {stats['size_mb']:.2f} MB
- FLOPs: {stats['flops']}

PERFORMANCE:
- Dice: {metrics['dice']:.4f}
- IoU: {metrics['iou']:.4f}  
- Hausdorff: {metrics['hausdorff']:.2f} pixels
- Inference: {inference_ms:.2f} ms/sample ({1000/inference_ms:.1f} samples/sec)
"""
    with open(f"{out_dir}/evaluation_summary.txt", "w") as f:
        f.write(summary)

def parse_args():
    p = argparse.ArgumentParser(description="Train & eval U-Net segmentation")
    p.add_argument("--data_root", required=True, help="Dataset path with train/val/test folders")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="results/models")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data loaders
    datasets = {split: NpySegDataset(args.data_root, split=split) for split in ['train', 'val', 'test']}
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k=='train'), num_workers=4) 
               for k, v in datasets.items()}

    # Model setup
    model = UNet(n_channels=1, n_classes=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience, min_lr=1e-7)

    # Print initial stats
    stats = get_model_stats(model)
    print(f"\n🏗️  MODEL: {stats['params']:,} params, {stats['size_mb']:.2f}MB, {stats['flops']}")
    print(f"🔧 CONFIG: AdamW(lr={args.lr}, wd={args.weight_decay}), batch={args.batch_size}, patience={args.patience}")

    # Training loop
    metrics_log = []
    best_dice, best_ckpt = 0.0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders['train'], loss_fn, optimizer, device)
        val_loss, val_dice, val_iou, val_hd = validate_epoch(model, loaders['val'], loss_fn, device)
        
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
              f"Dice: {val_dice:.4f} IoU: {val_iou:.4f} HD: {val_hd:.2f} LR: {current_lr:.1e}")
        
        # Log metrics
        metrics_log.append([epoch, train_loss, val_loss, val_dice, val_iou, val_hd, current_lr])
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_ckpt = f"{args.out_dir}/unet_best_epoch{epoch}.pt"
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 New best: {best_ckpt}")

    # Final evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_dice, test_iou, test_hd = validate_epoch(model, loaders['test'], loss_fn, device)
    inference_ms = measure_inference_time(model, loaders['test'], device)
    
    print(f"\n🎯 TEST RESULTS: Dice: {test_dice:.4f} | IoU: {test_iou:.4f} | HD: {test_hd:.2f}")
    
    # Save results
    with open(f"{args.out_dir}/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_hd", "lr"])
        writer.writerows(metrics_log)
        writer.writerow(["test", "", "", test_dice, test_iou, test_hd, ""])
        writer.writerow(["inference_ms", "", "", "", "", "", inference_ms])

    # Final recap and summary
    test_metrics = {'dice': test_dice, 'iou': test_iou, 'hausdorff': test_hd}
    print_recap(stats, test_metrics, inference_ms)
    save_summary(args.out_dir, stats, test_metrics, inference_ms)
    
    print(f"\n✅ Results saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()