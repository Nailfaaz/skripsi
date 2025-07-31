import sys
import os

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data.loader import NpySegDataset
from src.models.unet import UNet
from src.models.swin_unet import SwinUNet
from src.engine.train import train_epoch, validate_epoch
from src.engine.performance import measure_performance_metrics, get_comprehensive_model_stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model', type=str, default='unet', choices=['unet'])  # Add other models later
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='results/models')
    return parser.parse_args()

def get_model(name):
    if name == 'unet':
        return UNet(n_channels=1, n_classes=1)
    elif name == 'swin':
        return SwinUNet(num_classes=1)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset and loader
    datasets = {split: NpySegDataset(args.data_root, split) for split in ['train', 'val', 'test']}
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == 'train'), num_workers=4) for k, v in datasets.items()}

    # Model
    model = get_model(args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience, min_lr=1e-7)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    best_dice, best_epoch, best_ckpt = 0.0, 0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders['train'], loss_fn, optimizer, device)
        val_metrics = validate_epoch(model, loaders['val'], loss_fn, device)
        scheduler.step(val_metrics['dice'])

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Dice: {val_metrics['dice']:.4f}")

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            best_ckpt = os.path.join(args.out_dir, f'{args.model}_best_epoch{epoch}.pt')
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Saved best model @ epoch {epoch}")

    # Final evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = validate_epoch(model, loaders['test'], loss_fn, device)
    perf_metrics = measure_performance_metrics(model, loaders['test'], device)

    print(f"🎯 Test Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    print(f"⚡ Inference time: {perf_metrics['inference_time_ms']:.2f} ms/sample")

if __name__ == '__main__':
    main()
