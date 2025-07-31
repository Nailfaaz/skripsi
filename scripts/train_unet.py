# script/train_unet.py
import os
import sys
import argparse
import tempfile
import csv
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from ptflops import get_model_complexity_info
import psutil

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

def precision_recall(pred, target, eps=1e-6):
    p, t = (pred > 0.5).float(), target.float()
    tp = (p * t).sum(dim=[1,2,3])
    fp = (p * (1-t)).sum(dim=[1,2,3])
    fn = ((1-p) * t).sum(dim=[1,2,3])
    
    precision = (tp / (tp + fp + eps)).mean().item()
    recall = (tp / (tp + fn + eps)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return precision, recall, f1

def hausdorff(pred, target):
    p = (pred > 0.5).cpu().numpy().astype(np.uint8)[0,0]
    t = target.cpu().numpy().astype(np.uint8)[0,0]
    coords_p, coords_t = np.column_stack(np.where(p)), np.column_stack(np.where(t))
    if coords_p.size and coords_t.size:
        return max(directed_hausdorff(coords_p, coords_t)[0], directed_hausdorff(coords_t, coords_p)[0])
    return float("nan")

def surface_distance_metrics(pred, target):
    """Calculate average surface distance and 95% Hausdorff distance"""
    p = (pred > 0.5).cpu().numpy().astype(np.uint8)[0,0]
    t = target.cpu().numpy().astype(np.uint8)[0,0]
    coords_p, coords_t = np.column_stack(np.where(p)), np.column_stack(np.where(t))
    
    if not coords_p.size or not coords_t.size:
        return float("nan"), float("nan")
    
    # Calculate all distances
    from scipy.spatial.distance import cdist
    distances_p_to_t = cdist(coords_p, coords_t).min(axis=1)
    distances_t_to_p = cdist(coords_t, coords_p).min(axis=1)
    
    # Average surface distance
    asd = (distances_p_to_t.mean() + distances_t_to_p.mean()) / 2
    
    # 95% Hausdorff distance
    hd95 = max(np.percentile(distances_p_to_t, 95), np.percentile(distances_t_to_p, 95))
    
    return asd, hd95

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
    metrics = {'loss': [], 'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 
               'hd': [], 'asd': [], 'hd95': []}
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=" Val "):
            imgs, masks = imgs.to(device), masks.to(device).float()
            logits = model(imgs)
            preds = torch.sigmoid(logits)
            
            metrics['loss'].append(loss_fn(logits, masks).item())
            metrics['dice'].append(dice_coef(preds, masks))
            metrics['iou'].append(iou_coef(preds, masks))
            
            prec, rec, f1 = precision_recall(preds, masks)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1'].append(f1)
            
            metrics['hd'].append(hausdorff(preds, masks))
            asd, hd95 = surface_distance_metrics(preds, masks)
            metrics['asd'].append(asd)
            metrics['hd95'].append(hd95)
    
    return {k: np.nanmean(v) for k, v in metrics.items()}

def measure_performance_metrics(model, loader, device, n_samples=100):
    """Measure inference time, memory usage, and throughput"""
    model.eval()
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            if len(times) >= n_samples: break
            imgs = imgs.to(device)
            
            # Warmup on first batch
            if not times:
                [model(imgs) for _ in range(10)]
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Memory before inference
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated()
            
            start = time.time()
            output = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            # Memory after inference
            if device.type == 'cuda':
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / (1024**2))  # MB
            
            batch_time = (end - start) / imgs.size(0)
            times.extend([batch_time] * imgs.size(0))
    
    avg_time_ms = np.mean(times[:n_samples]) * 1000
    throughput = 1000 / avg_time_ms
    avg_memory_mb = np.mean(memory_usage) if memory_usage else 0
    
    return {
        'inference_time_ms': avg_time_ms,
        'throughput_samples_per_sec': throughput,
        'memory_per_sample_mb': avg_memory_mb
    }

def get_comprehensive_model_stats(model, input_size=(1,224,224)):
    """Get comprehensive model statistics"""
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Model size on disk
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name) / (1024*1024)
    tmp.close(); os.remove(tmp.name)
    
    # FLOPs and MACs
    flops, params_str = get_model_complexity_info(model, input_size, as_strings=(True, False), 
                                                  print_per_layer_stat=False, verbose=False)
    
    # Layer information
    layer_count = len(list(model.modules()))
    conv_layers = len([m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))])
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': size_mb,
        'flops': flops[0] if isinstance(flops, tuple) else flops,
        'flops_human': flops[1] if isinstance(flops, tuple) else params_str,
        'total_layers': layer_count,
        'conv_layers': conv_layers,
        'parameters_human': params_str
    }

def save_comprehensive_results(out_dir, model_stats, train_metrics, val_metrics, test_metrics, 
                             performance_metrics, training_config, best_epoch):
    """Save comprehensive results in multiple formats"""
    
    # Create comprehensive dictionary
    results = {
        'model_architecture': {
            'name': 'UNet',
            'input_channels': 1,
            'output_channels': 1,
            **model_stats
        },
        'training_configuration': training_config,
        'training_results': {
            'best_epoch': best_epoch,
            'total_epochs': len(train_metrics),
            'final_train_loss': train_metrics[-1] if train_metrics else None
        },
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'performance_metrics': performance_metrics,
        'system_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    }
    
    # Save JSON for programmatic access
    with open(f"{out_dir}/results_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed text summary
    summary = f"""U-Net Segmentation Model - Comprehensive Evaluation Report
{'='*70}

MODEL ARCHITECTURE:
- Total Parameters: {model_stats['total_parameters']:,}
- Trainable Parameters: {model_stats['trainable_parameters']:,}
- Non-trainable Parameters: {model_stats['non_trainable_parameters']:,}
- Model Size: {model_stats['model_size_mb']:.2f} MB
- FLOPs: {model_stats['flops_human']}
- Total Layers: {model_stats['total_layers']}
- Convolutional Layers: {model_stats['conv_layers']}

TRAINING CONFIGURATION:
- Epochs: {training_config['epochs']}
- Batch Size: {training_config['batch_size']}
- Learning Rate: {training_config['lr']}
- Weight Decay: {training_config['weight_decay']}
- Optimizer: AdamW
- Loss Function: BCEWithLogitsLoss
- Best Epoch: {best_epoch}

SEGMENTATION METRICS:
- Dice Coefficient: {test_metrics['dice']:.4f}
- IoU (Jaccard): {test_metrics['iou']:.4f}
- Precision: {test_metrics['precision']:.4f}
- Recall (Sensitivity): {test_metrics['recall']:.4f}
- F1-Score: {test_metrics['f1']:.4f}

DISTANCE METRICS:
- Hausdorff Distance: {test_metrics['hd']:.2f} pixels
- Average Surface Distance: {test_metrics['asd']:.2f} pixels
- 95% Hausdorff Distance: {test_metrics['hd95']:.2f} pixels

PERFORMANCE METRICS:
- Inference Time: {performance_metrics['inference_time_ms']:.2f} ms/sample
- Throughput: {performance_metrics['throughput_samples_per_sec']:.1f} samples/sec
- Memory Usage: {performance_metrics['memory_per_sample_mb']:.2f} MB/sample

SYSTEM INFORMATION:
- PyTorch Version: {torch.__version__}
- CUDA Available: {torch.cuda.is_available()}
- CPU Cores: {psutil.cpu_count()}
- Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
"""
    
    with open(f"{out_dir}/evaluation_report.txt", "w") as f:
        f.write(summary)
    
    return results

def parse_args():
    p = argparse.ArgumentParser(description="Train & eval U-Net segmentation with comprehensive metrics")
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

    # Get model statistics
    model_stats = get_comprehensive_model_stats(model)
    print(f"\n🏗️  MODEL: {model_stats['total_parameters']:,} params, {model_stats['model_size_mb']:.2f}MB")
    print(f"🔧 CONFIG: AdamW(lr={args.lr}, wd={args.weight_decay}), batch={args.batch_size}")

    # Training configuration for logging
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'optimizer': 'AdamW',
        'loss_function': 'BCEWithLogitsLoss',
        'scheduler': 'ReduceLROnPlateau'
    }

    # Training loop
    train_losses = []
    val_metrics_history = []
    best_dice, best_epoch, best_ckpt = 0.0, 0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders['train'], loss_fn, optimizer, device)
        val_metrics = validate_epoch(model, loaders['val'], loss_fn, device)
        
        scheduler.step(val_metrics['dice'])
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d} | Train: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} "
              f"Dice: {val_metrics['dice']:.4f} IoU: {val_metrics['iou']:.4f} F1: {val_metrics['f1']:.4f}")
        
        # Log metrics
        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            best_ckpt = f"{args.out_dir}/unet_best_epoch{epoch}.pt"
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 New best: Dice {best_dice:.4f}")

    # Final evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = validate_epoch(model, loaders['test'], loss_fn, device)
    performance_metrics = measure_performance_metrics(model, loaders['test'], device)
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Dice: {test_metrics['dice']:.4f} | IoU: {test_metrics['iou']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}")
    print(f"   Inference: {performance_metrics['inference_time_ms']:.2f} ms/sample")
    
    # Save comprehensive results
    results = save_comprehensive_results(
        args.out_dir, model_stats, train_losses, val_metrics_history[-1], 
        test_metrics, performance_metrics, training_config, best_epoch
    )
    
    print(f"\n✅ Comprehensive results saved to: {args.out_dir}/")
    print(f"   - JSON: results_comprehensive.json")
    print(f"   - Report: evaluation_report.txt")

if __name__ == "__main__":
    main()