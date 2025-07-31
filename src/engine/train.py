# src/engine/train.py
import torch
import numpy as np
from tqdm import tqdm
from .metrics import dice_coef, iou_coef, precision_recall, hausdorff, surface_distance_metrics

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