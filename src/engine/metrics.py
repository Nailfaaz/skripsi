# src/engine/metrics.py
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist

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
    p = (pred > 0.5).cpu().numpy().astype(np.uint8)[0,0]
    t = target.cpu().numpy().astype(np.uint8)[0,0]
    coords_p, coords_t = np.column_stack(np.where(p)), np.column_stack(np.where(t))
    if not coords_p.size or not coords_t.size:
        return float("nan"), float("nan")
    distances_p_to_t = cdist(coords_p, coords_t).min(axis=1)
    distances_t_to_p = cdist(coords_t, coords_p).min(axis=1)
    asd = (distances_p_to_t.mean() + distances_t_to_p.mean()) / 2
    hd95 = max(np.percentile(distances_p_to_t, 95), np.percentile(distances_t_to_p, 95))
    return asd, hd95