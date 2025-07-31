# path: engines/trainer.py

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
import pandas as pd
import numpy as np
from statistics import mean
from pathlib import Path

def compute_iou_binary(preds, masks):
    p = (preds.cpu().numpy() == 1)
    m = (masks.cpu().numpy() == 1)
    inter = (p & m).sum()
    union = (p | m).sum()
    return float(inter / union) if union > 0 else 0.0

def compute_dice_binary(preds, masks):
    p = (preds.cpu().numpy() == 1)
    m = (masks.cpu().numpy() == 1)
    inter = (p & m).sum()
    size  = p.sum() + m.sum()
    return float(2 * inter / size) if size > 0 else 0.0

class BCEDiceLoss(nn.Module):
    """Combination of BCEWithLogits + Dice (for the positive class)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # outputs: [B,2,H,W], masks: [B,H,W] with {0,1}
        # take the logit for class=1
        logits_pos = outputs[:, 1, :, :]                # [B,H,W]
        m = masks.float()                               # [B,H,W]

        # BCE
        bce_loss = self.bce(logits_pos, m)

        # Dice on probabilities
        probs = torch.sigmoid(logits_pos)
        num = 2 * (probs * m).sum(dim=(1,2)) + self.smooth
        den = probs.sum(dim=(1,2)) + m.sum(dim=(1,2)) + self.smooth
        dice_loss = 1 - (num / den)
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


class Trainer:
    def __init__(self, model, cfg):
        # store cfg
        self.cfg = cfg

        # device & model
        self.device = cfg['device']
        self.model  = model.to(self.device)

        # combined loss
        self.criterion = BCEDiceLoss(smooth=1.0)

        # optimizer
        lr = float(cfg['train']['lr'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # training settings
        self.epochs     = int(cfg['train']['epochs'])
        self.batch_size = int(cfg['train']['batch_size'])
        self.save_dir   = Path(cfg['train']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # model complexity (FLOPs & Params)
        dummy = torch.randn(
            1,
            cfg['model'].get('n_channels', 3),
            cfg['data']['height'],
            cfg['data']['width']
        ).to(self.device)
        self.flops, self.params = profile(self.model, inputs=(dummy,), verbose=False)

        # record hyperparameters
        self.hyperparams = {
            'lr':         lr,
            'epochs':     self.epochs,
            'batch_size': self.batch_size
        }

    def train(self, train_ds, val_ds, test_ds=None):
        train_loader = DataLoader(train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_ds,
                                  batch_size=self.batch_size)
        test_loader  = (DataLoader(test_ds,
                                   batch_size=self.batch_size)
                        if test_ds else None)

        records = []
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # ----- TRAIN -----
            self.model.train()
            train_losses = []
            for imgs, masks in tqdm(train_loader,
                                     desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            avg_train = mean(train_losses)

            # ----- VALIDATION -----
            val_loss, val_iou, val_dice, val_time = self.evaluate(val_loader)

            records.append({
                'epoch':      epoch,
                'train_loss': avg_train,
                'val_loss':   val_loss,
                'val_iou':    val_iou,
                'val_dice':   val_dice,
                'val_haus':   0.0,       # still skipping Hausdorff
                'val_time_s': val_time,
            })

            print(
                f"Epoch {epoch}: "
                f"Train L={avg_train:.4f}, "
                f"Val   L={val_loss:.4f}, "
                f"IoU={val_iou:.4f}, "
                f"Dice={val_dice:.4f}, "
                f"Haus=0.0000, "
                f"InfT={val_time:.4f}s"
            )

            # save checkpoint
            torch.save(self.model.state_dict(),
                       self.save_dir / f"unet_epoch{epoch}.pth")

        total_train_time = time.time() - start_time

        # ----- TEST -----
        if test_loader:
            t_loss, t_iou, t_dice, t_time = self.evaluate(test_loader)
            print(
                f"Test: L={t_loss:.4f}, "
                f"IoU={t_iou:.4f}, "
                f"Dice={t_dice:.4f}, "
                f"Haus=0.0000, "
                f"InfT={t_time:.4f}s"
            )
        else:
            t_loss = t_iou = t_dice = t_time = None

        # save metrics
        pd.DataFrame(records).to_csv(self.save_dir / "metrics.csv", index=False)

        # write summary
        summary = {
            'flops':                 int(self.flops),
            'params':                int(self.params),
            'total_training_time_s': total_train_time,
            'hyperparameters':       self.hyperparams,
            'final_train_loss':      records[-1]['train_loss'],
            'final_val_loss':        records[-1]['val_loss'],
            'final_val_iou':         records[-1]['val_iou'],
            'final_val_dice':        records[-1]['val_dice'],
            'final_val_haus':        0.0,
            'val_avg_time_s':        records[-1]['val_time_s'],
            'dataset_root':          str(self.cfg['data']['root']),
            'dataset_name':          self.cfg['data']['dataset'],
        }
        if test_loader:
            summary.update({
                'test_loss':      t_loss,
                'test_iou':       t_iou,
                'test_dice':      t_dice,
                'test_haus':      0.0,
                'test_avg_time_s':t_time,
            })

        with open(self.save_dir / "summary.txt", 'w',
                  encoding='utf-8') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        print(f"All done! Metrics & summary saved under {self.save_dir}")

    def evaluate(self, loader):
        self.model.eval()
        losses, ious, dices, times = [], [], [], []

        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="Evaluating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                t0 = time.time()
                outputs = self.model(imgs)
                times.append(time.time() - t0)

                losses.append(self.criterion(outputs, masks).item())
                preds = outputs.argmax(dim=1)

                ious.append(compute_iou_binary(preds, masks))
                dices.append(compute_dice_binary(preds, masks))

        return (
            mean(losses),
            mean(ious),
            mean(dices),
            mean(times),
        )
