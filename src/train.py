# src/train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.data.loader import NpySegDataset
from src.models.unet       import UNet          # or your own class
# from src.models.swin_mobilenet import SwinMobileNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/processed/shenzhen")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    return p.parse_args()

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = loss_fn(preds, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + DataLoader
    ds = NpySegDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model, loss, optimizer
    model     = UNet(in_channels=1, out_channels=1).to(device)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dl, loss_fn, optimizer, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} — Loss: {avg_loss:.4f}")
        # TODO: add checkpoint saving here

if __name__ == "__main__":
    main()
