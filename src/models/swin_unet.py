# src/models/swin_unet.py
import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import swin_t

class SwinUNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(SwinUNet, self).__init__()
        self.backbone = swin_t(pretrained=pretrained)
        self.backbone.features = nn.Identity()  # disable final classifier

        # Extractor to get intermediate layers (optional; use hooks or forward hooks in real use)
        self.encoder = nn.Sequential(
            self.backbone.features[0],  # patch embed
            self.backbone.features[1],  # stage 1
            self.backbone.features[2],  # stage 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        x = features[-1]  # use last layer
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        out = self.decoder(x)
        return out