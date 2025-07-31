# path: models/swin_unet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer

class SwinUNet(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 img_size=224,
                 patch_size=4,
                 embed_dim=96,
                 depths=[2,2,6,2],
                 num_heads=[3,6,12,24],
                 window_size=7,
                 drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        self.img_size   = img_size

        # 1) backbone without head
        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=n_channels,
            num_classes=0,        # strip off classifier
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

        # 2) discover feature shape
        dummy = torch.zeros(1, n_channels, img_size, img_size)
        feats = self.backbone.forward_features(dummy)
        if feats.ndim == 3:           # [B,L,C]
            _, L, C = feats.shape
            side = int(math.sqrt(L))
        elif feats.ndim == 4:         # [B,H,W,C]
            _, side, _, C = feats.shape
        else:
            raise RuntimeError(f"Bad feat shape {feats.shape}")
        self.feature_side     = side
        self.feature_channels = C

        # 3) decoder
        current = C
        decoder_ch = [384,192,96,48]
        self.decoder_layers = nn.ModuleList()
        for tgt in decoder_ch:
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(current, tgt, 2, 2),
                nn.BatchNorm2d(tgt), nn.ReLU(inplace=True),
                nn.Conv2d(tgt, tgt, 3, padding=1),
                nn.BatchNorm2d(tgt), nn.ReLU(inplace=True),
            ))
            current = tgt

        # 4) final upsample (×4 →224)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(current, 24, 4, 4),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.BatchNorm2d(12), nn.ReLU(inplace=True),
        )

        # 5) head
        self.head = nn.Conv2d(12, n_classes, 1)

        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B,_,H,W = x.shape
        if (H,W) != (self.img_size,self.img_size):
            x = F.interpolate(x, size=(self.img_size,self.img_size),
                              mode='bilinear',align_corners=False)

        feats = self.backbone.forward_features(x)
        if feats.ndim == 3:   # [B,L,C]
            Bf,L,Cf = feats.shape
            side = int(math.sqrt(L))
            feats = feats.permute(0,2,1).view(Bf,Cf,side,side)
        else:                  # [B,H,W,C]
            feats = feats.permute(0,3,1,2).contiguous()

        # decoder
        for dec in self.decoder_layers:
            feats = dec(feats)
        feats = self.final_upsample(feats)
        out   = self.head(feats)
        if out.shape[-2:] != (H,W):
            out = F.interpolate(out, size=(H,W), mode='bilinear',align_corners=False)
        return out
