# path: models/mobile_unetr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

class ConvBlock(nn.Module):
    """Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsample + optional skip connection + 2× ConvBlock"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MobileUNetR(nn.Module):
    """
    MobileNetV3→U-Net-like refiner.
    model_size: 'large' or 'small'
    """
    def __init__(self,
                 n_channels: int = 3,
                 n_classes:  int = 2,
                 model_size: str = 'large',
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        # 1) backbone
        if model_size == 'large':
            backbone = mobilenet_v3_large(pretrained=pretrained)
        else:
            backbone = mobilenet_v3_small(pretrained=pretrained)
        backbone.classifier = nn.Identity()
        # adapt first conv if needed
        if n_channels != 3:
            old = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                n_channels, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False
            )
        self.backbone = backbone

        # 2) find skip connections & channel sizes
        self.feature_indices = []
        self.enc_channels    = []
        dummy = torch.randn(1, n_channels, 224, 224)
        x = dummy
        with torch.no_grad():
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                # pick a few depths for skip connections
                if model_size == 'large':
                    key_idxs = [0, 2, 4, 7, len(self.backbone.features) - 1]
                else:
                    key_idxs = [0, 1, 3, 6, len(self.backbone.features) - 1]
                if i in key_idxs:
                    self.feature_indices.append(i)
                    self.enc_channels.append(x.shape[1])

        # 3) bottleneck
        bottleneck_ch = self.enc_channels[-1]
        decoder_ch    = [256, 128, 64, 32, 16]
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_ch, decoder_ch[0]),
            nn.Dropout2d(dropout_rate),
            ConvBlock(decoder_ch[0], decoder_ch[0])
        )

        # 4) decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_ch) - 1):
            skip_ch = self.enc_channels[-2 - i] if i < len(self.enc_channels) - 1 else 0
            self.decoder_blocks.append(
                UpBlock(decoder_ch[i], skip_ch, decoder_ch[i + 1])
            )

        # 5) final conv
        self.final_conv = nn.Sequential(
            ConvBlock(decoder_ch[-1], decoder_ch[-1]),
            nn.Conv2d(decoder_ch[-1], n_classes, kernel_size=1)
        )

        # init any remaining layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, _, H, W = x.shape
        # 1) encoder, collecting skips
        features = []
        cur = x
        for i, layer in enumerate(self.backbone.features):
            cur = layer(cur)
            if i in self.feature_indices:
                features.append(cur)

        # 2) bottleneck
        x = self.bottleneck(features[-1])

        # 3) decoder w/ skips
        for idx, block in enumerate(self.decoder_blocks):
            skip = features[-2 - idx] if idx < len(features) - 1 else None
            x = block(x, skip)

        # 4) final conv + resize
        x = self.final_conv(x)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
