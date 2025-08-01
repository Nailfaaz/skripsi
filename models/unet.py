# path: models/unet.py

import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch,out_ch))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up   = nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2)-x1.size(2)
        diffX = x2.size(3)-x1.size(3)
        x1 = nn.functional.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1)
    def forward(self,x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        self.inc   = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        factor     = 2 if bilinear else 1
        self.down4 = Down(512,1024//factor)
        self.up1   = Up(1024,512//factor,bilinear)
        self.up2   = Up(512,256//factor,bilinear)
        self.up3   = Up(256,128//factor,bilinear)
        self.up4   = Up(128,64,bilinear)
        self.outc  = OutConv(64,n_classes)

    def forward(self,x):
        x1=self.inc(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        return self.outc(x)
