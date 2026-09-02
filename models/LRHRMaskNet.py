
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1), nn.SiLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.se(x)
        return x * w


class LRBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Conv2d(64, 256, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
        )

        self.res = nn.Sequential(ResidualBlock(128), SEBlock(128))


    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return x


class HRBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Conv2d(64, 128, 5, stride=1, padding=2), nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.PixelUnshuffle(downscale_factor=2),

        )
        self.res = nn.Sequential(ResidualBlock(512), SEBlock(512))
    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return x


class FusionDecoder(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.proj_lr = nn.Conv2d(128, 256, 1)
        self.proj_hr = nn.Conv2d(512, 256, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(256+256, 256, 1), nn.SiLU(inplace=True),
            nn.Conv2d(256, 1, 1), nn.Sigmoid()
        )
        nn.init.zeros_(self.gate[-2].bias)

        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv2d(256, 256, kernel_size=1),  # Pointwise
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # Depthwise
            nn.Conv2d(512, 512, kernel_size=1),  # Pointwise
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(512)

        # Upsampling path with skip connections
        self.up1 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.SiLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(inplace=True),
        )

        # Final conv layers
        self.final = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, lr_feat, hr_feat):


        l = self.proj_lr(lr_feat)  # (N,256,H,W)
        h = self.proj_hr(hr_feat)  # (N,256,H,W)
        g = self.gate(torch.cat([l, h], dim=1))  # (N,1,H,W), 初值≈0.5
        x = g * h + (1 - g) * l

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u1 = F.interpolate(b, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1)

        u2 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2)

        out = self.final(u2)
        return out



class LRHRMaskNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.lr_branch = LRBranch(in_channels=in_channels)
        self.hr_branch = HRBranch(in_channels=in_channels)

        self.decoder = FusionDecoder()

    def forward(self, lr, hr):
        Hl, Wl = lr.shape[-2:];
        Hh, Wh = hr.shape[-2:]
        assert Hh == 4 * Hl and Wh == 4 * Wl, "Expect HR to be 4× LR."
        assert Hh % 2 == 0 and Wh % 2 == 0, "HR H,W must be even for PixelUnshuffle(2)."
        hr_feat = self.hr_branch(hr)
        lr_feat = self.lr_branch(lr)

        out = self.decoder(lr_feat, hr_feat)
        return out

