import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, n_colors=10):
        super().__init__()
        self.color_fc = nn.Linear(n_colors, 64)  # Color embedding

        # First encoder now takes polygon channels + 64 color channels
        self.enc1 = ConvBlock(in_channels + 64, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, color_onehot):
        # Expand color conditioning
        color_embed = self.color_fc(color_onehot)  # (B, 64)
        B, _, H, W = x.shape
        color_map = color_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, 64, H, W)

        # Encoder
        x1 = self.enc1(torch.cat([x, color_map], dim=1))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Bottleneck
        x4 = self.bottleneck(self.pool(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return torch.sigmoid(self.final(x))
