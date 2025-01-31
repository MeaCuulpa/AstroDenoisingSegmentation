import torch
import torch.nn as nn


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = self.conv_block(32, 16)

        self.final = nn.Conv2d(16, 2, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 128 -> 128
        e2 = self.pool(e1)  # 128 -> 64
        e2 = self.enc2(e2)  # 64 -> 64

        # Bottleneck
        bn = self.pool(e2)  # 64 -> 32
        bn = self.bottleneck(bn)

        # Decoder
        d1 = self.up1(bn)  # 32 -> 64
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)  # 64 -> 128
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return torch.sigmoid(self.final(d2))
