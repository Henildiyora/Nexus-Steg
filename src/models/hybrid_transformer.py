import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self,channels, reduction=16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2,1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Channel Attention
        ca = self.channel_attn(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attn(sa_input)
        x = x * sa

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.cbam = CBAM(out_channels)

    def forward(self, x):
        return F.relu(self.cbam(self.conv(x)) + self.shortcut(x))


class ViTBottleneck(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class HidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32
        self.enc1 = ResidualBlock(6, 64)    # cover(3) + secret(3) = 6
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.vit = ViTBottleneck(256)

        # Decoder with U-Net skip connections (concat doubles the channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ResidualBlock(256 + 256, 128)  # vit_out + skip3
        self.dec2 = ResidualBlock(128 + 128, 64)   # dec3_out + skip2
        self.dec1 = ResidualBlock(64 + 64, 64)     # dec2_out + skip1
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.4))

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.pool(s3)

        x = self.vit(x)

        x = self.upsample(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.upsample(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.upsample(x)
        x = self.dec1(torch.cat([x, s1], dim=1))

        residual = torch.tanh(self.final(x))
        return cover + self.alpha.clamp(0.1, 0.8) * residual


class RevealNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(3, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.vit = ViTBottleneck(256)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ResidualBlock(256 + 256, 128)
        self.dec2 = ResidualBlock(128 + 128, 64)
        self.dec1 = ResidualBlock(64 + 64, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, stego_img):
        s1 = self.enc1(stego_img)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.pool(s3)

        x = self.vit(x)

        x = self.upsample(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.upsample(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.upsample(x)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return torch.tanh(self.final(x))
