import torch
import torch.nn as nn


class SpectralConv(nn.Module):
    """Conv2d with spectral normalization for stable GAN training."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SRNetBlock(nn.Module):
    """Residual block inspired by SRNet steganalysis architecture."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
        )
        self.shortcut = (
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class SteganalysisDiscriminator(nn.Module):
    """
    SRNet-inspired discriminator (the "Sentry") for adversarial training.
    Uses high-pass preprocessing to detect steganographic artifacts in the
    residual domain, similar to how real steganalysis tools operate.
    """

    def __init__(self):
        super().__init__()

        # Learnable high-pass preprocessing (SRNet-style)
        self.prep = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self._init_highpass()

        self.features = nn.Sequential(
            SRNetBlock(16, 32),
            nn.AvgPool2d(2),    # 128
            SRNetBlock(32, 64),
            nn.AvgPool2d(2),    # 64
            SRNetBlock(64, 128),
            nn.AvgPool2d(2),    # 32
            SRNetBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def _init_highpass(self):
        """Initialize first conv layer with SRM-like high-pass filters."""
        with torch.no_grad():
            w = self.prep[0].weight
            nn.init.xavier_normal_(w)
            # Embed a 3x3 high-pass kernel in the center of the first few filters
            hp = torch.tensor(
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=w.dtype
            ) / 4.0
            for i in range(min(3, w.shape[0])):
                for c in range(w.shape[1]):
                    w[i, c, 1:4, 1:4] = hp

    def forward(self, x):
        x = self.prep(x)
        x = self.features(x)
        return self.classifier(x)
