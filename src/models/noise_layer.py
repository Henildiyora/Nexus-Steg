import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DiffJPEG(nn.Module):
    """
    Differentiable JPEG approximation via block-wise quantization
    using a straight-through estimator.
    """

    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, Hp, Wp = x.shape
        x = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, C, H/8, W/8, 8, 8)
        shape = x.shape
        x_flat = x.contiguous().view(-1, 8, 8)

        # Map quality to quantization levels in [-1, 1] range
        levels = max(8, self.quality * 2)
        step = 2.0 / levels

        # Straight-through estimator: gradient flows through as if identity
        x_q = x_flat + (torch.round(x_flat / step) * step - x_flat).detach()

        x_q = x_q.view(*shape)
        x_q = x_q.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
        return x_q[:, :, :H, :W]


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma_range=(0.5, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def forward(self, x):
        sigma = random.uniform(*self.sigma_range)
        k = self.kernel_size
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        kernel = (g[:, None] * g[None, :])
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(x.shape[1], 1, k, k)
        pad = k // 2
        return F.conv2d(F.pad(x, [pad]*4, mode="reflect"), kernel, groups=x.shape[1])


class GaussianNoise(nn.Module):
    def __init__(self, std_range=(0.01, 0.05)):
        super().__init__()
        self.std_range = std_range

    def forward(self, x):
        std = random.uniform(*self.std_range)
        return x + torch.randn_like(x) * std


class PixelDropout(nn.Module):
    def __init__(self, drop_range=(0.05, 0.15)):
        super().__init__()
        self.drop_range = drop_range

    def forward(self, x):
        p = random.uniform(*self.drop_range)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > p).float()
        return x * mask

class RandomResizing(nn.Module):
    """
    Simulates the downsampling and upsampling artifacts of social media.
    This is critical for surviving WhatsApp/Telegram compression.
    """
    def __init__(self, scale_range=(0.5, 0.9)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x):
        orig_size = x.shape[2:]
        # Randomly choose a scale factor (e.g., 256px -> 128px)
        scale = random.uniform(*self.scale_range)
        new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        
        # Downsample then Upsample back to original size
        down = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
        return F.interpolate(down, size=orig_size, mode='bilinear', align_corners=False)

class DifferentiableNoiseLayer(nn.Module):
    """
    Randomly applies distortions to force the decoder to learn robust extraction.
    Updated for Phase 4 robustness including resizing artifacts.
    """
    def __init__(self):
        super().__init__()
        self.jpeg_50 = DiffJPEG(quality=50)
        self.jpeg_90 = DiffJPEG(quality=90)
        self.blur = GaussianBlur()
        self.noise = GaussianNoise()
        self.dropout = PixelDropout()
        self.resizing = RandomResizing() # New Phase 4 Component

    def forward(self, x):
        if not self.training:
            return x

        # Added 'resizing' and 'extreme_combo' for higher impact
        distortion = random.choice([
            "identity", "jpeg_50", "jpeg_90", "blur", "noise", 
            "dropout", "resizing", "combined", "extreme_combo"
        ])

        if distortion == "identity":
            return x
        elif distortion == "jpeg_50":
            return self.jpeg_50(x)
        elif distortion == "jpeg_90":
            return self.jpeg_90(x)
        elif distortion == "blur":
            return self.blur(x)
        elif distortion == "noise":
            return self.noise(x)
        elif distortion == "dropout":
            return self.dropout(x)
        elif distortion == "resizing":
            return self.resizing(x)
        elif distortion == "combined": # JPEG + noise
            return self.noise(self.jpeg_90(x))
        else: # extreme_combo: Resizing + JPEG + Noise
            return self.noise(self.jpeg_50(self.resizing(x)))