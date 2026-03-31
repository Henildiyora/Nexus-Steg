import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F

from src.models.noise_layer import DifferentiableNoiseLayer
from src.models.discriminator import SteganalysisDiscriminator


class FFTLoss(nn.Module):
    """Penalizes spectral discrepancies to survive JPEG/WhatsApp."""
    def forward(self, x, y):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            fx = torch.fft.rfft2(x.float())
            fy = torch.fft.rfft2(y.float())
            return F.l1_loss(torch.abs(fx), torch.abs(fy))

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss that properly normalizes inputs from the training range
    [-1, 1] to ImageNet range before feeding into VGG.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)

        self.register_buffer(
            "mean", torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        )
        self.mse = nn.MSELoss()

    def _normalize(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std

    def forward(self, x, y):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = self._normalize(x.float())
            y32 = self._normalize(y.float())
            return self.mse(self.vgg(x32), self.vgg(y32))


def compute_psnr(img1, img2):
    """PSNR between two tensors in [-1, 1] range."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(4.0 / mse)


class SSIMCalculator:
    """SSIM with a pre-computed Gaussian window to avoid re-allocation every call."""

    def __init__(self, window_size=11, sigma=1.5, channels=3, device="cpu"):
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        self.pad = window_size // 2
        self.channels = channels

    def __call__(self, img1, img2):
        C = self.channels
        pad = self.pad
        w = self.window

        mu1 = F.conv2d(img1, w, padding=pad, groups=C)
        mu2 = F.conv2d(img2, w, padding=pad, groups=C)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=C) - mu12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


class NexusTrainer:
    def __init__(self, hiding_net, reveal_net, device_manager, total_epochs=100):
        self.device = device_manager.device
        self.hiding_net = hiding_net.to(self.device)
        self.reveal_net = reveal_net.to(self.device)

        self.noise_layer = DifferentiableNoiseLayer().to(self.device)

        self.discriminator = SteganalysisDiscriminator().to(self.device)

        self.optimizer_g = optim.Adam(
            list(self.hiding_net.parameters()) + list(self.reveal_net.parameters()),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=5e-5,
            betas=(0.5, 0.999),
        )

        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=total_epochs, eta_min=1e-5)
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=total_epochs, eta_min=5e-6)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.perceptual_loss = VGGPerceptualLoss(self.device)

        self.fft_loss = FFTLoss()
        self.ssim_calc = SSIMCalculator(device=self.device)
        self.recovery_weight = 5.0
        self.adv_weight = 0.0
        self._global_step = 0
        self.d_train_every = 2

    def step_schedulers(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler")
            self.scheduler_g.step()
            self.scheduler_d.step()

    def _get_all_generator_params(self):
        return list(self.hiding_net.parameters()) + list(self.reveal_net.parameters())

    def train_step(self, cover, secret, phase=1, scaler=None, recovery_aux=True):
        self._global_step += 1

        # Discriminator Step (every d_train_every steps)
        train_d = (self._global_step % self.d_train_every == 0)
        if train_d:
            self.optimizer_d.zero_grad()

            with torch.no_grad():
                stego_detach = self.hiding_net(cover, secret).detach()

            d_real = self.discriminator(cover)
            d_fake = self.discriminator(stego_detach)

            real_label = torch.ones_like(d_real) * 0.9
            fake_label = torch.zeros_like(d_fake) + 0.1

            loss_d = 0.5 * (
                self.bce_loss(d_real, real_label) + self.bce_loss(d_fake, fake_label)
            )

            if scaler is not None:
                scaler.scale(loss_d).backward()
                scaler.unscale_(self.optimizer_d)
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                scaler.step(self.optimizer_d)
                scaler.update()
            else:
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_d.step()
        else:
            with torch.no_grad():
                stego_detach = self.hiding_net(cover, secret).detach()
                d_real = self.discriminator(cover)
                d_fake = self.discriminator(stego_detach)
                loss_d = 0.5 * (
                    self.bce_loss(d_real, torch.ones_like(d_real) * 0.9)
                    + self.bce_loss(d_fake, torch.zeros_like(d_fake) + 0.1)
                )

        # ---- Generator Step ----
        self.optimizer_g.zero_grad(set_to_none=True)

        stego = self.hiding_net(cover, secret)
        stego_noised = self.noise_layer(stego) if phase > 1 else stego
        revealed = self.reveal_net(stego_noised)

        # Invisibility: lighter perceptual/FFT in phase 1 so embedding can grow.
        if phase == 1:
            l_inv = self.mse_loss(stego, cover) + \
                    0.05 * self.perceptual_loss(stego, cover) + \
                    0.05 * self.fft_loss(stego, cover)
        else:
            l_inv = self.mse_loss(stego, cover) + \
                    0.1 * self.perceptual_loss(stego, cover) + \
                    0.1 * self.fft_loss(stego, cover)

        # Recovery: phase 1 = MSE only. Phase 2 can run noise+adv with MSE-only first
        # (recovery_aux=False) to avoid a huge SSIM spike when secret SSIM is still ~0.2.
        l_rec = F.mse_loss(revealed, secret)
        if phase == 2 and recovery_aux:
            ssim_rec = self.ssim_calc(revealed, secret)
            l_rec = l_rec + 0.25 * (1.0 - ssim_rec) \
                          + 0.05 * self.perceptual_loss(revealed, secret)
        elif phase >= 3:
            ssim_rec = self.ssim_calc(revealed, secret)
            l_rec = l_rec + 0.5 * (1.0 - ssim_rec) \
                          + 0.1 * self.perceptual_loss(revealed, secret)

        # Adversarial loss
        l_adv = 0
        if phase >= 2:
            d_fake_for_g = self.discriminator(stego)
            l_adv = self.bce_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))

        total_loss = l_inv + self.recovery_weight * l_rec + self.adv_weight * l_adv

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer_g)
            nn.utils.clip_grad_norm_(self._get_all_generator_params(), max_norm=1.0)
            scaler.step(self.optimizer_g)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._get_all_generator_params(), max_norm=1.0)
            self.optimizer_g.step()

        return total_loss.item(), l_inv.item(), l_rec.item(), loss_d.item()

    @torch.no_grad()
    def validate(self, val_loader):
        self.hiding_net.eval()
        self.reveal_net.eval()
        self.noise_layer.eval()

        psnr_stego_sum, ssim_stego_sum = 0.0, 0.0
        psnr_secret_sum, ssim_secret_sum = 0.0, 0.0
        count = 0
        sample = None

        for cover, secret in val_loader:
            cover, secret = cover.to(self.device), secret.to(self.device)
            stego = self.hiding_net(cover, secret)
            revealed = self.reveal_net(stego)

            psnr_stego_sum += compute_psnr(stego, cover).item()
            ssim_stego_sum += self.ssim_calc(stego, cover).item()
            psnr_secret_sum += compute_psnr(revealed, secret).item()
            ssim_secret_sum += self.ssim_calc(revealed, secret).item()
            count += 1

            if sample is None:
                sample = (cover[0:1], secret[0:1], stego[0:1], revealed[0:1])

        self.hiding_net.train()
        self.reveal_net.train()
        self.noise_layer.train()

        n = max(count, 1)
        return {
            "psnr_stego": psnr_stego_sum / n,
            "ssim_stego": ssim_stego_sum / n,
            "psnr_secret": psnr_secret_sum / n,
            "ssim_secret": ssim_secret_sum / n,
            "sample": sample,
        }
