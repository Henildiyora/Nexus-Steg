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

        fx = torch.fft.rfft2(x)
        fy = torch.fft.rfft2(y)
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
        return self.mse(self.vgg(self._normalize(x)), self.vgg(self._normalize(y)))


def compute_psnr(img1, img2):
    """PSNR between two tensors in [-1, 1] range."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(4.0 / mse)  # max range is 2.0 for [-1,1], so peak^2 = 4


def _gaussian_kernel_1d(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """Structural similarity between two tensors in [-1, 1] range."""
    device = img1.device
    C = img1.shape[1]

    kernel_1d = _gaussian_kernel_1d(window_size, sigma, device)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    window = kernel_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = nn.functional.conv2d(img1, window, padding=pad, groups=C)
    mu2 = nn.functional.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=pad, groups=C) - mu12

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
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
        )

        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=total_epochs, eta_min=1e-6)
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=total_epochs, eta_min=1e-6)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.perceptual_loss = VGGPerceptualLoss(self.device)

        self.fft_loss = FFTLoss()
        self.recovery_weight = 5.0
        self.adv_weight = 0.0

    def step_schedulers(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def _get_all_generator_params(self):
        return list(self.hiding_net.parameters()) + list(self.reveal_net.parameters())

    def train_step(self, cover, secret, phase=1, scaler=None):
        # Discriminator Step
        self.optimizer_d.zero_grad()

        with torch.amp.autocast(device_type=ac_dtype, enabled=use_amp):
            with torch.no_grad():
                stego_detach = self.hiding_net(cover, secret).detach()

            d_real = self.discriminator(cover)
            d_fake = self.discriminator(stego_detach)

            real_label = torch.ones_like(d_real)
            fake_label = torch.zeros_like(d_fake)

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

        # ---- Generator Step ----
        self.optimizer_g.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=ac_dtype, enabled=use_amp):
            stego = self.hiding_net(cover, secret)

        # Apply noise distortions between encoder and decoder
        stego = self.hiding_net(cover, secret)
        stego_noised = self.noise_layer(stego) if phase > 1 else stego
        revealed = self.reveal_net(stego_noised)

        # Multi-Objective Loss
        l_inv = self.mse_loss(stego, cover) + \
                0.1 * self.perceptual_loss(stego, cover) + \
                0.1 * self.fft_loss(stego, cover)
        l_rec = F.mse_loss(revealed, secret)
        
        # Adversarial Loss 
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

        for cover, secret in val_loader:
            cover, secret = cover.to(self.device), secret.to(self.device)
            stego = self.hiding_net(cover, secret)
            revealed = self.reveal_net(stego)

            psnr_stego_sum += compute_psnr(stego, cover).item()
            ssim_stego_sum += compute_ssim(stego, cover).item()
            psnr_secret_sum += compute_psnr(revealed, secret).item()
            ssim_secret_sum += compute_ssim(revealed, secret).item()
            count += 1

        self.hiding_net.train()
        self.reveal_net.train()
        self.noise_layer.train()

        n = max(count, 1)
        return {
            "psnr_stego": psnr_stego_sum / n,
            "ssim_stego": ssim_stego_sum / n,
            "psnr_secret": psnr_secret_sum / n,
            "ssim_secret": ssim_secret_sum / n,
        }
