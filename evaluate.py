"""
Nexus-Steg Model Evaluation Test Suite

Loads a trained checkpoint and runs 9 real-world attack scenarios to verify
model robustness. Produces a pass/fail report and saves visual evidence.

Usage:
    python evaluate.py --checkpoint checkpoints/nexus_epoch_99.pth
    python evaluate.py --checkpoint checkpoints/nexus_epoch_99.pth --cover_dir path/to/covers --secret_dir path/to/secrets
"""

import argparse
import io
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.models.discriminator import SteganalysisDiscriminator
from src.engine.trainer import compute_psnr, SSIMCalculator


# ── Attack functions (use real PIL operations, not differentiable approximations) ──

def attack_jpeg(stego_tensor, quality):
    """Save as real JPEG at given quality and reload — simulates actual compression."""
    B, C, H, W = stego_tensor.shape
    result = torch.zeros_like(stego_tensor)
    for i in range(B):
        img = stego_tensor[i] * 0.5 + 0.5  # [-1,1] -> [0,1]
        img = transforms.ToPILImage()(img.cpu().clamp(0, 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img_reloaded = Image.open(buf).convert("RGB")
        t = transforms.ToTensor()(img_reloaded)
        result[i] = t * 2 - 1  # [0,1] -> [-1,1]
    return result.to(stego_tensor.device)


def attack_blur(stego_tensor, sigma=2.0):
    k = 11
    coords = torch.arange(k, dtype=torch.float32, device=stego_tensor.device) - k // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = (g[:, None] * g[None, :])
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(3, 1, k, k)
    pad = k // 2
    return F.conv2d(
        F.pad(stego_tensor, [pad] * 4, mode="reflect"), kernel, groups=3
    )


def attack_noise(stego_tensor, std=0.05):
    return stego_tensor + torch.randn_like(stego_tensor) * std


def attack_resize(stego_tensor, scale=0.5):
    H, W = stego_tensor.shape[2:]
    small = F.interpolate(
        stego_tensor, scale_factor=scale, mode="bilinear", align_corners=False
    )
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)


def attack_social_media(stego_tensor):
    """Resize 75% + JPEG Q=70 — simulates WhatsApp/Discord."""
    resized = attack_resize(stego_tensor, scale=0.75)
    return attack_jpeg(resized, quality=70)


# ── Evaluation engine ──

class Evaluator:
    def __init__(self, checkpoint_path, device_mgr, cover_dir, secret_dir):
        self.device = device_mgr.device
        self.ssim = SSIMCalculator(device=self.device)

        self.hiding_net = HidingNetwork().to(self.device)
        self.reveal_net = RevealNetwork().to(self.device)
        self.discriminator = SteganalysisDiscriminator().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.hiding_net.load_state_dict(ckpt["hiding_net"])
        self.reveal_net.load_state_dict(ckpt["reveal_net"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        print(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt.get('epoch', '?')})")

        alpha = self.hiding_net.alpha.item()
        print(f"Learned alpha (residual scale): {alpha:.4f}")

        self.hiding_net.eval()
        self.reveal_net.eval()
        self.discriminator.eval()

        pipeline = DataPipeline(batch_size=8)
        _, self.val_loader = pipeline.get_train_val_loaders(
            cover_dir, secret_dir, val_split=0.2
        )

        self.out_dir = "results/evaluation"
        os.makedirs(self.out_dir, exist_ok=True)

        self.results = []

    def _verdict(self, value, pass_thresh, warn_thresh=None):
        if value >= pass_thresh:
            return "PASS"
        if warn_thresh is not None and value >= warn_thresh:
            return "WARN"
        return "FAIL"

    def _collect_batches(self, n=4):
        covers, secrets = [], []
        for cover, secret in self.val_loader:
            covers.append(cover.to(self.device))
            secrets.append(secret.to(self.device))
            if len(covers) >= n:
                break
        return torch.cat(covers)[:n * 8], torch.cat(secrets)[:n * 8]

    @torch.no_grad()
    def _run_attack_test(self, name, attack_fn, psnr_pass, psnr_warn=None, ssim_pass=None):
        covers, secrets = self._collect_batches(n=3)
        stegos = self.hiding_net(covers, secrets)
        attacked = attack_fn(stegos)
        revealed = self.reveal_net(attacked.to(self.device))

        psnr_vals, ssim_vals = [], []
        for i in range(covers.shape[0]):
            psnr_vals.append(compute_psnr(revealed[i:i+1], secrets[i:i+1]).item())
            ssim_vals.append(self.ssim(revealed[i:i+1], secrets[i:i+1]).item())

        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)

        v_psnr = self._verdict(avg_psnr, psnr_pass, psnr_warn)
        v_ssim = self._verdict(avg_ssim, ssim_pass, None) if ssim_pass else ""

        passed = v_psnr == "PASS"
        warned = v_psnr == "WARN" or v_ssim == "WARN"

        print(f"\n  TEST: {name}")
        print(f"    Secret PSNR: {avg_psnr:.2f} dB  [{v_psnr} > {psnr_pass}dB]")
        if ssim_pass:
            print(f"    Secret SSIM: {avg_ssim:.4f}    [{v_ssim} > {ssim_pass}]")

        # Save visual evidence: pick first sample
        idx = 0
        strip = torch.cat([
            covers[idx:idx+1],
            secrets[idx:idx+1],
            stegos[idx:idx+1],
            attacked[idx:idx+1].to(self.device),
            revealed[idx:idx+1],
        ], dim=3)
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_image(strip * 0.5 + 0.5, f"{self.out_dir}/{fname}.png")

        status = "PASS" if passed and not warned else ("WARN" if warned else "FAIL")
        self.results.append((name, status, avg_psnr, avg_ssim))
        return avg_psnr, avg_ssim

    @torch.no_grad()
    def test_basic_recovery(self):
        covers, secrets = self._collect_batches(n=3)
        stegos = self.hiding_net(covers, secrets)
        revealed = self.reveal_net(stegos)

        psnr_stego, ssim_stego = [], []
        psnr_secret, ssim_secret = [], []
        for i in range(covers.shape[0]):
            psnr_stego.append(compute_psnr(stegos[i:i+1], covers[i:i+1]).item())
            ssim_stego.append(self.ssim(stegos[i:i+1], covers[i:i+1]).item())
            psnr_secret.append(compute_psnr(revealed[i:i+1], secrets[i:i+1]).item())
            ssim_secret.append(self.ssim(revealed[i:i+1], secrets[i:i+1]).item())

        avg = lambda lst: sum(lst) / len(lst)
        ps, ss = avg(psnr_stego), avg(ssim_stego)
        pr, sr = avg(psnr_secret), avg(ssim_secret)

        v_ps = self._verdict(ps, 30)
        v_ss = self._verdict(ss, 0.90)
        v_pr = self._verdict(pr, 28, 24)
        v_sr = self._verdict(sr, 0.80, 0.60)

        print(f"\n  TEST: Basic Recovery (no attack)")
        print(f"    PSNR(stego vs cover):  {ps:.2f} dB  [{v_ps} > 30dB]")
        print(f"    SSIM(stego vs cover):  {ss:.4f}     [{v_ss} > 0.90]")
        print(f"    PSNR(secret recover):  {pr:.2f} dB  [{v_pr} > 28dB]")
        print(f"    SSIM(secret recover):  {sr:.4f}     [{v_sr} > 0.80]")

        strip = torch.cat([
            covers[0:1], secrets[0:1], stegos[0:1], revealed[0:1]
        ], dim=3)
        save_image(strip * 0.5 + 0.5, f"{self.out_dir}/basic_recovery.png")

        worst = min([v_ps, v_ss, v_pr, v_sr], key=lambda v: {"PASS": 2, "WARN": 1, "FAIL": 0}[v])
        self.results.append(("Basic Recovery", worst, pr, sr))

    @torch.no_grad()
    def test_steganalysis_detection(self):
        covers, secrets = self._collect_batches(n=3)
        stegos = self.hiding_net(covers, secrets)

        d_cover = torch.sigmoid(self.discriminator(covers))
        d_stego = torch.sigmoid(self.discriminator(stegos))

        # Discriminator outputs: >0.5 = "real cover", <0.5 = "stego detected"
        cover_acc = (d_cover > 0.5).float().mean().item() * 100
        stego_detect = (d_stego < 0.5).float().mean().item() * 100

        # For steganography, we WANT the discriminator to fail (near 50%)
        avg_acc = (cover_acc + stego_detect) / 2
        passed = avg_acc < 60

        verdict = "PASS" if passed else "FAIL"
        print(f"\n  TEST: Steganalysis Detection")
        print(f"    Discriminator correct on covers: {cover_acc:.1f}%")
        print(f"    Discriminator detects stegos:    {stego_detect:.1f}%")
        print(f"    Average accuracy:                {avg_acc:.1f}%  [{verdict} < 60%]")

        self.results.append(("Steganalysis Detection", verdict, avg_acc, 0))

    def run_all(self):
        print("=" * 56)
        print("       NEXUS-STEG EVALUATION REPORT")
        print("=" * 56)

        self.test_basic_recovery()

        self._run_attack_test(
            "JPEG-90 Robustness",
            lambda x: attack_jpeg(x, 90),
            psnr_pass=22, psnr_warn=18, ssim_pass=0.50,
        )
        self._run_attack_test(
            "JPEG-50 Robustness (heavy)",
            lambda x: attack_jpeg(x, 50),
            psnr_pass=18, psnr_warn=15, ssim_pass=0.35,
        )
        self._run_attack_test(
            "Gaussian Blur (sigma=2.0)",
            lambda x: attack_blur(x, sigma=2.0),
            psnr_pass=18, psnr_warn=15, ssim_pass=0.40,
        )
        self._run_attack_test(
            "Gaussian Noise (std=0.05)",
            lambda x: attack_noise(x, std=0.05),
            psnr_pass=20, psnr_warn=16, ssim_pass=0.45,
        )
        self._run_attack_test(
            "Screenshot Sim (resize 50%)",
            lambda x: attack_resize(x, scale=0.5),
            psnr_pass=18, psnr_warn=15, ssim_pass=0.35,
        )
        self._run_attack_test(
            "Social Media (WhatsApp-like)",
            attack_social_media,
            psnr_pass=16, psnr_warn=13, ssim_pass=0.30,
        )

        self.test_steganalysis_detection()

        # ── Summary ──
        passes = sum(1 for _, s, _, _ in self.results if s == "PASS")
        warns = sum(1 for _, s, _, _ in self.results if s == "WARN")
        fails = sum(1 for _, s, _, _ in self.results if s == "FAIL")
        total = len(self.results)

        print("\n" + "=" * 56)
        print(f"  SUMMARY: {passes}/{total} PASS | {warns}/{total} WARN | {fails}/{total} FAIL")
        print(f"  Visual evidence saved to: {self.out_dir}/")
        print("=" * 56)

        # Save text report
        with open(f"{self.out_dir}/report.txt", "w") as f:
            f.write("NEXUS-STEG EVALUATION REPORT\n\n")
            for name, status, psnr, ssim in self.results:
                f.write(f"{status:5s} | {name:35s} | PSNR={psnr:.2f}dB  SSIM={ssim:.4f}\n")
            f.write(f"\nSUMMARY: {passes}/{total} PASS | {warns}/{total} WARN | {fails}/{total} FAIL\n")
        print(f"  Report saved to: {self.out_dir}/report.txt")


def main():
    parser = argparse.ArgumentParser(description="Nexus-Steg Evaluation Suite")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--cover_dir", type=str, default="datasets/cover", help="Cover images directory"
    )
    parser.add_argument(
        "--secret_dir", type=str, default="datasets/secret/test",
        help="Secret images directory",
    )
    args = parser.parse_args()

    dm = DeviceManager()
    evaluator = Evaluator(args.checkpoint, dm, args.cover_dir, args.secret_dir)
    evaluator.run_all()


if __name__ == "__main__":
    main()
