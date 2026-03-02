import argparse
import csv
import os
import random

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.engine.trainer import NexusTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class NexusApp:
    def __init__(self, epochs=100, batch_size=None, checkpoint_every=10, patience=15,
                 num_workers=None):
        self.device_mgr = DeviceManager()
        self.device = self.device_mgr.device
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.patience = patience

        if batch_size is None:
            batch_size = 64 if self.device_mgr.is_cuda else 4

        self.pipeline = DataPipeline(batch_size=batch_size, num_workers=num_workers)
        self.train_loader, self.val_loader = self.pipeline.get_train_val_loaders(
            cover_dir="datasets/cover",
            secret_dir="datasets/secret/MUL-PanSharpen",
            val_split=0.2,
        )

        self.hiding_net = HidingNetwork().to(self.device)
        self.reveal_net = RevealNetwork().to(self.device)

        self.trainer = NexusTrainer(
            self.hiding_net, self.reveal_net, self.device_mgr, total_epochs=epochs
        )

        self.use_amp = self.device_mgr.is_cuda
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.use_amp
        )

        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def save_visual_results(self, epoch, cover, secret, stego, revealed):
        comparison = torch.cat([cover, secret, stego, revealed], dim=3)
        save_image(comparison * 0.5 + 0.5, f"results/epoch_{epoch}.png")

    def run_sanity(self):
        """Verify initial losses match expected values and visualize data pipeline."""
        print("=" * 56)
        print("  SANITY CHECK")
        print("=" * 56)

        cover, secret = next(iter(self.train_loader))
        cover = cover.to(self.device)
        secret = secret.to(self.device)

        #  Visualize what the network actually receives
        n = min(8, cover.shape[0])
        grid = torch.cat([cover[:n], secret[:n]], dim=0)
        save_image(grid * 0.5 + 0.5, "results/sanity_inputs.png", nrow=n)
        print(f"  Saved {n} cover + {n} secret images to results/sanity_inputs.png")
        print(f"  Cover range: [{cover.min():.3f}, {cover.max():.3f}]")
        print(f"  Secret range: [{secret.min():.3f}, {secret.max():.3f}]")

        # Verify loss @ init
        self.hiding_net.eval()
        self.reveal_net.eval()
        with torch.no_grad():
            stego = self.hiding_net(cover, secret)
            revealed = self.reveal_net(stego)

            l_inv = torch.nn.functional.mse_loss(stego, cover).item()
            l_rec = torch.nn.functional.mse_loss(revealed, secret).item()

            d_out = self.trainer.discriminator(cover)
            l_disc = torch.nn.functional.binary_cross_entropy_with_logits(
                d_out, torch.ones_like(d_out)
            ).item()

        print(f"\n  Initial losses (before any training):")
        print(f"    l_inv  = {l_inv:.4f}  (expected: small, ~0.01-0.10)")
        print(f"    l_rec  = {l_rec:.4f}  (expected: ~0.30-0.70)")
        print(f"    l_disc = {l_disc:.4f}  (expected: ~0.693 = log(2))")

        ok = True
        if l_disc < 0.3 or l_disc > 1.5:
            print("  WARNING: l_disc is far from 0.693 -- discriminator init may be off")
            ok = False
        if l_inv > 0.5:
            print("  WARNING: l_inv is unexpectedly high -- check alpha / encoder init")
            ok = False

        if ok:
            print("  All initial losses look reasonable.")
        print("=" * 56)
        self.hiding_net.train()
        self.reveal_net.train()

    # Overfit one batch
    def run_overfit_one_batch(self, steps=200):
        """Train on a single batch to verify model capacity."""
        print("=" * 56)
        print("  OVERFIT ONE BATCH (Karpathy Recipe 2.8)")
        print(f"  Training on 1 batch for {steps} steps...")
        print("=" * 56)

        cover, secret = next(iter(self.train_loader))
        cover = cover.to(self.device)
        secret = secret.to(self.device)

        self.hiding_net.train()
        self.reveal_net.train()

        for step in range(steps):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, l_inv, l_rec, l_disc = self.trainer.train_step(
                    cover, secret, phase=1,
                    scaler=self.scaler if self.use_amp else None,
                )

            if step % 10 == 0 or step == steps - 1:
                print(
                    f"  Step {step:3d}/{steps} | "
                    f"loss={loss:.6f}  inv={l_inv:.6f}  rec={l_rec:.6f}  disc={l_disc:.4f}"
                )

        print()
        if loss < 0.01:
            print("  PASS: Loss reached near-zero. Model has sufficient capacity.")
        elif loss < 0.1:
            print("  WARN: Loss is low but not near-zero. Model likely OK but check recovery.")
        else:
            print("  FAIL: Loss did not converge. Possible capacity or learning rate problem.")
        print("=" * 56)

    # Main training loop with early stopping

    def run(self):
        print(f"Starting Nexus-Steg Training on {self.device}")

        csv_path = "results/training_log.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "phase", "train_loss", "l_inv", "l_rec", "l_disc",
            "val_psnr_stego", "val_ssim_stego", "val_psnr_secret", "val_ssim_secret",
            "lr",
        ])

        best_psnr_secret = -float("inf")
        epochs_without_improvement = 0

        try:
            for epoch in range(self.epochs):
                if epoch < 30:
                    phase = 1
                    self.trainer.recovery_weight = 10.0
                    self.trainer.adv_weight = 0.0
                elif epoch < 60:
                    phase = 2
                    self.trainer.recovery_weight = 20.0
                    self.trainer.adv_weight = 0.01
                else:
                    phase = 3
                    self.trainer.recovery_weight = 30.0
                    self.trainer.adv_weight = 0.05

                self.hiding_net.train()
                self.reveal_net.train()
                self.trainer.noise_layer.train()

                total_loss, total_inv, total_rec, total_disc = 0.0, 0.0, 0.0, 0.0
                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch}/{self.epochs} (Phase {phase})",
                )

                for i, (cover, secret) in pbar:
                    cover = cover.to(self.device, non_blocking=True)
                    secret = secret.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        loss, l_inv, l_rec, l_disc = self.trainer.train_step(
                            cover,
                            secret,
                            phase=phase,
                            scaler=self.scaler if self.use_amp else None,
                        )

                    total_loss += loss
                    total_inv += l_inv
                    total_rec += l_rec
                    total_disc += l_disc
                    pbar.set_postfix(
                        loss=f"{loss:.4f}",
                        inv=f"{l_inv:.4f}",
                        rec=f"{l_rec:.4f}",
                        disc=f"{l_disc:.4f}",
                    )

                n_batches = len(self.train_loader)
                self.trainer.step_schedulers()

                metrics = self.trainer.validate(self.val_loader)
                print(
                    f"  Val | PSNR(stego): {metrics['psnr_stego']:.2f}dB  "
                    f"SSIM(stego): {metrics['ssim_stego']:.4f}  "
                    f"PSNR(secret): {metrics['psnr_secret']:.2f}dB  "
                    f"SSIM(secret): {metrics['ssim_secret']:.4f}"
                )

                current_lr = self.trainer.optimizer_g.param_groups[0]["lr"]
                csv_writer.writerow([
                    epoch, phase,
                    total_loss / n_batches,
                    total_inv / n_batches,
                    total_rec / n_batches,
                    total_disc / n_batches,
                    metrics["psnr_stego"],
                    metrics["ssim_stego"],
                    metrics["psnr_secret"],
                    metrics["ssim_secret"],
                    current_lr,
                ])
                csv_file.flush()

                sample = metrics.get("sample")
                if sample is not None:
                    self.save_visual_results(epoch, *sample)

                # Early stopping
                if metrics["psnr_secret"] > best_psnr_secret:
                    best_psnr_secret = metrics["psnr_secret"]
                    epochs_without_improvement = 0
                    best_ckpt = {
                        "epoch": epoch,
                        "hiding_net": self.hiding_net.state_dict(),
                        "reveal_net": self.reveal_net.state_dict(),
                        "discriminator": self.trainer.discriminator.state_dict(),
                        "optimizer_g": self.trainer.optimizer_g.state_dict(),
                        "optimizer_d": self.trainer.optimizer_d.state_dict(),
                        "scheduler_g": self.trainer.scheduler_g.state_dict(),
                        "scheduler_d": self.trainer.scheduler_d.state_dict(),
                        "scaler": self.scaler.state_dict(),
                    }
                    torch.save(best_ckpt, "checkpoints/nexus_best.pth")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        print(
                            f"\n  Early stopping: no improvement in PSNR(secret) for "
                            f"{self.patience} epochs. Best: {best_psnr_secret:.2f}dB"
                        )
                        break

                is_last = epoch == self.epochs - 1
                if is_last or (epoch + 1) % self.checkpoint_every == 0:
                    checkpoint = {
                        "epoch": epoch,
                        "hiding_net": self.hiding_net.state_dict(),
                        "reveal_net": self.reveal_net.state_dict(),
                        "discriminator": self.trainer.discriminator.state_dict(),
                        "optimizer_g": self.trainer.optimizer_g.state_dict(),
                        "optimizer_d": self.trainer.optimizer_d.state_dict(),
                        "scheduler_g": self.trainer.scheduler_g.state_dict(),
                        "scheduler_d": self.trainer.scheduler_d.state_dict(),
                        "scaler": self.scaler.state_dict(),
                    }
                    torch.save(checkpoint, f"checkpoints/nexus_epoch_{epoch}.pth")
        finally:
            csv_file.close()
            print(f"  Training log saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Nexus-Steg Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity checks only (verify losses + visualize inputs)")
    parser.add_argument("--overfit_one_batch", action="store_true",
                        help="Train on a single batch for 200 steps to verify model capacity")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (default: auto, use 2 for Colab)")
    args = parser.parse_args()

    set_seed(args.seed)

    app = NexusApp(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    if args.sanity:
        app.run_sanity()
    elif args.overfit_one_batch:
        app.run_overfit_one_batch(steps=200)
    else:
        app.run()


if __name__ == "__main__":
    main()
