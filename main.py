import torch
from torchvision.utils import save_image
from tqdm import tqdm
import os

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.engine.trainer import NexusTrainer


class NexusApp:
    def __init__(self, epochs=100, batch_size=None, checkpoint_every=10):
        self.device_mgr = DeviceManager()
        self.device = self.device_mgr.device
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every

        if batch_size is None:
            batch_size = 64 if self.device_mgr.is_cuda else 4

        self.pipeline = DataPipeline(batch_size=batch_size)
        self.train_loader, self.val_loader = self.pipeline.get_train_val_loaders(
            cover_dir="datasets/cover",
            secret_dir="datasets/secret/train",
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

    def run(self):
        print(f"Starting Nexus-Steg Training on {self.device}")

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

            total_loss = 0.0
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
                        scaler=self.scaler if self.use_amp else None
                    )

                total_loss += loss
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    inv=f"{l_inv:.4f}",
                    rec=f"{l_rec:.4f}",
                    disc=f"{l_disc:.4f}",
                )

            self.trainer.step_schedulers()

            metrics = self.trainer.validate(self.val_loader)
            print(
                f"  Val | PSNR(stego): {metrics['psnr_stego']:.2f}dB  "
                f"SSIM(stego): {metrics['ssim_stego']:.4f}  "
                f"PSNR(secret): {metrics['psnr_secret']:.2f}dB  "
                f"SSIM(secret): {metrics['ssim_secret']:.4f}"
            )

            sample = metrics.get("sample")
            if sample is not None:
                self.save_visual_results(epoch, *sample)

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


if __name__ == "__main__":
    app = NexusApp(epochs=100, batch_size=None, checkpoint_every=10)
    app.run()
