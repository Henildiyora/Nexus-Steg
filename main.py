import torch
from torchvision.utils import save_image
from tqdm import tqdm
import os

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.engine.trainer import NexusTrainer


class NexusApp:
    def __init__(self, epochs=5):
        self.device_mgr = DeviceManager()
        self.device = self.device_mgr.device
        self.epochs = epochs

        self.pipeline = DataPipeline(batch_size=2)
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

    def run(self):
        print(f"Starting Nexus-Steg Training on {self.device}")

        for epoch in range(self.epochs):
        
            # Define phases based on your progress through 100 epochs
            if epoch < 30:
                phase = 1
                self.trainer.recovery_weight = 10.0  # Focus on pure hiding/recovery
                self.trainer.adv_weight = 0.0       # No adversarial pressure yet
            elif epoch < 60:
                phase = 2
                self.trainer.recovery_weight = 20.0  # Increase penalty for recovery errors
                self.trainer.adv_weight = 0.01      # Start mild adversarial pressure
            else:
                phase = 3
                self.trainer.recovery_weight = 30.0  # Max punishment for blurry secrets
                self.trainer.adv_weight = 0.05      # Max pressure to pass AI sentry

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
                cover = cover.to(self.device)
                secret = secret.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
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

            # Validation metrics
            metrics = self.trainer.validate(self.val_loader)
            print(
                f"  Val | PSNR(stego): {metrics['psnr_stego']:.2f}dB  "
                f"SSIM(stego): {metrics['ssim_stego']:.4f}  "
                f"PSNR(secret): {metrics['psnr_secret']:.2f}dB  "
                f"SSIM(secret): {metrics['ssim_secret']:.4f}"
            )

            # Visual results from last batch
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    stego = self.hiding_net(cover, secret)
                    revealed = self.reveal_net(stego)
                self.save_visual_results(
                    epoch, cover[0:1], secret[0:1], stego[0:1], revealed[0:1]
                )

            # Checkpoint with full state for resumption
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
    app = NexusApp(epochs=100)
    app.run()
