import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from src.core.device import DeviceManager
import tifffile as tiff
import numpy as np


class StegoDataset(Dataset):
    def __init__(self, cover_dir, secret_dir, transform=None):
        valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        self.cover_path = sorted(
            [
                os.path.join(cover_dir, f)
                for f in os.listdir(cover_dir)
                if f.lower().endswith(valid_ext)
            ]
        )

        self.secret_path = sorted(
            [
                os.path.join(secret_dir, f)
                for f in os.listdir(secret_dir)
                if f.lower().endswith(valid_ext)
            ]
        )

        print(
            f"Cover path: {os.path.abspath(cover_dir)} | Found: {len(self.cover_path)} images"
        )
        print(
            f"Secret path: {os.path.abspath(secret_dir)} | Found: {len(self.secret_path)} images"
        )

        if len(self.cover_path) == 0 or len(self.secret_path) == 0:
            raise RuntimeError(
                "Error: One of the dataset folders is empty or paths are wrong."
            )

        self.transform = transform

    def __len__(self):
        return min(len(self.cover_path), len(self.secret_path))

    def __getitem__(self, index):
        cover = Image.open(self.cover_path[index]).convert("RGB")
        secret_path = self.secret_path[index]

        try:
            if secret_path.lower().endswith((".tif", ".tiff")):
                img = tiff.imread(secret_path)

                if len(img.shape) == 3:
                    if img.shape[0] < img.shape[2]:
                        img = img.transpose(1, 2, 0)
                    img = img[:, :, :3]

                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                secret = Image.fromarray(img).convert("RGB")
            else:
                secret = Image.open(secret_path).convert("RGB")

        except Exception as e:
            print(f"Error loading {secret_path}: {e}")
            secret = Image.new("RGB", (256, 256), (0, 0, 0))

        if self.transform:
            cover = self.transform(cover)
            secret = self.transform(secret)

        return cover, secret


class DataPipeline:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.device_manager = DeviceManager()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def get_train_val_loaders(self, cover_dir, secret_dir, val_split=0.2, seed=42):
        """Build one dataset and split it into non-overlapping train/val sets."""
        dataset = StegoDataset(cover_dir, secret_dir, transform=self.transform)
        total = len(dataset)
        indices = list(range(total))

        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        split = int(total * val_split)
        val_indices = indices[:split]
        train_indices = indices[split:]

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        workers = self.device_manager.get_optimal_workers()
        print(
            f"Dataset split: {len(train_indices)} train / {len(val_indices)} val  "
            f"({workers} workers)"
        )

        train_loader = DataLoader(
            train_set,
            self.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device_manager.is_cuda,
        )
        val_loader = DataLoader(
            val_set,
            self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.device_manager.is_cuda,
        )
        return train_loader, val_loader
