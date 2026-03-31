import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.core.device import DeviceManager


class ImageDirDataset(Dataset):
    """Loads all images from a single directory. Returns one image per index."""

    VALID_EXT = (".png", ".jpg", ".jpeg")

    def __init__(self, directory, transform=None):
        self.paths = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(self.VALID_EXT)
        )
        if not self.paths:
            raise RuntimeError(
                f"No images found in {os.path.abspath(directory)} "
                f"(looked for {self.VALID_EXT})"
            )
        print(f"  {os.path.abspath(directory)}: {len(self.paths)} images")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def split_batch_collate(batch):
    """Stack images and split the batch in half to form (cover, secret) pairs.

    Follows HiNet's convention: first half = covers, second half = secrets.
    Requires an even-sized batch; drops the last image if odd.
    """
    images = torch.stack(batch)
    n = len(images)
    if n % 2 != 0:
        images = images[:-1]
        n -= 1
    mid = n // 2
    return images[:mid], images[mid:]


class DataPipeline:
    def __init__(self, batch_size=16, num_workers=None):
        self.batch_size = batch_size
        self.device_manager = DeviceManager()
        self._num_workers_override = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _resolve_workers(self):
        if self._num_workers_override is not None:
            return self._num_workers_override
        return self.device_manager.get_optimal_workers()

    def get_loaders(self, train_dir, val_dir):
        """Build train and val DataLoaders from two separate image directories.

        Each loader yields (cover, secret) tuples via split_batch_collate,
        so batch_size must be even (each side gets batch_size // 2 images).
        """
        assert self.batch_size % 2 == 0, (
            f"batch_size must be even for HiNet-style split, got {self.batch_size}"
        )

        train_set = ImageDirDataset(train_dir, transform=self.train_transform)
        val_set = ImageDirDataset(val_dir, transform=self.test_transform)

        workers = self._resolve_workers()
        persist = workers > 0

        print(
            f"  Train: {len(train_set)} images | Val: {len(val_set)} images | "
            f"Batch: {self.batch_size} ({self.batch_size // 2} pairs) | "
            f"Workers: {workers}"
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device_manager.is_cuda,
            persistent_workers=persist,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.device_manager.is_cuda,
            persistent_workers=persist,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
        return train_loader, val_loader

    def get_val_loader(self, val_dir):
        """Build a single validation DataLoader (for evaluation only)."""
        assert self.batch_size % 2 == 0, (
            f"batch_size must be even for HiNet-style split, got {self.batch_size}"
        )

        val_set = ImageDirDataset(val_dir, transform=self.test_transform)
        workers = self._resolve_workers()
        persist = workers > 0

        print(f"  Val: {len(val_set)} images | Workers: {workers}")

        return DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.device_manager.is_cuda,
            persistent_workers=persist,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
