import torch
import os
import platform


class DeviceManager:
    """
    Handles hardware acceleration detection for Apple Silicon (MPS),
    Windows/Linux CUDA, and CPU fallback.
    """

    def __init__(self):
        self.device = self._detect_device()
        self.is_cuda = self.device.type == "cuda"
        self.is_mps = self.device.type == "mps"

        if self.is_cuda:
            torch.backends.cudnn.benchmark = True

    def _detect_device(self):
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS is available. Using Apple Silicon GPU.")
            return torch.device("mps")
        else:
            print("No GPU available. Using CPU.")
            return torch.device("cpu")

    @staticmethod
    def get_optimal_workers():
        cores = os.cpu_count() or 1
        system = platform.system()
        if system == "Windows":
            return min(4, max(1, cores - 1))
        else:
            return min(8, max(1, cores - 1))

    def get_scaler(self):
        if self.is_cuda:
            return torch.amp.GradScaler("cuda")
        return None
