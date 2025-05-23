# src/datasets/transforms.py
import torch
import torchaudio.transforms as T
import random
from typing import Optional, Union


class BaseTransform:
    """Base class for augmentations."""
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError


class TimeMask(BaseTransform):
    """Randomly mask consecutive time steps."""
    
    def __init__(self, max_width: int = 30, prob: float = 0.5):
        self.max_width = max_width
        self.prob = prob
        self.masker = T.TimeMasking(time_mask_param=max_width)
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            seed = int(seed)  # Convert to Python int
            random.seed(seed)
            torch.manual_seed(seed)
            
        if random.random() < self.prob:
            return self.masker(x)
        return x


class FrequencyMask(BaseTransform):
    """Randomly mask consecutive frequency bins."""
    
    def __init__(self, max_width: int = 10, prob: float = 0.5):
        self.max_width = max_width
        self.prob = prob
        self.masker = T.FrequencyMasking(freq_mask_param=max_width)
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            seed = int(seed)  # Convert to Python int
            random.seed(seed)
            torch.manual_seed(seed)
            
        if random.random() < self.prob:
            return self.masker(x)
        return x


class GaussianNoise(BaseTransform):
    """Add Gaussian noise to the input."""
    
    def __init__(self, min_snr: float = 0.001, max_snr: float = 0.005, prob: float = 0.3):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            seed = int(seed)  # Convert to Python int
            random.seed(seed)
            torch.manual_seed(seed)
            
        if random.random() < self.prob:
            noise_level = random.uniform(self.min_snr, self.max_snr)
            noise = torch.randn_like(x) * noise_level
            return x + noise
        return x


class TimeStretch(BaseTransform):
    """Time stretch augmentation for waveforms."""
    
    def __init__(self, min_rate: float = 0.9, max_rate: float = 1.1, prob: float = 0.5):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            seed = int(seed)  # Convert to Python int
            random.seed(seed)
            torch.manual_seed(seed)
            
        if random.random() < self.prob:
            rate = random.uniform(self.min_rate, self.max_rate)
            # Time stretching is complex - for now, return as-is
            # In practice, you'd use torchaudio.functional.time_stretch
            # or librosa.effects.time_stretch
            return x
        return x


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        for i, transform in enumerate(self.transforms):
            # Use different seed for each transform but deterministic
            transform_seed = None if seed is None else seed + i * 1000
            x = transform(x, seed=transform_seed)
        return x


def build_augmentation_pipeline(config: dict) -> Compose:
    """Build augmentation pipeline from config."""
    transforms = []
    
    if config.get("time_mask", {}).get("enabled", False):
        params = config["time_mask"]
        transforms.append(TimeMask(
            max_width=params.get("max_width", 30),
            prob=params.get("prob", 0.5)
        ))
    
    if config.get("freq_mask", {}).get("enabled", False):
        params = config["freq_mask"]
        transforms.append(FrequencyMask(
            max_width=params.get("max_width", 10),
            prob=params.get("prob", 0.5)
        ))
    
    if config.get("noise", {}).get("enabled", False):
        params = config["noise"]
        transforms.append(GaussianNoise(
            min_snr=params.get("min_snr", 0.001),
            max_snr=params.get("max_snr", 0.005),
            prob=params.get("prob", 0.3)
        ))
    
    return Compose(transforms)