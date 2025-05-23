# src/datasets/dataset.py
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import numpy as np


class PhonemeContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning of phonemes.
    
    Returns multiple augmented views of each audio file.
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        metadata: List[Dict],
        feature_extractor,
        augmentation_pipeline,
        config: Dict,
        mode: str = "train"
    ):
        """
        Args:
            file_paths: List of paths to audio files
            labels: Integer labels for each file
            metadata: Metadata dict for each file
            feature_extractor: Feature extraction module
            augmentation_pipeline: Augmentation transforms
            config: Dataset configuration
            mode: "train" or "val"
        """
        self.file_paths = file_paths
        self.labels = labels
        self.metadata = metadata
        self.feature_extractor = feature_extractor
        self.augmentation_pipeline = augmentation_pipeline
        self.config = config
        self.mode = mode
        
        # Extract key parameters
        self.target_sr = config.get("target_sr", 16000)
        self.max_length_ms = config.get("max_length_ms", 2000)
        self.max_samples = int(self.max_length_ms * self.target_sr / 1000)
        
        # Contrastive learning parameters
        self.n_views = config.get("contrastive", {}).get("views_per_sample", 2) if mode == "train" else 1
        
        # Cache for small datasets
        self.use_cache = len(file_paths) < 500
        self.waveform_cache = {} if self.use_cache else None
        
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with:
                - 'views': Tensor of shape [n_views, C, H, W] or [C, H, W]
                - 'label': Integer label
                - 'metadata': Dict with file info
                - 'index': Original index
        """
        # Load waveform (from cache if available)
        waveform = self._load_waveform(idx)
        
        # Generate multiple views
        views = []
        for view_idx in range(self.n_views):
            # Clone waveform for this view
            view_waveform = waveform.clone()
            
            # Apply waveform augmentations (time stretch, etc.)
            if self.mode == "train":
                view_waveform = self._augment_waveform(
                    view_waveform, 
                    seed=idx * 10000 + view_idx
                )
            
            # Extract features (MFCC, mel-spec, etc.)
            features = self.feature_extractor(view_waveform)  # [1, 1, freq, time]
            
            # Apply spectrogram augmentations
            if self.mode == "train" and self.augmentation_pipeline is not None:
                features = self.augmentation_pipeline(
                    features,
                    seed=idx * 20000 + view_idx
                )
            
            # Remove batch dimension, keep [C, H, W] format
            features = features.squeeze(0)  # [1, freq, time]
            views.append(features)
        
        # Stack views or return single view
        if len(views) > 1:
            views = torch.stack(views)  # [n_views, C, H, W]
        else:
            views = views[0]  # [C, H, W] for validation
        
        return {
            'views': views,
            'label': self.labels[idx],
            'metadata': self.metadata[idx],
            'index': idx
        }
    
    def _load_waveform(self, idx: int) -> torch.Tensor:
        """Load and preprocess waveform."""
        # Check cache first
        if self.use_cache and idx in self.waveform_cache:
            return self.waveform_cache[idx].clone()
        
        # Load audio - convert Path to string for torchaudio
        waveform, sr = torchaudio.load(str(self.file_paths[idx]))
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or trim to fixed length
        waveform = self._pad_or_trim(waveform)
        
        # Cache if using cache
        if self.use_cache:
            self.waveform_cache[idx] = waveform.clone()
        
        return waveform
    
    def _augment_waveform(self, waveform: torch.Tensor, seed: int) -> torch.Tensor:
        """Apply waveform-level augmentations."""
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        
        # Could add time stretching here if needed
        # For now, keep it simple
        
        return waveform
    
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to fixed length."""
        length = waveform.shape[-1]
        
        if length > self.max_samples:
            # Random crop for training, center crop for validation
            if self.mode == "train":
                start = random.randint(0, length - self.max_samples)
            else:
                start = (length - self.max_samples) // 2
            waveform = waveform[..., start:start + self.max_samples]
            
        elif length < self.max_samples:
            # Pad with zeros
            pad_total = self.max_samples - length
            if self.mode == "train":
                pad_left = random.randint(0, pad_total)
            else:
                pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        
        return waveform