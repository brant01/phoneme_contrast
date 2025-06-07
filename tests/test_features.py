# src/datasets/features.py
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, samples] or [samples]
        Returns:
            features: [batch, channels, features, time]
        """
        raise NotImplementedError


class MFCCExtractor(FeatureExtractor):
    """Extract MFCC features from waveform."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        # Create MFCC transform
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": f_min,
                "f_max": f_max or sample_rate / 2,
            },
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, samples] or [samples]
        Returns:
            mfcc: [batch, 1, n_mfcc, time_frames]
        """
        # Handle different input shapes
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [batch, 1, samples]

        # Compute MFCC
        mfcc = self.mfcc(waveform)  # [batch, n_mfcc, time]

        # Add channel dimension for compatibility with CNNs
        mfcc = mfcc.unsqueeze(1)  # [batch, 1, n_mfcc, time]

        return mfcc


class MelSpectrogramExtractor(FeatureExtractor):
    """Extract log mel spectrogram features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate / 2,
        )

        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, samples] or [samples]
        Returns:
            log_mel: [batch, 1, n_mels, time_frames]
        """
        # Handle different input shapes
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Compute mel spectrogram
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)

        # Add channel dimension
        log_mel = log_mel.unsqueeze(1)

        return log_mel


def build_feature_extractor(config: Dict) -> FeatureExtractor:
    """Build feature extractor from config."""

    extractor_type = config.get("type", "mfcc")

    if extractor_type == "mfcc":
        params = config.get("mfcc_params", {})
        return MFCCExtractor(**params)
    elif extractor_type == "mel":
        params = config.get("mel_params", {})
        return MelSpectrogramExtractor(**params)
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}")
