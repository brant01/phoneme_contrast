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
    """Extract MFCC features from waveform with optional delta and delta-delta features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        add_delta: bool = False,
        add_delta_delta: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.add_delta = add_delta
        self.add_delta_delta = add_delta_delta

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

        # Delta computation for temporal derivatives
        if self.add_delta or self.add_delta_delta:
            self.compute_deltas = T.ComputeDeltas()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, samples] or [samples]
        Returns:
            features: [batch, 1, n_features, time_frames]
                     where n_features = n_mfcc * (1 + add_delta + add_delta_delta)
        """
        # Ensure we have [batch, samples] format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        elif waveform.dim() == 3:
            # If we get [batch, 1, samples], squeeze out channel dim
            waveform = waveform.squeeze(1)  # [batch, samples]

        # Compute MFCC - expects [batch, samples]
        mfcc = self.mfcc(waveform)  # [batch, n_mfcc, time]

        features = [mfcc]  # Start with base MFCC features

        # Add delta features (first derivative)
        if self.add_delta:
            delta_mfcc = self.compute_deltas(mfcc)  # [batch, n_mfcc, time]
            features.append(delta_mfcc)

        # Add delta-delta features (second derivative)
        if self.add_delta_delta:
            if self.add_delta:
                # Compute delta-delta from delta features
                delta_delta_mfcc = self.compute_deltas(features[1])  # Use delta features
            else:
                # Compute delta first, then delta-delta
                delta_mfcc = self.compute_deltas(mfcc)
                delta_delta_mfcc = self.compute_deltas(delta_mfcc)
            features.append(delta_delta_mfcc)

        # Concatenate all features along the feature dimension
        combined_features = torch.cat(features, dim=1)  # [batch, total_features, time]

        # Add channel dimension for CNN compatibility
        combined_features = combined_features.unsqueeze(1)  # [batch, 1, total_features, time]

        return combined_features


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
        # Ensure we have [batch, samples] format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [batch, samples]

        # Compute mel spectrogram
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)

        # Add channel dimension
        log_mel = log_mel.unsqueeze(1)  # [batch, 1, n_mels, time]

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
