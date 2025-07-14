# src/models/phoneme_tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel
from src.models.registry import model_registry


class TemporalBlock(nn.Module):
    """Temporal convolution block with dilated convolutions and residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        # First conv block
        out = self.conv1(x)
        # Remove future information by cropping
        if out.size(2) > x.size(2):
            out = out[:, :, : x.size(2)]
        out = F.relu(self.bn1(out))
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        # Remove future information by cropping
        if out.size(2) > x.size(2):
            out = out[:, :, : x.size(2)]
        out = F.relu(self.bn2(out))
        out = self.dropout2(out)

        # Add residual connection
        if self.residual is not None:
            residual = self.residual(residual)

        return F.relu(out + residual)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for TCN."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention."""
        # Compute attention weights
        attn = self.conv(x)  # [batch, 1, time]
        attn = torch.sigmoid(attn)

        # Apply attention
        return x * attn


@model_registry.register("phoneme_tcn")
class PhonemeNetTCN(BaseModel):
    """Temporal Convolutional Network for phoneme representation learning."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Extract config parameters
        self.in_channels = config.get("in_channels", 40)  # MFCC features
        self.embedding_dim = config.get("embedding_dim", 128)
        self.num_channels = config.get("num_channels", [64, 128, 256])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.use_attention = config.get("use_attention", True)

        # Build the network
        self._build_network()

        # Initialize weights
        self._initialize_weights()

    def _build_network(self) -> None:
        """Build the TCN architecture."""
        # Input projection to match expected input format
        # Input will be [batch, 1, freq, time], we need [batch, freq, time]
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),  # Keep spatial dimensions
            nn.BatchNorm2d(1),
        )

        # TCN blocks with increasing dilation
        tcn_blocks = []
        in_channels = self.in_channels

        for i, out_channels in enumerate(self.num_channels):
            dilation = 2**i  # Exponentially increasing dilation
            tcn_blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout_rate=self.dropout_rate,
                )
            )
            in_channels = out_channels

        self.tcn_blocks = nn.ModuleList(tcn_blocks)

        # Attention mechanism
        if self.use_attention:
            self.attention = TemporalAttention(self.num_channels[-1])

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(self.num_channels[-1], self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 1, freq, time]

        Returns:
            embeddings: L2-normalized embeddings [batch, embedding_dim]
        """
        # Input projection
        x = self.input_proj(x)  # [batch, 1, freq, time]

        # Reshape for TCN processing: [batch, freq, time]
        batch_size, _, freq_bins, time_steps = x.shape
        x = x.squeeze(1)  # [batch, freq, time]

        # Pass through TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)

        # Global temporal pooling
        x = self.global_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # [batch, channels]

        # Project to embedding space
        x = self.projection(x)  # [batch, embedding_dim]

        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)

        return x
