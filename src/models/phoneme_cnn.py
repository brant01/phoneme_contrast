# src/models/phoneme_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel
from src.models.registry import model_registry


@model_registry.register("phoneme_cnn")
class PhonemeNet(BaseModel):
    """CNN for phoneme representation learning with attention mechanism."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Extract config parameters
        self.in_channels = config.get("in_channels", 1)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.use_attention = config.get("use_attention", True)
        self.dropout_rate = config.get("dropout_rate", 0.1)

        # Build the network
        self._build_network()

        # Initialize weights
        self._initialize_weights()

    def _build_network(self) -> None:
        """Build the network architecture."""
        # Convolutional backbone
        self.conv_blocks = nn.ModuleList(
            [
                # Block 1: 1 -> 32 channels
                nn.Sequential(
                    nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),  # Downsample
                    nn.Dropout2d(self.dropout_rate),
                ),
                # Block 2: 32 -> 64 channels
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),  # Downsample
                    nn.Dropout2d(self.dropout_rate),
                ),
                # Block 3: 64 -> 128 channels
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(self.dropout_rate),
                ),
            ]
        )

        # Attention mechanism
        if self.use_attention:
            self.attention = SpatialAttention(128)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(128, self.embedding_dim), nn.BatchNorm1d(self.embedding_dim)
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights.

        Uses Kaiming initialization for Conv2d layers,
        constant initialization for BatchNorm layers,
        and normal initialization for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
        # Pass through conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)

        # Global pooling
        x = self.global_pool(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]

        # Project to embedding space
        x = self.projection(x)  # [batch, embedding_dim]

        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)

        return x


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        # Compute attention weights
        attn = self.conv(x)  # [batch, 1, H, W]
        attn = torch.sigmoid(attn)

        # Apply attention
        return x * attn


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Add shortcut
        out += self.shortcut(x)
        out = F.relu(out)

        return out


@model_registry.register("phoneme_cnn_deep")
class PhonemeNetDeep(BaseModel):
    """Deep CNN for phoneme representation learning with residual connections."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Extract config parameters
        self.in_channels = config.get("in_channels", 1)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.use_attention = config.get("use_attention", True)
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.hidden_dims = config.get("hidden_dims", [64, 128, 256, 512])
        self.use_residual = config.get("use_residual", True)

        # Build the network
        self._build_network()

        # Initialize weights
        self._initialize_weights()

    def _build_network(self) -> None:
        """Build the deep network architecture."""
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dims[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build residual blocks
        layers = []
        in_channels = self.hidden_dims[0]

        for i, out_channels in enumerate(self.hidden_dims):
            # Determine stride (downsample every other layer except first)
            stride = 1 if i == 0 else 2

            # Add residual block
            if self.use_residual:
                layers.append(ResidualBlock(in_channels, out_channels, stride, self.dropout_rate))
            else:
                # Non-residual block
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(self.dropout_rate),
                    )
                )

            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Attention mechanism
        if self.use_attention:
            self.attention = SpatialAttention(self.hidden_dims[-1])

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        # Initial convolution
        x = self.init_conv(x)

        # Pass through residual blocks
        x = self.conv_blocks(x)

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Project to embedding space
        x = self.projection(x)

        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)

        return x
