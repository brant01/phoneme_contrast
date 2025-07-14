#!/usr/bin/env python3
"""Test the deep CNN model architecture."""

import torch
from omegaconf import OmegaConf

from src.models.registry import model_registry


def test_cnn_deep():
    """Test the PhonemeNetDeep model."""
    print("Testing PhonemeNetDeep model...")

    # Load config
    config = OmegaConf.create(
        {
            "type": "phoneme_cnn_deep",
            "in_channels": 1,
            "embedding_dim": 128,
            "use_attention": True,
            "dropout_rate": 0.2,
            "hidden_dims": [64, 128, 256, 512],
            "use_residual": True,
        }
    )

    # Create model
    model = model_registry.create(config.type, config)
    print(f"✓ Model created successfully: {model.__class__.__name__}")

    # Test forward pass
    batch_size = 4
    freq_bins = 40  # MFCC features
    time_steps = 200  # ~2 seconds at 16kHz with hop_length=160

    x = torch.randn(batch_size, 1, freq_bins, time_steps)
    print(f"✓ Input shape: {x.shape}")

    # Forward pass
    embeddings = model(x)
    print(f"✓ Output shape: {embeddings.shape}")
    print(f"✓ Output normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(batch_size))}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Compare with small model
    small_config = OmegaConf.create(
        {
            "type": "phoneme_cnn",
            "in_channels": 1,
            "embedding_dim": 128,
            "use_attention": True,
            "dropout_rate": 0.1,
        }
    )

    small_model = model_registry.create(small_config.type, small_config)
    small_params = sum(p.numel() for p in small_model.parameters())
    print("\nComparison with PhonemeNet (small):")
    print(f"  Small model parameters: {small_params:,}")
    print(f"  Deep model is {total_params / small_params:.1f}x larger")

    # Test different input sizes
    print("\nTesting various input sizes:")
    for time_steps in [100, 200, 400]:
        x = torch.randn(batch_size, 1, freq_bins, time_steps)
        embeddings = model(x)
        print(f"  Input: {x.shape} -> Output: {embeddings.shape} ✓")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_cnn_deep()
