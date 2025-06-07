# scripts/debug_shapes.py
import torch

from src.datasets.features import MFCCExtractor

# Create extractor
extractor = MFCCExtractor(n_mfcc=40)

# Test input shapes
test_inputs = [
    torch.randn(16000),  # 1D: raw waveform
    torch.randn(1, 16000),  # 2D: [channels, samples]
    torch.randn(1, 1, 16000),  # 3D: [batch, channels, samples]
]

for i, inp in enumerate(test_inputs):
    out = extractor(inp)
    print(f"Input shape {inp.shape} -> Output shape {out.shape}")
