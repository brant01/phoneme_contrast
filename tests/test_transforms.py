# tests/test_transforms.py
import pytest
import torch
from src.datasets.transforms import (
    TimeMask, FrequencyMask, GaussianNoise, 
    Compose, build_augmentation_pipeline
)


class TestTransforms:
    """Test individual transforms."""
    
    @pytest.fixture
    def sample_spectrogram(self):
        """Create a sample spectrogram [batch, channel, freq, time]."""
        return torch.randn(1, 1, 40, 100)
    
    def test_time_mask(self, sample_spectrogram):
        """Test time masking."""
        transform = TimeMask(max_width=10, prob=1.0)
        output = transform(sample_spectrogram, seed=42)
        
        # Should have same shape
        assert output.shape == sample_spectrogram.shape
        
        # Should have some masked values (zeros)
        assert (output == 0).any()
    
    def test_frequency_mask(self, sample_spectrogram):
        """Test frequency masking."""
        transform = FrequencyMask(max_width=5, prob=1.0)
        output = transform(sample_spectrogram, seed=42)
        
        assert output.shape == sample_spectrogram.shape
        assert (output == 0).any()
    
    def test_gaussian_noise(self, sample_spectrogram):
        """Test Gaussian noise addition."""
        transform = GaussianNoise(min_snr=0.01, max_snr=0.02, prob=1.0)
        output = transform(sample_spectrogram, seed=42)
        
        assert output.shape == sample_spectrogram.shape
        # Should be different from input
        assert not torch.allclose(output, sample_spectrogram)
    
    def test_deterministic_with_seed(self, sample_spectrogram):
        """Test that transforms are deterministic with seed."""
        transform = GaussianNoise(prob=1.0)
        
        output1 = transform(sample_spectrogram, seed=42)
        output2 = transform(sample_spectrogram, seed=42)
        
        assert torch.allclose(output1, output2)
    
    def test_probabilistic_application(self, sample_spectrogram):
        """Test that transforms respect probability."""
        transform = TimeMask(prob=0.0)  # Never apply
        output = transform(sample_spectrogram, seed=42)
        
        # Should be unchanged
        assert torch.allclose(output, sample_spectrogram)


class TestCompose:
    """Test transform composition."""
    
    def test_compose_multiple(self):
        """Test composing multiple transforms."""
        transforms = [
            GaussianNoise(prob=1.0),
            TimeMask(prob=1.0),
        ]
        compose = Compose(transforms)
        
        x = torch.randn(1, 1, 40, 100)
        output = compose(x, seed=42)
        
        # Should be modified
        assert not torch.allclose(output, x)
        assert output.shape == x.shape


class TestBuildPipeline:
    """Test building pipeline from config."""
    
    def test_build_from_config(self):
        """Test building augmentation pipeline from config."""
        config = {
            "time_mask": {"enabled": True, "max_width": 20},
            "freq_mask": {"enabled": True, "max_width": 5},
            "noise": {"enabled": False}
        }
        
        pipeline = build_augmentation_pipeline(config)
        
        # Should have 2 transforms (noise disabled)
        assert len(pipeline.transforms) == 2
        assert isinstance(pipeline.transforms[0], TimeMask)
        assert isinstance(pipeline.transforms[1], FrequencyMask)