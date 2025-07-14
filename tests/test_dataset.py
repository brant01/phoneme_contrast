# tests/test_dataset.py
from pathlib import Path

import pytest
import torch

from src.datasets import PhonemeContrastiveDataset
from src.datasets.features import MFCCExtractor
from src.datasets.parser import parse_dataset
from src.datasets.transforms import build_augmentation_pipeline


class TestPhonemeContrastiveDataset:
    """Test the contrastive dataset."""

    @pytest.fixture
    def dataset_components(self):
        """Create components needed for dataset."""
        # Parse data
        data_dir = Path("data/raw/New Stimuli 9-8-2024")
        if not data_dir.exists():
            pytest.skip("Data directory not found")

        file_paths, labels, label_map, metadata = parse_dataset(data_dir)

        # Create feature extractor
        feature_extractor = MFCCExtractor(n_mfcc=40)

        # Create augmentation pipeline
        aug_config = {
            "time_mask": {"enabled": True, "max_width": 30, "prob": 0.5},
            "freq_mask": {"enabled": True, "max_width": 10, "prob": 0.5},
        }
        augmentation_pipeline = build_augmentation_pipeline(aug_config)

        # Dataset config
        config = {"target_sr": 16000, "max_length_ms": 2000, "contrastive": {"views_per_sample": 2}}

        return {
            "file_paths": file_paths[:10],  # Use subset for testing
            "labels": labels[:10],
            "metadata": metadata[:10],
            "feature_extractor": feature_extractor,
            "augmentation_pipeline": augmentation_pipeline,
            "config": config,
        }

    def test_dataset_creation(self, dataset_components):
        """Test creating a dataset."""
        dataset = PhonemeContrastiveDataset(mode="train", **dataset_components)

        assert len(dataset) == 10
        assert dataset.n_views == 2

    def test_dataset_getitem_train(self, dataset_components):
        """Test getting items in train mode."""
        dataset = PhonemeContrastiveDataset(mode="train", **dataset_components)

        item = dataset[0]

        # Check structure
        assert "views" in item
        assert "label" in item
        assert "metadata" in item
        assert "index" in item

        # Check shapes
        views = item["views"]
        assert views.dim() == 4  # [n_views, C, H, W]
        assert views.shape[0] == 2  # 2 views
        assert views.shape[1] == 1  # 1 channel
        assert views.shape[2] == 40  # 40 MFCCs

        # Check label
        assert isinstance(item["label"], int)
        assert 0 <= item["label"] < 38  # We have 38 phonemes

    def test_dataset_getitem_val(self, dataset_components):
        """Test getting items in validation mode."""
        dataset = PhonemeContrastiveDataset(mode="val", **dataset_components)

        item = dataset[0]
        views = item["views"]

        # Should have single view for validation
        assert views.dim() == 3  # [C, H, W]
        assert views.shape[0] == 1  # 1 channel

    def test_deterministic_augmentation(self, dataset_components):
        """Test that augmentations are deterministic with same index."""
        dataset = PhonemeContrastiveDataset(mode="train", **dataset_components)

        # Get same item twice
        item1 = dataset[0]
        item2 = dataset[0]

        # Views should be identical (same seed used)
        assert torch.allclose(item1["views"], item2["views"])
