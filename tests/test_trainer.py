# tests/test_trainer.py
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.models import model_registry
from src.training.losses import SupervisedContrastiveLoss
from src.training.trainer import ContrastiveTrainer
from src.utils.logging import create_logger


class TestContrastiveTrainer:
    """Test the trainer."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""

        class SimpleDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                # Random data
                views = torch.randn(2, 1, 40, 50)  # 2 views
                label = idx % 10  # 10 classes
                return {"views": views, "label": label, "index": idx}

        return SimpleDataset()

    @pytest.fixture
    def trainer_components(self, simple_dataset):
        """Create components for trainer."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir)

        # Create model
        model_config = {"embedding_dim": 64, "use_attention": False}
        model = model_registry.create("phoneme_cnn", model_config)

        # Create data loaders
        train_loader = DataLoader(simple_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(simple_dataset, batch_size=16, shuffle=False)

        # Create loss function
        loss_fn = SupervisedContrastiveLoss(temperature=0.5)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create logger
        logger = create_logger(output_dir / "logs")

        # Training config
        config = {"eval_every": 1, "save_every": 2, "gradient_clip_val": 1.0}

        return {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": None,
            "device": torch.device("cpu"),
            "config": config,
            "output_dir": output_dir,
            "logger": logger,
        }

    def test_trainer_creation(self, trainer_components):
        """Test creating a trainer."""
        trainer = ContrastiveTrainer(**trainer_components)

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.checkpoint_dir.exists()

    def test_train_epoch(self, trainer_components):
        """Test training for one epoch."""
        trainer = ContrastiveTrainer(**trainer_components)

        # Run one epoch
        trainer.train(num_epochs=1)

        # Check that metrics were recorded
        assert "train_loss" in trainer.metrics_history
        assert len(trainer.metrics_history["train_loss"]) == 1

        # Check that validation was run
        assert "val_loss" in trainer.metrics_history
