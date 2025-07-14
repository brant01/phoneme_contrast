# scripts/train.py
import logging
import platform
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets import PhonemeContrastiveDataset
from src.datasets.features import build_feature_extractor
from src.datasets.parser import parse_dataset
from src.datasets.samplers import ContrastiveBatchSampler
from src.datasets.transforms import build_augmentation_pipeline
from src.models import model_registry
from src.training.losses import get_loss_fn
from src.training.trainer import ContrastiveTrainer
from src.utils.device import get_best_device
from src.utils.logging import create_logger
from src.utils.system_resources import adjust_params_for_system


def setup_data(cfg: DictConfig, logger: logging.Logger):
    """Setup datasets and data loaders."""

    # Parse dataset
    data_path = Path(cfg.data.data_path)
    file_paths, labels, label_map, metadata = parse_dataset(data_path, logger)

    # Create feature extractor and augmentation pipeline
    feature_extractor = build_feature_extractor(cfg.data.feature_extractor)
    augmentation_pipeline = build_augmentation_pipeline(cfg.data.augmentation)

    # Split into train/val
    n_files = len(file_paths)
    n_train = int(n_files * cfg.data.train_split)
    indices = torch.randperm(n_files).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create datasets
    train_dataset = PhonemeContrastiveDataset(
        file_paths=[file_paths[i] for i in train_indices],
        labels=[labels[i] for i in train_indices],
        metadata=[metadata[i] for i in train_indices],
        feature_extractor=feature_extractor,
        augmentation_pipeline=augmentation_pipeline,
        config=dict(cfg.data),
        mode="train",
    )

    val_dataset = PhonemeContrastiveDataset(
        file_paths=[file_paths[i] for i in val_indices],
        labels=[labels[i] for i in val_indices],
        metadata=[metadata[i] for i in val_indices],
        feature_extractor=feature_extractor,
        augmentation_pipeline=augmentation_pipeline,
        config=dict(cfg.data),
        mode="val",
    )

    # Create samplers
    train_labels = [labels[i] for i in train_indices]
    train_sampler = ContrastiveBatchSampler(
        labels=train_labels,
        classes_per_batch=cfg.data.contrastive.classes_per_batch,
        samples_per_class=cfg.data.contrastive.samples_per_class,
        views_per_sample=cfg.data.contrastive.views_per_sample,
        shuffle=True,
        seed=cfg.experiment.seed,
        min_samples_to_exclude=0,  # Set to 0 to include ALL classes
    )

    # Verify all classes are included
    unique_train_classes = set(train_labels)
    logger.info(f"Total unique classes in training: {len(unique_train_classes)}")
    logger.info(f"Classes in sampler: {len(train_sampler.valid_classes)}")
    logger.info(f"Classes per batch: {cfg.data.contrastive.classes_per_batch}")
    logger.info(f"Samples per class: {cfg.data.contrastive.samples_per_class}")
    logger.info(f"Total batches per epoch: {len(train_sampler)}")

    # Determine number of workers (0 on Windows to avoid multiprocessing issues)
    num_workers = cfg.training.num_workers if platform.system() != "Windows" else 0

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    valid_classes = train_loader.batch_sampler.valid_classes
    total_phonemes = len(set(train_dataset.labels))
    logger.info(f"Using {len(valid_classes)} out of {total_phonemes} phonemes in training")

    excluded = set(train_dataset.labels) - set(valid_classes)
    if excluded:
        logger.warning(f"Excluded phonemes (insufficient samples): {sorted(excluded)}")

    logger.debug(f"Train batches: {len(train_loader)}")
    logger.debug(
        f"Batch composition: {train_loader.batch_sampler.classes_per_batch} classes × "
        f"{train_loader.batch_sampler.samples_per_class} samples × "
        f"{train_loader.batch_sampler.views_per_sample} views = {cfg.training.batch_size} samples/batch"
    )

    return train_loader, val_loader, len(label_map)


def setup_model(cfg: DictConfig, num_classes: int, device: torch.device):
    """Setup model, optimizer, and loss function."""

    # Create model
    model_config = dict(cfg.model)
    model = model_registry.create(cfg.model.type, model_config)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0),
    )

    # Create scheduler
    scheduler = None
    if cfg.training.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.get("min_lr", 1e-6)
        )

    # Create loss function
    loss_config = dict(cfg.training.loss)
    loss_type = loss_config.pop("type")
    loss_fn = get_loss_fn(loss_type, **loss_config)

    return model, optimizer, scheduler, loss_fn


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Setup logging
    output_dir = Path(cfg.experiment.output_dir)
    logger = create_logger(output_dir / "logs", console_log_level=cfg.logging.level)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seeds
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Get device
    device = get_best_device(cfg.get("device", "auto"), logger)

    # Adjust parameters for system
    cfg = adjust_params_for_system(cfg, device, logger)

    # Setup data
    logger.info("Setting up data...")
    train_loader, val_loader, num_classes = setup_data(cfg, logger)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Number of phoneme classes: {num_classes}")

    # Setup model
    logger.info("Setting up model...")
    model, optimizer, scheduler, loss_fn = setup_model(cfg, num_classes, device)

    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=OmegaConf.to_container(cfg, resolve=True),  # Pass full config
        output_dir=output_dir,
        logger=logger,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(num_epochs=cfg.training.epochs)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
