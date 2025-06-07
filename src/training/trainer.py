# src/training/trainer.py
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class ContrastiveTrainer:
    """Trainer for contrastive learning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        device: torch.device,
        config: Dict[str, Any],
        output_dir: Path,
        logger: logging.Logger,
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            config: Training configuration
            output_dir: Directory to save outputs
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.logger = logger

        # Create output directories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history = defaultdict(list)

    def train(self, num_epochs: int) -> None:
        """Run the training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = {}
            if self.val_loader and (epoch + 1) % self.config.get("eval_every", 1) == 0:
                val_metrics = self._validate()

            # Classifier evaluation
            if self.val_loader and (epoch + 1) % self.config.get("eval_classifier_every", 5) == 0:
                classifier_metrics = self._evaluate_classifier(epoch + 1)
                val_metrics.update(classifier_metrics)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self._save_checkpoint("periodic")

            # Save best model based on validation loss or classifier accuracy
            metric_for_best = val_metrics.get(self.config.get("best_metric", "loss"), float("inf"))
            is_best = False

            if self.config.get("best_metric", "loss") == "loss":
                is_best = metric_for_best < self.best_val_loss
                if is_best:
                    self.best_val_loss = metric_for_best
            else:  # For accuracy metrics, higher is better
                is_best = metric_for_best > self.best_val_loss
                if is_best:
                    self.best_val_loss = metric_for_best

            if is_best:
                self._save_checkpoint("best")
                self.logger.info(
                    f"New best model! {self.config.get('best_metric', 'loss')}: {metric_for_best:.4f}"
                )

        # Save final checkpoint
        self._save_checkpoint("final")
        self._save_metrics()

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch in pbar:
            # Move batch to device
            views, labels = self._prepare_batch(batch)

            # Forward pass
            embeddings = self._forward_pass(views)
            loss = self.loss_fn(embeddings, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get("gradient_clip_val"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip_val"]
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        metrics = {"loss": total_loss / num_batches, "lr": self.optimizer.param_groups[0]["lr"]}

        return metrics

    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                views, labels = self._prepare_batch(batch)
                embeddings = self._forward_pass(views)
                loss = self.loss_fn(embeddings, labels)

                total_loss += loss.item()
                num_batches += 1

        metrics = {"loss": total_loss / num_batches}

        return metrics

    def _prepare_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for training."""
        views = batch["views"].to(self.device)
        labels = batch["label"].to(self.device)

        # Handle different view formats
        if views.dim() == 5:  # [batch, n_views, C, H, W]
            batch_size, n_views = views.shape[:2]
            # Flatten batch and views dimensions
            views = views.view(batch_size * n_views, *views.shape[2:])
            # Repeat labels for each view
            labels = labels.repeat_interleave(n_views)

        return views, labels

    def _forward_pass(self, views: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(views)

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log metrics to console and store in history.

        Args:
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        # Store in history
        for k, v in train_metrics.items():
            self.metrics_history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            self.metrics_history[f"val_{k}"].append(v)

        # Log to console
        log_str = f"Epoch {self.current_epoch + 1}"
        log_str += f" | Train Loss: {train_metrics['loss']:.4f}"
        if "loss" in val_metrics:
            log_str += f" | Val Loss: {val_metrics['loss']:.4f}"
        if "linear_accuracy" in val_metrics:
            log_str += f" | Linear Acc: {val_metrics['linear_accuracy']:.3f}"
        if "rf_accuracy" in val_metrics:
            log_str += f" | RF Acc: {val_metrics['rf_accuracy']:.3f}"
        log_str += f" | LR: {train_metrics['lr']:.6f}"

        self.logger.info(log_str)

    def _save_checkpoint(self, tag: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def _save_metrics(self) -> None:
        """Save metrics history to JSON file.

        Saves all accumulated training and validation metrics
        to metrics.json in the output directory.
        """
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _evaluate_classifier(self, epoch: int) -> Dict[str, float]:
        """Evaluate embeddings with simple classifiers."""
        self.model.eval()

        # Collect embeddings from BOTH train and val for better evaluation
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            # Get validation embeddings
            for batch in self.val_loader:
                views, batch_labels = self._prepare_batch(batch)
                emb = self.model(views)
                all_embeddings.append(emb.cpu())
                all_labels.extend(batch_labels.cpu().tolist())

            # Also get training embeddings (without augmentation)
            for batch in self.train_loader:
                views, batch_labels = self._prepare_batch(batch)
                # Use only first view to avoid augmentation effects
                if views.dim() == 5:  # Has multiple views
                    views = views[:, 0]  # Take first view only
                emb = self.model(views)
                all_embeddings.append(emb.cpu())
                all_labels.extend(batch_labels.cpu().tolist())

        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = np.array(all_labels)

        results = {}

        # Now we have ~126 samples, enough for evaluation
        from sklearn.model_selection import cross_val_score

        # Linear classifier
        try:
            linear_clf = LogisticRegression(max_iter=1000, random_state=42)
            linear_scores = cross_val_score(linear_clf, embeddings, labels, cv=5)
            results["linear_accuracy"] = linear_scores.mean()
        except (ValueError, RuntimeError) as e:
            self.logger.warning(f"Linear classifier failed: {e}")

        # Random Forest
        try:
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_scores = cross_val_score(rf_clf, embeddings, labels, cv=5)
            results["rf_accuracy"] = rf_scores.mean()
        except (ValueError, RuntimeError) as e:
            self.logger.warning(f"Random Forest classifier failed: {e}")

        return results
