# src/datasets/samplers.py
from torch.utils.data import Sampler
import numpy as np
from typing import List, Iterator
from collections import defaultdict
from ..utils.logging import get_logger


class ContrastiveBatchSampler(Sampler[List[int]]):
    """
    Creates batches for contrastive learning.
    
    Each batch contains:
    - K different phoneme classes
    - M samples per class (from different speakers/recordings)
    - Each sample contributes N augmented views
    
    Total batch size = K * M * N
    """
    
    def __init__(
        self,
        labels: List[int],
        classes_per_batch: int,
        samples_per_class: int,
        views_per_sample: int,
        shuffle: bool = True,
        seed: int = 42,
        min_samples_to_exclude: int = 0  # Add this parameter
    ):
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.views_per_sample = views_per_sample
        self.shuffle = shuffle
        self.seed = seed
        self.min_samples_to_exclude = min_samples_to_exclude
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        # Report on class distribution
        class_counts = {label: len(indices) for label, indices in self.label_to_indices.items()}
        logger = get_logger()
        logger.info(f"Class distribution: min={min(class_counts.values())}, max={max(class_counts.values())}")
        
        # Only filter if min_samples_to_exclude > 0
        if min_samples_to_exclude > 0:
            self.valid_classes = [
                label for label, indices in self.label_to_indices.items()
                if len(indices) >= min_samples_to_exclude
            ]
            excluded = set(self.label_to_indices.keys()) - set(self.valid_classes)
            if excluded:
                logger.warning(f"Excluded classes {sorted(excluded)} with < {min_samples_to_exclude} samples")
        else:
            # Include ALL classes, even those with only 1 sample
            self.valid_classes = list(self.label_to_indices.keys())
            
            # Log classes that will need oversampling
            undersampled = [
                (label, len(indices)) 
                for label, indices in self.label_to_indices.items() 
                if len(indices) < samples_per_class
            ]
            if undersampled:
                logger.info(f"Classes requiring oversampling (sampling with replacement):")
                for label, count in sorted(undersampled):
                    logger.info(f"  Class {label}: {count} samples (need {samples_per_class})")
        
        logger.info(f"Using {len(self.valid_classes)} classes for training")
        
        self.rng = np.random.RandomState(seed)
        
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle classes
        classes = self.valid_classes.copy()
        if self.shuffle:
            self.rng.shuffle(classes)
            
        # Generate batches
        for i in range(0, len(classes), self.classes_per_batch):
            batch_classes = classes[i:i + self.classes_per_batch]
            if len(batch_classes) < self.classes_per_batch:
                continue  # Skip incomplete batches
                
            batch_indices = []
            for label in batch_classes:
                # Sample indices for this class
                class_indices = self.label_to_indices[label]
                
                # Always allow sampling with replacement for small classes
                if len(class_indices) < self.samples_per_class:
                    sampled = self.rng.choice(
                        class_indices, 
                        size=self.samples_per_class,
                        replace=True  # Sample with replacement
                    )
                else:
                    sampled = self.rng.choice(
                        class_indices, 
                        size=self.samples_per_class,
                        replace=False
                    )
                    
                batch_indices.extend(sampled)
                
            yield batch_indices
            
    def __len__(self) -> int:
        return len(self.valid_classes) // self.classes_per_batch