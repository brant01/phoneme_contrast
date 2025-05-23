# src/datasets/samplers.py
from torch.utils.data import Sampler
import numpy as np
from typing import List, Iterator
from collections import defaultdict


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
        seed: int = 42
    ):
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.views_per_sample = views_per_sample
        self.shuffle = shuffle
        self.seed = seed
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            
        # Filter classes with enough samples
        self.valid_classes = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= samples_per_class
        ]
        
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
                if self.shuffle:
                    sampled = self.rng.choice(
                        class_indices, 
                        size=self.samples_per_class,
                        replace=len(class_indices) < self.samples_per_class
                    )
                else:
                    sampled = class_indices[:self.samples_per_class]
                    
                batch_indices.extend(sampled)
                
            yield batch_indices
            
    def __len__(self) -> int:
        return len(self.valid_classes) // self.classes_per_batch