"""
Helper functions to handle stratification with small class sizes.
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from typing import Tuple, Optional


def safe_train_test_split(X, y, test_size=0.2, random_state=42, min_samples_per_class=2):
    """
    Perform train-test split handling classes with few samples.
    
    Strategy:
    1. If we can do stratified split safely, do it
    2. Otherwise, use random split without stratification
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    n_samples = len(y)
    
    # Calculate expected test size
    if test_size < 1:
        n_test_samples = int(n_samples * test_size)
    else:
        n_test_samples = int(test_size)
    
    # Check if stratified split is feasible
    # Need at least 1 sample per class in test set
    can_stratify = (n_test_samples >= n_classes) and np.all(class_counts >= 2)
    
    if can_stratify:
        # Safe to use stratified split
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    else:
        # Use random split without stratification
        print(f"Warning: Cannot use stratified split ({n_classes} classes, {n_test_samples} test samples)")
        print(f"Classes with <2 samples: {np.sum(class_counts < 2)}")
        print("Using random split instead")
        
        return train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)


def get_cv_splitter(y, n_splits=5, min_samples_per_class=2):
    """
    Get appropriate cross-validation splitter based on class distribution.
    
    Returns:
    - StratifiedKFold if possible
    - LeaveOneOut if too few samples per class
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = np.min(class_counts)
    
    if min_count >= n_splits:
        # Can use StratifiedKFold
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        # Use LeaveOneOut
        print(f"Using LeaveOneOut CV (min class size: {min_count} < {n_splits})")
        return LeaveOneOut()