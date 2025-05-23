# src/training/metrics.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple


def compute_embedding_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics on embeddings.
    
    Args:
        embeddings: [N, D] tensor of L2-normalized embeddings
        labels: [N] tensor of labels
        
    Returns:
        Dictionary of metrics
    """
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    
    metrics = {}
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Create masks for same/different class pairs
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_positive = label_matrix.float()
    mask_negative = 1.0 - mask_positive
    
    # Remove diagonal (self-distances)
    eye = torch.eye(len(labels))
    mask_positive = mask_positive * (1 - eye)
    mask_negative = mask_negative * (1 - eye)
    
    # Compute average distances
    if mask_positive.sum() > 0:
        avg_pos_dist = (distances * mask_positive).sum() / mask_positive.sum()
        metrics['avg_positive_distance'] = avg_pos_dist.item()
    
    if mask_negative.sum() > 0:
        avg_neg_dist = (distances * mask_negative).sum() / mask_negative.sum()
        metrics['avg_negative_distance'] = avg_neg_dist.item()
    
    # Compute separation ratio
    if 'avg_positive_distance' in metrics and 'avg_negative_distance' in metrics:
        metrics['separation_ratio'] = (
            metrics['avg_negative_distance'] / 
            (metrics['avg_positive_distance'] + 1e-8)
        )
    
    return metrics


def evaluate_clustering(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_neighbors: int = 5
) -> Dict[str, float]:
    """
    Evaluate clustering quality using k-NN.
    
    Args:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of labels
        n_neighbors: Number of neighbors for k-NN
        
    Returns:
        Dictionary of metrics
    """
    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    # k-NN classification
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn, embeddings, labels, cv=5)
    
    return {
        'knn_accuracy_mean': scores.mean(),
        'knn_accuracy_std': scores.std()
    }