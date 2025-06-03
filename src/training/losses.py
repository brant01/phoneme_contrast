# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    https://arxiv.org/abs/2004.11362
    
    Takes embeddings and labels, computes loss that:
    - Pulls together embeddings from the same class
    - Pushes apart embeddings from different classes
    """
    
    def __init__(
        self, 
        temperature: float = 0.07, 
        base_temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction
        
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: Hidden vectors of shape [batch_size, embedding_dim].
                      Assumes L2 normalized
            labels: Ground truth labels of shape [batch_size]
            mask: Contrastive mask of shape [batch_size, batch_size], 
                  mask_{i,j}=1 if sample i and j have the same label.
                  If None, will be computed from labels.
                  
        Returns:
            A scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size == 1:
            raise ValueError('Batch size must be greater than 1 for contrastive loss')
            
        # Compute cosine similarity
        # features should already be L2 normalized
        similarity = torch.matmul(features, features.T)
        
        # Create label mask if not provided
        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        # Compute logits
        logits = similarity / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # Compute mean of log-likelihood over positive
        # Avoid nan from log(0) when mask has no positives for a sample
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
    
    Modified to work with supervised contrastive learning setup.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [batch_size, embedding_dim]
            labels: Ground truth labels (ignored for compatibility)
                   
        Returns:
            Scalar loss value
        """
        batch_size = features.shape[0]
        
        # For supervised contrastive setup, we need to create positive pairs
        # This is a simplified version that treats samples with same label as positives
        if labels is not None:
            # Use supervised contrastive approach
            device = features.device
            
            # Compute similarity matrix
            similarity = torch.matmul(features, features.T) / self.temperature
            
            # Create positive mask based on labels
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            
            # Mask out self-similarity
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
            mask = mask * logits_mask
            
            # For numerical stability
            logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
            logits = similarity - logits_max.detach()
            
            # Compute log probability
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
            
            # Compute mean of log-likelihood over positive
            mask_sum = mask.sum(1)
            mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
            
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
            
            # Loss
            loss = -mean_log_prob_pos
            
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
                
            return loss
        else:
            # Original NT-Xent implementation for pairs
            if batch_size % 2 != 0:
                raise ValueError('Batch size must be even for NT-Xent loss without labels')
                
            # ... rest of original implementation ...
            # (keeping the original pair-based logic if needed)
            raise NotImplementedError("NT-Xent without labels not implemented in this version")


# Loss function registry
_LOSSES = {
    'supervised_contrastive': SupervisedContrastiveLoss,
    'ntxent': NTXentLoss,
}


def get_loss_fn(name: str, **kwargs):
    """Get a loss function by name."""
    if name not in _LOSSES:
        raise ValueError(f"Loss {name} not found. Available: {list(_LOSSES.keys())}")
    return _LOSSES[name](**kwargs)