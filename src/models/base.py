# src/models/base.py
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            embeddings: Output tensor [batch, embedding_dim]
        """
        raise NotImplementedError
        
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings."""
        return self.config.get("embedding_dim", 128)