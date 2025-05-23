# src/models/registry.py
from typing import Dict, Type, Callable
from src.models.base import BaseModel


class ModelRegistry:
    """Registry for models."""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}
        
    def register(self, name: str) -> Callable:
        """Decorator to register a model."""
        def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
            if name in self._models:
                raise ValueError(f"Model {name} already registered")
            self._models[name] = cls
            return cls
        return decorator
        
    def get(self, name: str) -> Type[BaseModel]:
        """Get a model class by name."""
        if name not in self._models:
            raise ValueError(
                f"Model {name} not found. Available: {list(self._models.keys())}"
            )
        return self._models[name]
        
    def create(self, name: str, config: Dict) -> BaseModel:
        """Create a model instance."""
        model_class = self.get(name)
        return model_class(config)
        
    def list(self) -> list[str]:
        """List all registered models."""
        return list(self._models.keys())


# Global registry instance
model_registry = ModelRegistry()