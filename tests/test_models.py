# tests/test_models.py
import pytest
import torch
from src.models import model_registry, PhonemeNet


class TestModelRegistry:
    """Test the model registry."""
    
    def test_registry_registration(self):
        """Test that PhonemeNet is registered."""
        assert "phoneme_cnn" in model_registry.list()
        
    def test_registry_get(self):
        """Test getting a model class."""
        model_class = model_registry.get("phoneme_cnn")
        assert model_class == PhonemeNet
        
    def test_registry_create(self):
        """Test creating a model instance."""
        config = {"embedding_dim": 64}
        model = model_registry.create("phoneme_cnn", config)
        assert isinstance(model, PhonemeNet)
        assert model.embedding_dim == 64
        
    def test_registry_invalid_model(self):
        """Test that invalid model names raise errors."""
        with pytest.raises(ValueError):
            model_registry.get("invalid_model")


class TestPhonemeNet:
    """Test the PhonemeNet model."""
    
    @pytest.fixture
    def model_config(self):
        return {
            "in_channels": 1,
            "embedding_dim": 128,
            "use_attention": True,
            "dropout_rate": 0.1
        }
    
    @pytest.fixture
    def model(self, model_config):
        return PhonemeNet(model_config)
    
    def test_model_creation(self, model):
        """Test model initialization."""
        assert model.embedding_dim == 128
        assert model.use_attention == True
        assert len(model.conv_blocks) == 3
        
    def test_forward_pass(self, model):
        """Test forward pass with different batch sizes."""
        # Set model to eval mode to avoid BatchNorm issues with batch size 1
        model.eval()
        
        test_inputs = [
            torch.randn(1, 1, 40, 100),   # Single sample
            torch.randn(4, 1, 40, 100),   # Small batch
            torch.randn(16, 1, 40, 100),  # Larger batch
        ]
        
        with torch.no_grad():  # No need for gradients in testing
            for x in test_inputs:
                output = model(x)
                
                # Check output shape
                assert output.shape == (x.shape[0], 128)
                
                # Check L2 normalization
                norms = torch.norm(output, p=2, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
            
    def test_model_without_attention(self, model_config):
        """Test model without attention."""
        model_config["use_attention"] = False
        model = PhonemeNet(model_config)
        model.eval()  # Set to eval mode
        
        x = torch.randn(4, 1, 40, 100)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 128)
        
    def test_different_embedding_dims(self, model_config):
        """Test model with different embedding dimensions."""
        for dim in [64, 128, 256]:
            model_config["embedding_dim"] = dim
            model = PhonemeNet(model_config)
            model.eval()  # Set to eval mode
            
            x = torch.randn(2, 1, 40, 100)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, dim)

    def test_forward_pass_training(self, model):
        """Test forward pass in training mode (requires batch size > 1)."""
        model.train()
        
        # Only test with batch sizes > 1
        test_inputs = [
            torch.randn(2, 1, 40, 100),   # Min batch size for BatchNorm
            torch.randn(4, 1, 40, 100),   # Small batch
            torch.randn(16, 1, 40, 100),  # Larger batch
        ]
        
        for x in test_inputs:
            output = model(x)
            
            # Check output shape
            assert output.shape == (x.shape[0], 128)
            
            # Check L2 normalization
            norms = torch.norm(output, p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)