# Supplementary Methods

## Detailed Hyperparameter Settings

### Model Architecture Specifications

#### Baseline TCN Detailed Parameters
```yaml
model:
  type: phoneme_tcn
  in_channels: 40
  embedding_dim: 128
  num_channels: [64, 128, 256]
  kernel_size: 3
  dropout_rate: 0.2
  use_attention: true
  dilation_pattern: [1, 2, 4]  # Exponential dilation
```

#### Enhanced Architecture TCN Parameters  
```yaml
model:
  type: phoneme_tcn
  in_channels: 40
  embedding_dim: 128
  num_channels: [64, 128, 256, 512]
  kernel_size: 3
  dropout_rate: 0.2
  use_attention: true
  dilation_pattern: [1, 2, 4, 8]  # Extended dilation
```

#### Enhanced Features TCN Parameters
```yaml
model:
  type: phoneme_tcn
  in_channels: 180  # 60 MFCC + 60 Delta + 60 Delta-Delta
  embedding_dim: 128
  num_channels: [64, 128, 256]
  kernel_size: 3
  dropout_rate: 0.2
  use_attention: true
  dilation_pattern: [1, 2, 4]
```

### Training Configuration Details

#### Optimizer Settings
```yaml
training:
  optimizer: Adam
  learning_rate: 0.0003
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
```

#### Learning Rate Schedule
```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 100  # Total epochs
  eta_min: 1e-6
  last_epoch: -1
```

#### Loss Function Parameters
```yaml
loss:
  type: supervised_contrastive
  temperature: 0.07
  base_temperature: 0.07
  contrast_mode: all
```

### Data Augmentation Specifications

#### Temporal Augmentations
```yaml
time_stretch:
  enabled: true
  min_rate: 0.9
  max_rate: 1.1
  prob: 0.5

time_mask:
  enabled: true
  max_width: 30  # frames
  prob: 0.5
```

#### Spectral Augmentations
```yaml
freq_mask:
  enabled: true
  max_width: 10  # frequency bins
  prob: 0.5

noise_injection:
  enabled: true
  min_snr: 0.001
  max_snr: 0.005
  prob: 0.3
```

## Statistical Analysis Code

### Bootstrap Confidence Interval Implementation
```python
def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for accuracy scores."""
    scores = np.array(scores)
    n_samples = len(scores)
    
    bootstrap_scores = []
    np.random.seed(42)  # Reproducibility
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_scores.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return ci_lower, ci_upper
```

### Effect Size Calculation
```python
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    group1, group2 = np.array(group1), np.array(group2)
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + 
                         (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    
    d = (mean1 - mean2) / pooled_std
    return d
```

## Computational Environment Details

### Hardware Specifications
- **GPU**: NVIDIA RTX A2000 12GB GDDR6
- **CPU**: Intel Xeon processor with 16 cores
- **RAM**: 64GB DDR4
- **Storage**: 1TB NVMe SSD

### Software Environment
```yaml
python: 3.12
pytorch: 2.0.1+cu117
numpy: 1.24.3
scipy: 1.10.1
scikit-learn: 1.3.0
hydra-core: 1.3.2
matplotlib: 3.7.1
```

### Package Management
```bash
# Complete environment reproduction
uv venv
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Feature Extraction Implementation

### MFCC Computation Details
```python
mfcc_params = {
    'n_mfcc': 60,
    'n_fft': 400,
    'hop_length': 160, 
    'n_mels': 128,
    'fmin': 0,
    'fmax': 8000,
    'window': 'hann',
    'center': True,
    'pad_mode': 'reflect',
    'power': 2.0,
    'normalized': False,
    'onesided': True
}
```

### Delta Feature Computation
```python
def compute_deltas(mfcc, width=2):
    """Compute delta features using central difference."""
    deltas = np.zeros_like(mfcc)
    
    for t in range(width, mfcc.shape[1] - width):
        deltas[:, t] = (mfcc[:, t + width] - mfcc[:, t - width]) / (2 * width)
    
    # Handle boundaries with forward/backward differences
    for t in range(width):
        deltas[:, t] = mfcc[:, 1] - mfcc[:, 0]
        deltas[:, -(t+1)] = mfcc[:, -1] - mfcc[:, -2]
    
    return deltas
```

## Cross-Validation Implementation

### Leave-One-Out Cross-Validation
```python
def loo_cross_validation(model, data, labels):
    """Implement Leave-One-Out Cross-Validation."""
    n_samples = len(data)
    predictions = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Create train/test splits
        train_idx = np.concatenate([np.arange(i), np.arange(i+1, n_samples)])
        test_idx = i
        
        # Train model
        model.fit(data[train_idx], labels[train_idx])
        
        # Test model
        predictions[i] = model.predict(data[test_idx:test_idx+1])[0]
    
    accuracy = np.mean(predictions == labels)
    return accuracy, predictions
```

## Reproducibility Information

### Random Seed Control
All experiments used fixed random seeds for reproducibility:
```python
# Python random
random.seed(42)

# NumPy random
np.random.seed(42)

# PyTorch random
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Experiment Tracking
Each experiment automatically logs:
- All hyperparameters
- Model architecture details
- Training metrics per epoch
- Final evaluation results
- Random seeds used
- Computational environment info