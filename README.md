# Phoneme Contrast Learning

Deep learning system for phoneme classification using contrastive learning to focus on phonemic differences rather than speaker characteristics.

## Overview

This project implements a contrastive learning approach to train neural networks that can distinguish between different phonemes while being invariant to speaker characteristics (male vs female voices). The system uses supervised contrastive loss to learn embeddings where phonemes are clustered by their acoustic properties rather than speaker identity.

## Key Results

- **Best Performance**: 94.5% Random Forest accuracy on phoneme classification (achieved with cnn_small model)
- **Linear Probe**: 44.8% accuracy (indicating non-linear embedding space)
- **Speaker Invariance**: Model successfully groups phonemes regardless of speaker gender
- **Phonetic Clustering**: Embeddings show linguistically meaningful confusion patterns
- **Optimal Config**: batch_size=64, temperature=0.15, lr=0.0003, embedding_dim=128, augmentation_prob=0.5

## Architecture

- **Data Pipeline**: WAV audio → resampling → MFCC/mel-spectrogram features → augmentation → contrastive views
- **Models**: 
  - **PhonemeNet** (cnn_small): Lightweight CNN (3 conv blocks) with spatial attention → global pooling → L2-normalized embeddings (304K params)
  - **PhonemeNetDeep** (cnn_deep): Deep CNN with residual connections (4 blocks: 64→128→256→512 channels) and spatial attention (4.9M params)
- **Loss**: Supervised contrastive loss (temperature=0.15) pulls same phonemes together, pushes different phonemes apart
- **Evaluation**: Linear probe and Random Forest classifiers on frozen embeddings

## Requirements

- Python 3.12+
- PyTorch, torchaudio
- hydra-core, omegaconf
- scikit-learn, soundfile
- wandb (optional, for experiment tracking)

## Setup

```bash
# Install with uv (recommended)
uv venv && uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

### Training

```bash
# Single run with default config
uv run scripts/train.py

# Train with optimal hyperparameters
uv run scripts/train.py \
  training.loss.temperature=0.15 \
  training.learning_rate=0.0003 \
  training.batch_size=64 \
  model.embedding_dim=128 \
  data.augmentation.noise.prob=0.5 \
  data.augmentation.time_stretch.prob=0.5 \
  data.augmentation.time_mask.prob=0.5

# Train deep model
uv run scripts/train.py \
  model=cnn_deep \
  training.epochs=300 \
  training.batch_size=64 \
  training.learning_rate=0.0003

# Multirun with parameter sweeps  
uv run scripts/train.py -m \
  training.loss.temperature=0.1,0.15,0.2 \
  model.embedding_dim=64,128,256

# Compare models (small vs deep)
uv run scripts/train.py -m \
  model=cnn_small,cnn_deep \
  training.epochs=300 \
  training.batch_size=64
```

### Evaluation

```bash
# Evaluate trained model
uv run scripts/evaluate.py model_path=path/to/model.pt

# Analysis of multirun results
uv run scripts/analyze_multirun_complete.py multirun/YYYY-MM-DD/HH-MM-SS/
```

### Testing

```bash
uv run pytest
```

## Configuration

Experiments are configured using Hydra. See `configs/` directory for:

- `config.yaml`: Main configuration
- `model/`: Model architectures (CNN, Conformer, etc.)
- `data/`: Dataset configurations  
- `system/`: System resource settings

## Data Structure

Place audio data in `data/` directory (excluded from git):

```
data/
├── train/
│   ├── phoneme1/
│   │   ├── speaker1_sample1.wav
│   │   └── speaker2_sample1.wav
│   └── phoneme2/
└── test/
```

## Project Structure

```
src/
├── datasets/     # Data loading, features, transforms
├── models/       # Neural network architectures  
├── training/     # Training loop, losses, metrics
└── utils/        # Logging, device management

scripts/          # Training and evaluation scripts
configs/          # Hydra configuration files
tests/           # Unit tests
```

## Analysis Outputs

The multirun analysis script generates comprehensive visualizations:

- **t-SNE Visualization**: 2D embedding space with phoneme labels and gender markers
- **Confusion Matrix**: Shows systematic phoneme confusions (e.g., voicing, manner)
- **Learning Curves**: Training/validation loss and accuracy over epochs
- **Distance Distributions**: Intra-class vs inter-class embedding distances
- **Performance Comparison**: Metrics across all experimental runs

## Best Practices

Based on extensive experiments, these hyperparameters work best:

- **Temperature**: 0.15 (for supervised contrastive loss)
- **Learning Rate**: 0.0003
- **Embedding Dimension**: 128
- **Batch Size**: 64 (outperforms 32 significantly)
- **Augmentation Probabilities**: 0.5 for all types (noise, time_stretch, time_mask)
- **Architecture**: CNN with spatial attention (cnn_small achieves best performance/efficiency tradeoff)
- **Training Duration**: 300 epochs (model converges by ~250-300)

## Common Phoneme Confusions

The model shows linguistically meaningful error patterns:

- **Voicing**: `b↔p`, `d↔t`, `g↔k` 
- **Manner**: `d↔f`, `p↔f`, `b↔f`
- **Place**: Similar articulation points confused

These confusions align with human perceptual difficulties.

## Development

This project follows modern Python practices:

- Package management: `uv`
- Formatting/linting: `ruff` (88-char line length)
- Testing: `pytest`
- Configuration: `hydra`/`omegaconf`
- Logging: Centralized logger instance

Always use `uv run` prefix for commands to ensure proper environment management.