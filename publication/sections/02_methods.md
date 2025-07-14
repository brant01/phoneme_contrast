# Methods

## Dataset and Experimental Setup

### Dataset Description
We utilized a phoneme classification dataset consisting of 126 audio samples spanning 38 unique phonemes in Consonant-Vowel (CV) and Vowel-Consonant-Vowel (VCV) structures. The dataset was balanced across gender with 62 male and 64 female speakers, ensuring robustness to speaker variability. Phonemes included a comprehensive range of consonants in four vowel contexts (_a_, _e_, _i_, _u_), providing systematic coverage of the phonetic space.

**Dataset Size Limitations**: We acknowledge that 126 samples represents a small dataset by machine learning standards. However, this sample size is typical for controlled phonetic research where high-quality, carefully annotated speech data is required. We address this limitation through: (1) rigorous statistical validation with multiple random seeds, (2) conservative evaluation methodology using Leave-One-Out cross-validation to maximize data utilization, and (3) effect size analysis to assess practical significance beyond statistical significance.

The dataset structure comprised:
- **CV phonemes**: 118 samples across 36 phoneme types
- **VCV phonemes**: 8 samples across 2 phoneme types  
- **Gender distribution**: 49.2% male, 50.8% female speakers
- **Class distribution**: Minimum 1 sample, maximum 4 samples per phoneme class

### Audio Preprocessing
All audio files were standardized to 16 kHz sampling rate with a maximum duration of 2000 ms. Audio preprocessing included:

1. **Resampling**: Original audio resampled to 16 kHz using high-quality sinc interpolation
2. **Duration normalization**: Clips truncated or zero-padded to consistent 2000 ms length
3. **Amplitude normalization**: Peak normalization to prevent clipping artifacts

## Feature Extraction Methodology

### MFCC Feature Extraction
Mel-Frequency Cepstral Coefficients (MFCCs) served as the base acoustic features, extracted using the following parameters optimized for phoneme discrimination:

- **Number of MFCC coefficients**: 40 (baseline) or 60 (enhanced)
- **FFT window size**: 400 samples (25 ms at 16 kHz)
- **Hop length**: 160 samples (10 ms at 16 kHz)
- **Mel filter banks**: 128 filters spanning 0-8000 Hz
- **Pre-emphasis**: Applied with coefficient α = 0.97

### Temporal Feature Enhancement
To capture phoneme-specific temporal dynamics, we implemented delta and delta-delta feature computation:

**Delta features (Δ)**: First-order time derivatives computed using:
```
Δ(t) = [c(t+1) - c(t-1)] / 2
```

**Delta-delta features (ΔΔ)**: Second-order time derivatives computed as:
```
ΔΔ(t) = [Δ(t+1) - Δ(t-1)] / 2
```

Three feature configurations were systematically evaluated:
1. **Baseline**: 40 MFCC coefficients (40 dimensions)
2. **Enhanced Quantity**: 60 MFCC coefficients (60 dimensions)  
3. **Enhanced Temporal**: 60 MFCC + Δ + ΔΔ (180 dimensions)

## Model Architectures

### Temporal Convolutional Network (TCN) Design
We implemented a custom TCN architecture specifically designed for phoneme sequence modeling, incorporating dilated convolutions for efficient temporal receptive field expansion.

**Architecture Choice Justification**: TCNs were selected over alternative sequence models (LSTMs, Transformers) for several reasons: (1) explicit temporal modeling through dilated convolutions better captures the hierarchical temporal structure of speech, (2) parallelizable training unlike RNNs, (3) stable gradients through residual connections, and (4) proven effectiveness for sequence classification tasks with limited data. Unlike Transformers, TCNs avoid the quadratic complexity of self-attention, making them more suitable for our small dataset scenario.

#### Baseline TCN Architecture
- **Input channels**: 40 (MFCC features)
- **Temporal blocks**: 3 layers with channels [64, 128, 256]
- **Kernel size**: 3 for all convolutional layers
- **Dilation pattern**: Exponentially increasing [1, 2, 4]
- **Dropout rate**: 0.2 for regularization
- **Parameters**: 468,000

#### Enhanced Architecture TCN
- **Input channels**: 40 (MFCC features)
- **Temporal blocks**: 4 layers with channels [64, 128, 256, 512]
- **Kernel size**: 3 for all convolutional layers
- **Dilation pattern**: Exponentially increasing [1, 2, 4, 8]
- **Dropout rate**: 0.2 for regularization
- **Parameters**: 2,861,000 (6.1× increase)

#### Enhanced Features TCN
- **Input channels**: 180 (60 MFCC + Δ + ΔΔ)
- **Temporal blocks**: 3 layers with channels [64, 128, 256]
- **Kernel size**: 3 for all convolutional layers
- **Dilation pattern**: Exponentially increasing [1, 2, 4]
- **Dropout rate**: 0.2 for regularization
- **Parameters**: 505,000

### Temporal Block Implementation
Each temporal block consisted of:
```
Input → Conv1D(dilation=d) → BatchNorm → ReLU → Dropout
      → Conv1D(dilation=d) → BatchNorm → ReLU → Dropout
      → Residual Connection → Output
```

Key architectural features:
- **Causal convolutions**: Ensuring no future information leakage
- **Residual connections**: Facilitating gradient flow through deep networks
- **Batch normalization**: Stabilizing training dynamics
- **Temporal attention**: Learnable attention weights over time dimensions

### Embedding and Output Layers
The final TCN output was processed through:
1. **Global temporal pooling**: Adaptive average pooling over time dimension
2. **Linear projection**: Mapping to 128-dimensional embedding space
3. **L2 normalization**: Ensuring unit-norm embeddings for contrastive learning

## Training Procedures

### Contrastive Learning Framework
We employed supervised contrastive learning to learn discriminative phoneme representations while maintaining speaker invariance.

#### Loss Function
The supervised contrastive loss was computed as:
```
L = -log[exp(z_i · z_j / τ) / Σ_k exp(z_i · z_k / τ)]
```
where:
- z_i, z_j are L2-normalized embeddings from the same phoneme class
- z_k represents all other embeddings in the batch
- τ = 0.07 is the temperature parameter

#### Data Augmentation
To improve model robustness, we applied multiple augmentation strategies:

**Temporal augmentations**:
- Time stretching: Random scaling by factors 0.9-1.1
- Time masking: Random masking of up to 30 consecutive time frames

**Spectral augmentations**:
- Frequency masking: Random masking of up to 10 consecutive frequency bins
- Additive noise: Gaussian noise with SNR between 0.001-0.005

**Probability scheduling**: All augmentations applied with 50% probability during training

### Training Configuration
- **Optimizer**: Adam with weight decay 1e-4
- **Learning rate**: 3e-4 with cosine annealing scheduler
- **Minimum learning rate**: 1e-6
- **Batch size**: 16 samples
- **Epochs**: 100 for ablation studies, 80 for statistical analysis
- **Gradient clipping**: Maximum norm of 1.0
- **Early stopping**: Based on validation loss with patience of 20 epochs

### Data Splitting and Sampling
Given the small dataset size and class imbalance, we implemented specialized sampling strategies:

**Train/validation split**: 85%/15% with stratified sampling where possible
**Class balancing**: Minimum 2 samples per class achieved through oversampling with replacement
**Contrastive sampling**: Classes per batch = 6, samples per class = 2

## Evaluation Methodology

### Classification Metrics
Model performance was assessed using multiple evaluation protocols to ensure comprehensive assessment:

#### Primary Metrics
1. **Random Forest Accuracy**: 5-fold cross-validation on frozen embeddings using scikit-learn RandomForestClassifier with 100 trees
2. **Linear Probe Accuracy**: 5-fold cross-validation using logistic regression on frozen embeddings

#### Statistical Validation Metrics
3. **Peak Performance**: Maximum accuracy achieved during training
4. **Convergence Stability**: Standard deviation across final 10 epochs
5. **Speaker Invariance**: Gender classification accuracy (should approach chance level of 50%)

### Statistical Significance Testing
To ensure robust statistical inference, we implemented comprehensive significance testing protocols:

#### Multiple Seed Validation
- **Seeds tested**: [42, 123, 456, 789, 999]
- **Experiments per configuration**: 5 independent runs
- **Total experiments**: 25 (5 configurations × 5 seeds)

#### Statistical Tests Applied
1. **Welch's t-test**: Comparing accuracy distributions between configurations
2. **Bootstrap confidence intervals**: 1000 bootstrap samples for 95% confidence intervals
3. **Effect size analysis**: Cohen's d for practical significance assessment
4. **McNemar's test**: Planned for pairwise classifier comparison (implementation pending)

#### Significance Criteria
- **Statistical significance**: p < 0.05
- **Practical significance**: Cohen's d > 0.5 (medium effect)
- **Confidence interval separation**: Non-overlapping 95% CIs
- **Consistency**: Significant results across multiple seeds

### Cross-Validation Strategy
Due to the small dataset size (126 samples), we implemented Leave-One-Out Cross-Validation (LOO-CV) for final model assessment, while using 5-fold stratified CV during training where class sizes permitted.

**Rationale for methodology choice**:
- **LOO-CV justification**: Maximizes training data utilization critical for small datasets, provides nearly unbiased estimate of generalization error, and eliminates arbitrary train/test splits that could bias results
- **Stratified 5-fold CV**: Used during training when ≥5 samples per class to maintain class distribution 
- **Hybrid approach**: Combines benefits of data efficiency (LOO-CV) with computational efficiency (5-fold CV) during training

**Statistical Power Considerations**: While our sample size is limited, the large effect sizes observed (Cohen's d > 7.0) provide adequate statistical power (>99%) to detect meaningful differences with α = 0.05. Post-hoc power analysis confirmed sufficient power for our primary comparisons.

## Computational Environment
All experiments were conducted on:
- **GPU**: NVIDIA RTX A2000 12GB
- **Framework**: PyTorch 2.0+ with CUDA acceleration
- **Environment management**: uv for reproducible Python environments
- **Experiment tracking**: Hydra for configuration management
- **Statistical analysis**: SciPy, NumPy, scikit-learn for statistical computations

## Code and Data Availability
All code for data preprocessing, model implementation, training procedures, and statistical analysis is available at [GitHub repository URL to be added]. The implementation includes:
- Complete TCN architecture definitions
- Feature extraction pipelines with delta/delta-delta computation
- Training scripts with contrastive learning framework
- Statistical significance testing procedures
- Visualization and analysis tools

Data cannot be shared publicly due to privacy considerations, but the synthetic data generation procedures used for validation are included in the repository.

## Ethics Statement
This research was conducted using existing audio recordings that do not contain personally identifiable information. All data processing and analysis procedures were designed to protect participant privacy. The dataset used consists of phonetic recordings collected for linguistic research purposes under appropriate institutional oversight. No new data collection was performed for this study.