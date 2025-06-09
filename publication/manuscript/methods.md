# Methods

## Dataset and Preprocessing

### Speech Data
We utilized a phoneme dataset comprising 126 speech samples representing 38 unique phonemes. The dataset included recordings from both male and female speakers, enabling analysis of speaker-invariant representations. Each sample consisted of a single phoneme utterance in various phonetic contexts, including consonant-vowel (CV) and vowel-consonant-vowel (VCV) structures.

### Feature Extraction
Audio files were preprocessed using the following pipeline:
1. **Resampling**: All audio files were resampled to 16 kHz to ensure consistency
2. **Feature Extraction**: We extracted 13-dimensional Mel-frequency cepstral coefficients (MFCCs) with:
   - Window length: 25 ms
   - Hop length: 10 ms
   - Number of mel filters: 40
   - Number of cepstral coefficients: 13
3. **Normalization**: Features were normalized per utterance using z-score normalization

### Data Augmentation
To improve model robustness and create multiple views for contrastive learning, we implemented a comprehensive augmentation pipeline:
- **Gaussian Noise**: Added with probability 0.5 and SNR randomly sampled from 10-40 dB
- **Time Masking**: Random temporal segments masked with probability 0.3
- **Frequency Masking**: Random frequency bins masked with probability 0.3
- **Time Stretching**: Applied with factors between 0.8-1.2

## Model Architecture

### CNN Backbone
We employed a convolutional neural network architecture specifically designed for phoneme feature extraction:

```
Input (MFCC features) → Conv Block 1 → Conv Block 2 → Conv Block 3 → 
Spatial Attention → Global Pooling → Projection Head → L₂ Normalization
```

Each convolutional block consisted of:
- 2D Convolution (kernel size: 3×3, stride: 1)
- Batch Normalization
- ReLU activation
- Max Pooling (2×2)
- Dropout (p=0.2)

The network progressively increased channel dimensions: 32 → 64 → 128 channels.

### Spatial Attention Mechanism
Following the convolutional blocks, we incorporated a spatial attention module to focus on phonetically relevant regions:
1. Global average pooling and global max pooling along the channel dimension
2. Concatenation and passage through a 7×7 convolutional layer
3. Sigmoid activation to generate attention weights
4. Element-wise multiplication with feature maps

### Projection Head
The attended features were globally pooled and passed through a projection head:
- Linear transformation to 256 dimensions
- ReLU activation
- Linear transformation to final embedding dimension (128)
- L₂ normalization to project embeddings onto a unit hypersphere

## Training Procedure

### Supervised Contrastive Loss
We employed supervised contrastive loss to train the model:

$$\mathcal{L} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

Where:
- $z_i$ represents the L₂-normalized embedding for sample $i$
- $P(i)$ denotes positive samples (same phoneme as $i$)
- $A(i)$ represents all samples except $i$
- $\tau$ is the temperature parameter (optimized to 0.15)

### Optimization
- **Optimizer**: Adam with learning rate 3×10⁻⁴
- **Batch Size**: 64 samples
- **Epochs**: 300
- **Learning Rate Schedule**: Cosine annealing with warm restarts
- **Weight Decay**: 1×10⁻⁴

### Multi-View Training
Each training batch contained two augmented views of each sample, effectively doubling the batch size for contrastive learning. Positive pairs included both augmented views of the same sample and different samples of the same phoneme.

## Evaluation Methodology

### Downstream Classification
We evaluated the learned representations using two approaches:
1. **Linear Evaluation**: Training a linear classifier on frozen embeddings
2. **Non-linear Evaluation**: Training a Random Forest classifier with 100 estimators

### Cross-Validation Strategy
Due to the limited dataset size, we employed:
- **5-fold cross-validation** for model performance evaluation
- **Leave-one-out cross-validation** for confusion matrix generation
- Stratification by phoneme to ensure balanced folds

### Speaker Invariance Analysis
To assess speaker invariance, we conducted:
1. **Gender Classification**: Attempting to predict speaker gender from embeddings
2. **Cross-Gender Transfer**: Training on one gender and testing on the other
3. **Within-Phoneme Distance Analysis**: Comparing distances between same-phoneme samples from different genders

### Phonetic Feature Analysis
We analyzed the organization of learned representations with respect to phonetic features:
- **Place of articulation**: labial, dental, velar
- **Manner of articulation**: stop, fricative
- **Voicing**: voiced, unvoiced
- **Vowel quality**: /a/, /e/, /i/, /u/
- **Syllable structure**: CV, VCV

For each feature, we computed:
- Mean within-category and between-category distances
- Linear separability using logistic regression
- Clustering quality metrics (silhouette score, adjusted Rand index)

## Statistical Analysis

### Significance Testing
- **McNemar's Test**: For comparing paired classifier performance
- **Permutation Tests**: For assessing feature organization significance
- **Bootstrap Confidence Intervals**: 95% CIs computed with 1000 bootstrap samples

### Visualization
- **t-SNE**: For 2D visualization of learned embeddings (perplexity=30)
- **Confusion Matrices**: Both raw counts and normalized by true class
- **Distance Distributions**: Kernel density estimates of intra- vs inter-class distances

## Implementation Details

The entire framework was implemented in PyTorch 2.0, with experiments managed using Hydra for configuration management. All experiments were conducted on a single NVIDIA GPU with mixed precision training for efficiency. Code and trained models are available at [repository URL].