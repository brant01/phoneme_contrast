# Abstract

## Speaker-Invariant Phoneme Recognition Through Contrastive Learning: A Deep Learning Approach

### Background
Phoneme recognition remains a fundamental challenge in speech processing, particularly when dealing with speaker variability. Traditional approaches often struggle to separate phonemic content from speaker-specific characteristics, leading to reduced generalization across different speakers. This study presents a novel deep learning framework that leverages supervised contrastive learning to develop speaker-invariant phoneme representations.

### Methods
We implemented a convolutional neural network (CNN) architecture with spatial attention mechanisms, trained using supervised contrastive loss. The model processes Mel-frequency cepstral coefficient (MFCC) features extracted from speech samples containing 38 distinct phonemes from both male and female speakers. The contrastive learning objective encourages the model to cluster same-phoneme embeddings while maintaining separation between different phonemes, regardless of speaker characteristics. We evaluated the learned representations using multiple downstream classifiers and conducted extensive analyses of phonetic feature organization and speaker invariance.

### Results
Our best model achieved 94.5% accuracy using a Random Forest classifier on the learned embeddings, demonstrating strong phoneme discrimination capability. Cross-gender transfer experiments revealed substantial speaker invariance, with gender classification accuracy near chance (52.3%) and cross-gender phoneme recognition accuracy of 32.5%. The learned representations showed hierarchical organization of phonetic features, with vowel contrasts exhibiting the strongest clustering (distance ratios: 1.19-1.47) compared to consonantal features (distance ratios: 0.99-1.02). Confusion analysis revealed systematic patterns aligned with phonetic similarity, particularly for voicing contrasts (e.g., /b/→/p/) and manner of articulation (e.g., /d/→/f/).

### Conclusions
Supervised contrastive learning successfully produces speaker-invariant phoneme representations that capture linguistically meaningful structure. The learned embeddings demonstrate strong clustering by phonetic features while remaining largely invariant to speaker gender. These findings suggest that contrastive learning objectives can effectively disentangle phonemic content from speaker-specific variation, offering promising directions for robust speech recognition systems. Future work should explore scaling to larger datasets and incorporating additional speaker variability factors beyond gender.

**Keywords:** phoneme recognition, contrastive learning, speaker invariance, deep learning, speech processing, phonetic features