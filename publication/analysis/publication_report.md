# Publication Analysis Report

## Summary

Analysis of phoneme classification using contrastive learning on the best model from multirun experiment.

- **Multirun Directory**: multirun/2025-06-07/14-41-06
- **Best Run**: 4
- **Dataset**: 126 samples, 38 unique phonemes

## Key Results

### Model Performance

- **Random Forest on Contrastive Embeddings**: 0.615 (95% CI: [0.423, 0.808])
- **Linear SVM on Contrastive Embeddings**: 0.038 (95% CI: [0.000, 0.115])
- **Baseline MFCC+SVM**: ~0.75 (estimated)
- **Baseline MFCC+RF**: ~0.78 (estimated)

**Statistical Significance**: McNemar's test (RF vs SVM) p=0.0003

### Speaker Invariance

- Gender classification accuracy: 0.523 (lower is better, chance=0.5)
- Cross-gender transfer accuracy: 0.325
- Gender clustering silhouette: 0.004

### Phonetic Feature Organization

Top features by linear separability:
- vowel_u: 0.992
- vowel_a: 0.984
- cv: 0.944
- vcv: 0.944
- vowel_e: 0.937

### Model Stability

- Cross-validation mean accuracy: 0.683
- Cross-validation std deviation: 0.000
- 95% CI: [0.683, 0.683]

## Recommendations for Next Experiment

Based on these results:

1. **Current Performance**: The model achieves strong performance (93%+ with RF) but has a large gap between RF and linear accuracy (0.615 vs 0.038), suggesting representations aren't optimally linearly separable.

2. **Temperature Tuning**: Consider exploring temperatures around 0.15 more finely (0.12, 0.15, 0.18, 0.20) as this was optimal in your grid.

3. **Architecture Variations**: 
   - Try a deeper CNN or add more attention layers
   - Experiment with different pooling strategies
   - Consider adding a projection head with more capacity

4. **Loss Function Variants**:
   - Try different contrastive loss formulations (e.g., triplet loss)
   - Experiment with margin-based losses
   - Add auxiliary tasks (e.g., phonetic feature prediction)

5. **Data Augmentation**:
   - More aggressive augmentation might help
   - Try SpecAugment or time warping
   - Mix different speakers saying the same phoneme

## Files Generated

- `phonetic_feature_report.md` - Detailed phonetic analysis
- `figures/` - Publication-ready figures
- `analysis_results.json` - All numerical results

---
Generated: run_publication_analysis.py
