# Complete Multirun Analysis Report
Generated: 2025-06-08 17:37:00
Analysis Directory: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-08/17-21-49

## Summary

Total Runs Analyzed: 2

### Best Run: 1
- Temperature: 0.15
- Learning Rate: 0.0003
- Best RF Accuracy: 0.915
- Best Linear Accuracy: 0.546
- Final Training Loss: 2.702

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.15         0.0003    2.944619     0.859848         0.429167
1      1         0.15         0.0003    2.702477     0.914583         0.546212

## Generated Visualizations

1. **multirun_comparison.png** - Comparison of all runs
2. **tsne_visualization.png** - t-SNE embedding visualization
3. **confusion_matrix.png** - Phoneme confusion matrix
4. **distance_distributions.png** - Intra vs inter-class distances
5. **learning_curves_best.png** - Learning curves for best run

## Files Generated

- runs_comparison.csv - Detailed comparison data
- embeddings.npz - Extracted embeddings
- analysis.log - Complete analysis log
- final_report.md - This report

## Next Steps

1. Review confusion matrix for most confusable phoneme pairs
2. Examine t-SNE clustering for phonetic feature groupings
3. Compare results with human perceptual data
4. Consider fine-tuning on specific phoneme contrasts

---
Analysis complete! All results saved to: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-08/17-21-49/complete_analysis
