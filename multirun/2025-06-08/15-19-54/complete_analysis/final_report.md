# Complete Multirun Analysis Report
Generated: 2025-06-08 17:08:00
Analysis Directory: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-08/15-19-54

## Summary

Total Runs Analyzed: 6

### Best Run: 5
- Temperature: 0.15
- Learning Rate: 0.0003
- Best RF Accuracy: 0.890
- Best Linear Accuracy: 0.423
- Final Training Loss: 3.071

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.15         0.0003    3.180302     0.877841         0.472727
1      1         0.15         0.0003    3.231937     0.878220         0.441667
2      2         0.15         0.0003    3.075783     0.846591         0.361553
3      3         0.15         0.0003    3.132686     0.822917         0.350000
4      4         0.15         0.0003    3.134731     0.877841         0.411174
5      5         0.15         0.0003    3.071134     0.889962         0.423106

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
Analysis complete! All results saved to: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-08/15-19-54/complete_analysis
