# Complete Multirun Analysis Report
Generated: 2025-05-29 07:53:13
Analysis Directory: C:\Users\Brantlab\projects\phoneme_contrast\multirun\2025-05-28\10-04-51

## Summary

Total Runs Analyzed: 4

### Best Run: 0
- Temperature: 0.07
- Learning Rate: 0.0003
- Best RF Accuracy: 0.866
- Best Linear Accuracy: 0.301
- Final Training Loss: 2.392

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.07         0.0003    2.391905     0.865720         0.300568
1      1         0.07         0.0003    2.285751     0.828220         0.319697
2      2         0.10         0.0003    3.417908     0.852652         0.393750
3      3         0.10         0.0003         NaN          NaN              NaN

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
Analysis complete! All results saved to: C:\Users\Brantlab\projects\phoneme_contrast\multirun\2025-05-28\10-04-51\complete_analysis
