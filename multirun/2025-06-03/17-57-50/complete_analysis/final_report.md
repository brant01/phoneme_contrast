# Complete Multirun Analysis Report
Generated: 2025-06-07 11:35:45
Analysis Directory: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-03/17-57-50

## Summary

Total Runs Analyzed: 6

### Best Run: 4
- Temperature: 0.15
- Learning Rate: 0.0003
- Best RF Accuracy: 0.933
- Best Linear Accuracy: 0.473
- Final Training Loss: 2.802

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.15         0.0003    2.973647     0.914773         0.441856
1      1         0.15         0.0003    2.921415     0.927083         0.453977
2      2         0.15         0.0003    3.039665     0.871970         0.411174
3      3         0.15         0.0003    2.940003     0.884470         0.374242
4      4         0.15         0.0003    2.802212     0.932765         0.472538
5      5         0.15         0.0003    2.858267     0.884091         0.466477

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
Analysis complete! All results saved to: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-03/17-57-50/complete_analysis
