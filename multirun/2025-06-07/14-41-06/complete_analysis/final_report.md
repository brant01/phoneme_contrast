# Complete Multirun Analysis Report
Generated: 2025-06-08 14:53:41
Analysis Directory: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-07/14-41-06

## Summary

Total Runs Analyzed: 8

### Best Run: 7
- Temperature: 0.15
- Learning Rate: 0.0003
- Best RF Accuracy: 0.945
- Best Linear Accuracy: 0.448
- Final Training Loss: 2.981

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.15         0.0003    2.997415     0.926515         0.453409
1      1         0.15         0.0003    3.074925     0.932955         0.422917
2      2         0.15         0.0003    3.032822     0.939015         0.460795
3      3         0.15         0.0003    3.072326     0.914583         0.417045
4      4         0.15         0.0003    2.947739     0.944697         0.515341
5      5         0.15         0.0003    3.152091     0.913826         0.386742
6      6         0.15         0.0003    2.986643     0.926515         0.490530
7      7         0.15         0.0003    2.980599     0.945265         0.447538

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
Analysis complete! All results saved to: /home/brantlab/projects/phoneme_contrast/multirun/2025-06-07/14-41-06/complete_analysis
