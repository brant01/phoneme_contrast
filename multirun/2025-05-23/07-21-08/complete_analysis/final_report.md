# Complete Multirun Analysis Report
Generated: 2025-05-28 14:36:12
Analysis Directory: C:\Users\Brantlab\projects\phoneme_contrast\multirun\2025-05-23\07-21-08

## Summary

Total Runs Analyzed: 9

### Best Run: 5
- Temperature: 0.07
- Learning Rate: 0.0003
- Best RF Accuracy: 0.829
- Best Linear Accuracy: 0.289
- Final Training Loss: 2.416

## All Runs Comparison

  run_id  temperature  learning_rate  final_loss  best_rf_acc  best_linear_acc
0      0         0.03         0.0001    2.380391     0.577273         0.129167
1      1         0.05         0.0001    2.107925     0.772917         0.233144
2      2         0.07         0.0001    3.215651     0.699053         0.300947
3      3         0.03         0.0003    1.649523     0.662879         0.165720
4      4         0.05         0.0003    1.719847     0.809848         0.233333
5      5         0.07         0.0003    2.415647     0.829167         0.288636
6      6         0.03         0.0005    0.999555     0.748674         0.153220
7      7         0.05         0.0005    1.859396     0.804545         0.245455
8      8         0.07         0.0005    2.122222     0.828977         0.313258

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
Analysis complete! All results saved to: C:\Users\Brantlab\projects\phoneme_contrast\multirun\2025-05-23\07-21-08\complete_analysis
