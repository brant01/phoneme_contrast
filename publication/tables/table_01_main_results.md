# Table 1: Primary Performance Comparison

| Approach | RF Accuracy (%) | Linear Accuracy (%) | Peak RF (%) | Parameters | Training Time (min) |
|----------|-----------------|--------------------|--------------|-----------|--------------------|
| Baseline TCN | 85.9 ± 1.3 | 49.7 ± 0.9 | 89.0 ± 1.8 | 468K | 12.3 ± 0.8 |
| Enhanced Architecture TCN | 82.8 ± 1.1 | 36.8 ± 1.2 | 83.5 ± 1.5 | 2,861K | 18.7 ± 1.2 |
| **Enhanced Features TCN** | **95.1 ± 0.4** | **73.6 ± 0.6** | **95.7 ± 0.3** | **505K** | **13.1 ± 0.7** |

**Caption:** Primary performance comparison across three approaches to phoneme classification. Values represent mean ± standard deviation across 5 random seeds. RF = Random Forest accuracy on frozen embeddings, Linear = Linear probe accuracy, Peak RF = Maximum RF accuracy during training. Enhanced features approach achieves superior performance with computational efficiency. Bold indicates best performance in each column.