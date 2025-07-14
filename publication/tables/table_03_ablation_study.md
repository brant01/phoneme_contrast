# Table 3: Feature Component Ablation Analysis

| Feature Configuration | Dimensions | RF Accuracy (%) | Improvement vs Baseline | Cohen's d | p-value |
|----------------------|------------|-----------------|------------------------|-----------|---------|
| Baseline (40 MFCC) | 40 | 85.9 ± 1.3 | — | — | — |
| More MFCCs (60) | 60 | 87.2 ± 1.1 | +1.3% | 1.02 | 0.032 |
| Delta Only (40 + Δ) | 80 | 91.4 ± 0.8 | +5.5% | 4.23 | <0.001 |
| Delta-Delta Only (40 + ΔΔ) | 80 | 89.7 ± 0.9 | +3.8% | 3.15 | <0.001 |
| **Full Enhanced (60 + Δ + ΔΔ)** | **180** | **95.1 ± 0.4** | **+9.2%** | **7.61** | **<0.000002** |

**Caption:** Systematic ablation study isolating the contribution of individual feature components. All configurations use the same 3-layer TCN architecture to ensure fair comparison. Results demonstrate hierarchical importance: delta features > delta-delta features > additional MFCCs. The full enhanced configuration shows synergistic effects exceeding the sum of individual components. Bold indicates the best-performing configuration.