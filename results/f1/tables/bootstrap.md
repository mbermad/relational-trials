| Task | Question | A | B | Split | A − B (better = +) | 95% CI | P(A better) |
|---|---|---|---|---|---|---|---|
| driver-dnf | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | -0.051 | [-0.080, -0.021] | 0.001 |
| driver-dnf | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | -0.052 | [-0.074, -0.030] | 0.000 |
| driver-dnf | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | val | -0.082 | [-0.122, -0.043] | 0.000 |
| driver-dnf | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | test | -0.001 | [-0.027, +0.025] | 0.482 |
| driver-dnf | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.051 | [+0.019, +0.085] | 1.000 |
| driver-dnf | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.006 | [-0.009, +0.023] | 0.784 |
| driver-dnf | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | val | +0.035 | [+0.005, +0.064] | 0.992 |
| driver-dnf | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | test | -0.041 | [-0.060, -0.022] | 0.000 |
| driver-dnf | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | val | +0.004 | [-0.015, +0.022] | 0.651 |
| driver-dnf | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | test | +0.011 | [-0.005, +0.027] | 0.900 |
| driver-dnf | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | -0.029 | [-0.051, -0.006] | 0.004 |
| driver-dnf | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | -0.012 | [-0.028, +0.004] | 0.081 |
| driver-dnf | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | -0.143 | [-0.187, -0.098] | 0.000 |
| driver-dnf | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | -0.108 | [-0.150, -0.069] | 0.000 |
| driver-top3 | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | +0.074 | [+0.035, +0.113] | 1.000 |
| driver-top3 | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | -0.353 | [-0.421, -0.285] | 0.000 |
| driver-top3 | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | val | -0.000 | [-0.025, +0.024] | 0.497 |
| driver-top3 | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | test | -0.344 | [-0.412, -0.273] | 0.000 |
| driver-top3 | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.109 | [+0.074, +0.147] | 1.000 |
| driver-top3 | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.075 | [+0.047, +0.104] | 1.000 |
| driver-top3 | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | val | +0.079 | [+0.052, +0.107] | 1.000 |
| driver-top3 | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | test | -0.012 | [-0.032, +0.007] | 0.102 |
| driver-top3 | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | val | +0.005 | [-0.004, +0.014] | 0.861 |
| driver-top3 | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | test | -0.003 | [-0.017, +0.011] | 0.339 |
| driver-top3 | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | +0.011 | [+0.001, +0.023] | 0.985 |
| driver-top3 | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | +0.302 | [+0.250, +0.357] | 1.000 |
| driver-top3 | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | -0.103 | [-0.155, -0.056] | 0.000 |
| driver-top3 | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | -0.145 | [-0.186, -0.104] | 0.000 |
| driver-position | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | +0.015 | [+0.001, +0.031] | 0.981 |
| driver-position | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | +0.016 | [-0.009, +0.040] | 0.893 |
| driver-position | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | val | +0.004 | [-0.010, +0.020] | 0.727 |
| driver-position | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | test | +0.051 | [+0.031, +0.070] | 1.000 |
| driver-position | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.066 | [+0.050, +0.084] | 1.000 |
| driver-position | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.008 | [-0.009, +0.024] | 0.815 |
| driver-position | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | val | +0.014 | [-0.002, +0.029] | 0.962 |
| driver-position | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | test | -0.020 | [-0.044, +0.001] | 0.034 |
| driver-position | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | val | +0.003 | [-0.005, +0.011] | 0.767 |
| driver-position | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | test | +0.014 | [-0.003, +0.032] | 0.951 |
| driver-position | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | -0.001 | [-0.014, +0.013] | 0.443 |
| driver-position | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | +0.037 | [+0.015, +0.061] | 0.999 |
| driver-position | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | -0.078 | [-0.103, -0.053] | 0.000 |
| driver-position | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | +0.025 | [+0.001, +0.050] | 0.982 |

Paired bootstrap over rows (2000 resamples), metrics averaged over seeds. AUROC difference for classification; NMAE difference (sign flipped) for regression.
