| Task | Question | A | B | Split | A − B (better = +) | 95% CI | P(A better) |
|---|---|---|---|---|---|---|---|
| study-outcome | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | +0.033 | [+0.009, +0.059] | 0.996 |
| study-outcome | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | +0.035 | [+0.007, +0.064] | 0.992 |
| study-outcome | label history | GNN (GloVe text) + label history, tuned | GNN (GloVe text), tuned | val | +0.029 | [+0.002, +0.057] | 0.982 |
| study-outcome | label history | GNN (GloVe text) + label history, tuned | GNN (GloVe text), tuned | test | -0.004 | [-0.033, +0.027] | 0.401 |
| study-outcome | label history | GNN, no text + label history | GNN, no text | val | +0.042 | [+0.014, +0.069] | 0.999 |
| study-outcome | label history | GNN, no text + label history | GNN, no text | test | +0.060 | [+0.031, +0.089] | 1.000 |
| study-outcome | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | val | +0.027 | [+0.001, +0.052] | 0.981 |
| study-outcome | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | test | +0.030 | [+0.002, +0.060] | 0.982 |
| study-outcome | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.002 | [-0.015, +0.017] | 0.581 |
| study-outcome | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.018 | [-0.000, +0.037] | 0.973 |
| study-outcome | text | GNN (GloVe text) | GNN, no text | val | +0.006 | [-0.020, +0.030] | 0.663 |
| study-outcome | text | GNN (GloVe text) | GNN, no text | test | +0.027 | [-0.002, +0.056] | 0.963 |
| study-outcome | tuning | GNN (GloVe text), tuned | GNN (GloVe text) | val | +0.003 | [-0.015, +0.020] | 0.662 |
| study-outcome | tuning | GNN (GloVe text), tuned | GNN (GloVe text) | test | +0.035 | [+0.016, +0.054] | 1.000 |
| study-outcome | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | val | -0.002 | [-0.024, +0.019] | 0.417 |
| study-outcome | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | test | -0.005 | [-0.028, +0.018] | 0.327 |
| study-outcome | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | val | -0.009 | [-0.024, +0.007] | 0.129 |
| study-outcome | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | test | -0.010 | [-0.027, +0.006] | 0.116 |
| study-outcome | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | +0.029 | [+0.008, +0.049] | 0.996 |
| study-outcome | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | +0.036 | [+0.012, +0.059] | 0.998 |
| study-outcome | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | +0.050 | [+0.023, +0.076] | 1.000 |
| study-outcome | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | +0.054 | [+0.024, +0.083] | 1.000 |
| study-adverse | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | -0.000 | [-0.001, +0.001] | 0.387 |
| study-adverse | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | -0.001 | [-0.002, +0.000] | 0.074 |
| study-adverse | label history | GNN (GloVe text) + label history, tuned | GNN (GloVe text), tuned | val | +0.000 | [-0.001, +0.001] | 0.561 |
| study-adverse | label history | GNN (GloVe text) + label history, tuned | GNN (GloVe text), tuned | test | -0.001 | [-0.002, +0.000] | 0.060 |
| study-adverse | label history | GNN, no text + label history | GNN, no text | val | -0.000 | [-0.001, +0.001] | 0.480 |
| study-adverse | label history | GNN, no text + label history | GNN, no text | test | +0.002 | [+0.001, +0.003] | 0.997 |
| study-adverse | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | val | +0.001 | [-0.001, +0.002] | 0.814 |
| study-adverse | label history | RelGNN (GloVe text) + label history | RelGNN (GloVe text) | test | +0.001 | [-0.000, +0.002] | 0.912 |
| study-adverse | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.002 | [-0.000, +0.004] | 0.965 |
| study-adverse | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.003 | [-0.000, +0.006] | 0.969 |
| study-adverse | text | GNN (GloVe text) | GNN, no text | val | +0.005 | [+0.003, +0.008] | 1.000 |
| study-adverse | text | GNN (GloVe text) | GNN, no text | test | +0.008 | [+0.005, +0.011] | 1.000 |
| study-adverse | tuning | GNN (GloVe text), tuned | GNN (GloVe text) | val | -0.000 | [-0.001, +0.001] | 0.376 |
| study-adverse | tuning | GNN (GloVe text), tuned | GNN (GloVe text) | test | +0.000 | [-0.001, +0.001] | 0.574 |
| study-adverse | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | val | -0.004 | [-0.006, -0.002] | 0.000 |
| study-adverse | RelGNN vs GraphSAGE | RelGNN (GloVe text) | GNN (GloVe text) | test | -0.005 | [-0.007, -0.003] | 0.000 |
| study-adverse | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | val | -0.003 | [-0.005, -0.001] | 0.001 |
| study-adverse | RelGNN vs GraphSAGE (with history) | RelGNN (GloVe text) + label history | GNN (GloVe text) + label history | test | -0.003 | [-0.006, -0.001] | 0.002 |
| study-adverse | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | +0.030 | [+0.020, +0.042] | 1.000 |
| study-adverse | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | +0.029 | [+0.017, +0.042] | 1.000 |
| study-adverse | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | +0.024 | [+0.015, +0.035] | 1.000 |
| study-adverse | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | +0.024 | [+0.013, +0.035] | 1.000 |

Paired bootstrap over rows (2000 resamples), metrics averaged over seeds. AUROC difference for classification; NMAE difference (sign flipped) for regression.
