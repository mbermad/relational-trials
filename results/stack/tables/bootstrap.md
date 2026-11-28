| Task | Question | A | B | Split | A − B (better = +) | 95% CI | P(A better) |
|---|---|---|---|---|---|---|---|
| user-engagement | label history | GNN (GloVe text) + label history | GNN (GloVe text) | val | +0.001 | [-0.001, +0.003] | 0.906 |
| user-engagement | label history | GNN (GloVe text) + label history | GNN (GloVe text) | test | -0.009 | [-0.011, -0.007] | 0.000 |
| user-engagement | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | val | +0.000 | [-0.001, +0.001] | 0.636 |
| user-engagement | label history | LightGBM + aggregates | LightGBM + aggregates, no label history | test | +0.000 | [-0.001, +0.001] | 0.635 |
| user-engagement | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | val | +0.004 | [+0.001, +0.006] | 0.999 |
| user-engagement | LightGBM vs best GNN | LightGBM + aggregates | GNN (GloVe text) + label history | test | +0.012 | [+0.009, +0.014] | 1.000 |
| user-engagement | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | val | -0.247 | [-0.259, -0.235] | 0.000 |
| user-engagement | LightGBM (own row) vs GNN | LightGBM (trial table) | GNN (GloVe text) | test | -0.248 | [-0.260, -0.236] | 0.000 |

Paired bootstrap over rows (2000 resamples), metrics averaged over seeds. AUROC difference for classification; NMAE difference (sign flipped) for regression.
