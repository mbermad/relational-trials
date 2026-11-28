| Model | Outcome AUROC (val) | Outcome AUROC (test) | Adverse NMAE (val) | Adverse NMAE (test) | Adverse MAE (test) | Seeds |
|---|---|---|---|---|---|---|
| LightGBM (trial table) | 0.689 ± 0.013 | 0.718 ± 0.035 | 0.112 ± 0.001 | 0.106 ± 0.002 | 36.051 ± 0.752 | 3 |
| LightGBM + aggregates, no label history | 0.699 ± 0.012 | 0.716 ± 0.010 | 0.108 ± 0.001 | 0.105 ± 0.002 | 35.677 ± 0.774 | 3 |
| LightGBM + aggregates | 0.700 ± 0.007 | 0.723 ± 0.041 | 0.107 ± 0.002 | 0.103 ± 0.004 | 34.960 ± 1.475 | 3 |
| GNN, no text | 0.633 ± 0.001 | 0.637 ± 0.010 | 0.142 ± 0.002 | 0.139 ± 0.001 | 47.013 ± 0.423 | 3 |
| GNN, no text + label history | 0.675 ± 0.011 | 0.697 ± 0.012 | 0.142 ± 0.000 | 0.137 ± 0.002 | 46.359 ± 0.809 | 3 |
| GNN (GloVe text) | 0.639 ± 0.007 | 0.664 ± 0.036 | 0.136 ± 0.001 | 0.130 ± 0.002 | 44.179 ± 0.796 | 3 |
| GNN (GloVe text) + label history | 0.671 ± 0.004 | 0.701 ± 0.017 | 0.136 ± 0.002 | 0.131 ± 0.002 | 44.352 ± 0.766 | 3 |
| GNN (GloVe text), tuned | 0.642 ± 0.017 | 0.699 ± 0.032 | 0.137 ± 0.003 | 0.130 ± 0.002 | 44.134 ± 0.603 | 3 |
| GNN (GloVe text) + label history, tuned | 0.671 ± 0.031 | 0.695 ± 0.064 | 0.136 ± 0.001 | 0.131 ± 0.003 | 44.444 ± 0.863 | 3 |
| RelGNN (GloVe text) | 0.637 ± 0.015 | 0.659 ± 0.039 | 0.140 ± 0.006 | 0.135 ± 0.003 | 45.877 ± 1.173 | 3 |
| RelGNN (GloVe text) + label history | 0.663 ± 0.012 | 0.689 ± 0.060 | 0.139 ± 0.002 | 0.134 ± 0.002 | 45.554 ± 0.545 | 3 |
| Blend: LightGBM + aggregates & GNN (GloVe text) | 0.700 ± 0.008 | 0.725 ± 0.036 | **0.107 ± 0.002** | **0.103 ± 0.004** | **34.960 ± 1.475** | 3 |
| Blend: LightGBM + aggregates & GNN (GloVe text) + label history | **0.701 ± 0.007** | **0.726 ± 0.031** | 0.107 ± 0.002 | 0.103 ± 0.004 | 34.960 ± 1.475 | 3 |
| Blend: LightGBM + aggregates, no label history & GNN (GloVe text) | 0.700 ± 0.010 | 0.717 ± 0.008 | 0.108 ± 0.001 | 0.105 ± 0.002 | 35.677 ± 0.774 | 3 |
| *RelBench reference: LightGBM (trial table)* | -- | 0.689 | -- | 0.129 | -- | 1 |
| *RelBench reference: GNN (GloVe text)* | -- | 0.669 | -- | 0.129 | -- | 1 |

Mean ± 95% CI over seeds. AUROC: higher is better. NMAE = MAE / train std (RelBench v3's regression metric): lower is better.
