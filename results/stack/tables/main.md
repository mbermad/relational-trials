| Model | user-engagement AUROC (val) | user-engagement AUROC (test) | Seeds |
|---|---|---|---|
| LightGBM (trial table) | 0.652 ± 0.003 | 0.653 ± 0.001 | 3 |
| LightGBM + aggregates, no label history | 0.903 ± 0.000 | 0.904 ± 0.001 | 3 |
| LightGBM + aggregates | **0.904 ± 0.001** | **0.904 ± 0.001** | 3 |
| GNN (GloVe text) | 0.899 ± 0.002 | 0.901 ± 0.002 | 3 |
| GNN (GloVe text) + label history | 0.900 ± 0.001 | 0.892 ± 0.002 | 3 |

Mean ± 95% CI over seeds. AUROC: higher is better. NMAE = MAE / train std (RelBench v3's regression metric): lower is better.
