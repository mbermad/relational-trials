| Model | DNF AUROC (val) | DNF AUROC (test) | Top-3 AUROC (val) | Top-3 AUROC (test) | Position NMAE (val) | Position NMAE (test) | Position MAE (test) | Seeds |
|---|---|---|---|---|---|---|---|---|
| LightGBM (trial table) | 0.620 ± 0.030 | 0.626 ± 0.066 | 0.661 ± 0.020 | 0.635 ± 0.015 | 0.523 ± 0.006 | 0.590 ± 0.007 | 4.146 ± 0.047 | 3 |
| LightGBM + aggregates, no label history | 0.633 ± 0.061 | 0.664 ± 0.041 | 0.740 ± 0.049 | 0.653 ± 0.114 | 0.497 ± 0.016 | 0.570 ± 0.010 | 4.005 ± 0.073 | 3 |
| LightGBM + aggregates | 0.684 ± 0.051 | 0.671 ± 0.038 | **0.849 ± 0.010** | 0.728 ± 0.023 | 0.430 ± 0.009 | **0.562 ± 0.035** | **3.950 ± 0.247** | 3 |
| GNN (GloVe text) | 0.763 ± 0.024 | **0.735 ± 0.066** | 0.764 ± 0.045 | **0.779 ± 0.033** | 0.445 ± 0.012 | 0.615 ± 0.030 | 4.322 ± 0.210 | 3 |
| GNN (GloVe text) + label history | 0.712 ± 0.020 | 0.682 ± 0.053 | 0.838 ± 0.006 | 0.426 ± 0.234 | 0.430 ± 0.006 | 0.599 ± 0.093 | 4.211 ± 0.654 | 3 |
| RelGNN (GloVe text) | **0.798 ± 0.017** | 0.694 ± 0.016 | 0.843 ± 0.044 | 0.767 ± 0.046 | 0.431 ± 0.017 | 0.635 ± 0.090 | 4.465 ± 0.632 | 3 |
| RelGNN (GloVe text) + label history | 0.716 ± 0.002 | 0.693 ± 0.042 | 0.843 ± 0.010 | 0.424 ± 0.194 | **0.427 ± 0.017** | 0.585 ± 0.024 | 4.109 ± 0.165 | 3 |
| *RelBench reference: LightGBM (trial table)* | -- | 0.726 | -- | 0.649 | -- | 0.602 | -- | 1 |
| *RelBench reference: GNN (GloVe text)* | -- | 0.726 | -- | 0.680 | -- | 0.578 | -- | 1 |

Mean ± 95% CI over seeds. AUROC: higher is better. NMAE = MAE / train std (RelBench v3's regression metric): lower is better.
