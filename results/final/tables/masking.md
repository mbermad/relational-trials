| Task | Model | History masked | Metric | Val | Test |
|---|---|---|---|---|---|
| study-adverse | lgbm_agg | none | nmae | 0.107 | 0.103 |
| study-adverse | lgbm_agg | all | nmae | 0.115 (+0.009) | 0.111 (+0.008) |
| study-adverse | lgbm_agg | condition | nmae | 0.112 (+0.005) | 0.107 (+0.004) |
| study-adverse | lgbm_agg | facility | nmae | 0.107 (+0.001) | 0.104 (+0.000) |
| study-adverse | lgbm_agg | intervention | nmae | 0.108 (+0.001) | 0.105 (+0.002) |
| study-adverse | lgbm_agg | sponsor | nmae | 0.108 (+0.001) | 0.104 (+0.001) |
| study-adverse | rdl_sage_hist | none | nmae | 0.136 | 0.131 |
| study-adverse | rdl_sage_hist | all | nmae | 0.137 (+0.000) | 0.132 (+0.001) |
| study-adverse | rdl_sage_hist | condition | nmae | 0.136 (-0.000) | 0.131 (+0.000) |
| study-adverse | rdl_sage_hist | facility | nmae | 0.137 (+0.000) | 0.132 (+0.001) |
| study-adverse | rdl_sage_hist | intervention | nmae | 0.137 (+0.000) | 0.131 (+0.000) |
| study-adverse | rdl_sage_hist | sponsor | nmae | 0.136 (-0.000) | 0.131 (+0.000) |
| study-outcome | lgbm_agg | none | roc_auc | 0.700 | 0.723 |
| study-outcome | lgbm_agg | all | roc_auc | 0.668 (-0.032) | 0.700 (-0.024) |
| study-outcome | lgbm_agg | condition | roc_auc | 0.687 (-0.012) | 0.730 (+0.007) |
| study-outcome | lgbm_agg | facility | roc_auc | 0.693 (-0.007) | 0.712 (-0.011) |
| study-outcome | lgbm_agg | intervention | roc_auc | 0.699 (-0.000) | 0.718 (-0.005) |
| study-outcome | lgbm_agg | sponsor | roc_auc | 0.691 (-0.009) | 0.718 (-0.005) |
| study-outcome | rdl_sage_hist | none | roc_auc | 0.671 | 0.701 |
| study-outcome | rdl_sage_hist | all | roc_auc | 0.618 (-0.052) | 0.660 (-0.041) |
| study-outcome | rdl_sage_hist | condition | roc_auc | 0.649 (-0.022) | 0.694 (-0.007) |
| study-outcome | rdl_sage_hist | facility | roc_auc | 0.668 (-0.003) | 0.700 (-0.001) |
| study-outcome | rdl_sage_hist | intervention | roc_auc | 0.671 (+0.001) | 0.698 (-0.002) |
| study-outcome | rdl_sage_hist | sponsor | roc_auc | 0.656 (-0.014) | 0.688 (-0.013) |

History columns replaced by their training mean at inference; the model is not retrained. In brackets: change vs. no masking (mean over seeds).
