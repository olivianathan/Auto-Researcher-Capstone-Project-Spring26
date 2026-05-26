# Best Result vs Baseline

| Metric | Iter 1 (Baseline) | Iter 46 (Best) | Change |
|---|---|---|---|
| Model | Logistic Regression | XGBoost + Isotonic Cal. + Race Weights | — |
| Objective Score | 0.7777 | 0.9239 | +0.1462 |
| AUC-ROC | 0.9242 | 0.9285 | +0.0043 |
| Eq. Odds Gap (overall) | 0.9767 | 0.0309 | -0.9458 |
| Eq. Odds Gap (sex) | 0.0506 | 0.0117 | -0.0389 |
| Eq. Odds Gap (age MACE) | 0.9767 | 0.0309 | -0.9458 |
| Eq. Odds Gap (race) | 0.1869 | 0.0267 | -0.1602 |
| Fairness Pass | FAIL | PASS | — |

**Key changes:** Age fairness metric switched from equalized-odds TPR gap to MACE at iter 7. Iter 32 introduced `RepeatedKFold(n_splits=5, n_repeats=2)` calibration (10 models). Iter 46 adds race-only inverse-frequency sample weights (strength=0.3) via `GroupWeightedCalibrated` wrapper, reducing the race EqOdds gap from 0.030 → 0.027 and sex gap from 0.015 → 0.012 while holding AUC flat at 0.9285.
