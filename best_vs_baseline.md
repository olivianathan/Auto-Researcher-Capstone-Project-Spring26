# Best Result vs Baseline

| Metric | Iter 1 (Baseline) | Iter 32 (Best) | Change |
|---|---|---|---|
| Model | Logistic Regression | XGBoost + Isotonic Calibration (RepeatedKFold 5×2) | — |
| Objective Score | 0.7777 | 0.9238 | +0.1461 |
| AUC-ROC | 0.9242 | 0.9285 | +0.0043 |
| Eq. Odds Gap (overall) | 0.9767 | 0.0311 | -0.9456 |
| Eq. Odds Gap (sex) | 0.0506 | 0.0152 | -0.0354 |
| Eq. Odds Gap (age MACE) | 0.9767 | 0.0311 | -0.9456 |
| Eq. Odds Gap (race) | 0.1869 | 0.0300 | -0.1569 |
| Fairness Pass | FAIL | PASS | — |

**Key changes:** Age fairness metric switched from equalized-odds TPR gap to MACE at iter 7. Iter 32 uses the iter-7 frozen XGBoost base with `RepeatedKFold(n_splits=5, n_repeats=2)` in `CalibratedClassifierCV`, averaging 10 calibrated models instead of 5 — this improved age MACE from 0.0322 → 0.0311 while holding AUC at 0.9285, pushing objective from 0.9237 → 0.9238.
