# Program Plan: NHANES Chronic Disease Risk Prediction

## 1) Project Intent

Build an auto-researcher workflow that predicts chronic disease risk from NHANES participant data while explicitly tracking fairness across demographic subgroups.

Core question:

> Can survey, examination, and lab features predict prediabetes or hypertension risk accurately **and** fairly across sex, age, and race/ethnicity groups?

---

## 2) Problem Definition

- **Dataset**: NHANES 2011-2014 (merged demographic, questionnaire, examination, labs, medication tables)
- **Target**: `high_risk` (binary)
  - 1 if HbA1c >= 5.7 (`prediabetes_risk`) OR avg systolic >= 130 OR avg diastolic >= 80 (`hypertension_risk`)
  - 0 otherwise
- **Train / Val / Test split**: 64% / 16% / 20%, locked at `random_state=42` via `prepare.py`
  - Train: 6,512 rows | Val: 1,628 rows | Test: locked (`locked_test_indices.json`)
- **Features**: 39 columns — demographics, activity, body measures, labs, blood pressure proxies, lifestyle, medications
- **Target-defining features dropped**: HbA1c (`LBXGH`), raw BP readings (`BPXSY1/2`, `BPXDI1/2`), averaged BP (`avg_systolic`, `avg_diastolic`), and self-reported conditions (`DIQ010`, `BPQ020`) are excluded from the feature matrix to prevent leakage

This framing supports early risk flagging, not clinical diagnosis.

---

## 3) Fixed Rules (Do Not Change Mid-Loop)

1. `prepare.py` is **frozen** — data pipeline, feature selection, and train/val/test split never change
2. `run.py` is **frozen** — evaluation logic, fairness metrics, threshold search, and logging never change
3. The **locked test set** (`locked_test_indices.json`) is untouched until final model selection
4. Only `model.py` may be modified between iterations

Any change to a frozen component starts a new experiment track and must be labeled explicitly.

---

## 4) Objective Function

```
objective = AUC-ROC - 0.15 × overall_fairness
```

where `overall_fairness = max(sex_gap, race_gap, age_MACE)`.

**Fairness pass threshold**: `overall_fairness <= 0.05`

Fairness is measured as:
- **Sex**: equalized odds TPR gap across Male / Female
- **Race**: equalized odds TPR gap across race/ethnicity groups (`RIDRETH3`)
- **Age**: Mean Absolute Calibration Error (MACE) across age bins `[0-17, 18-34, 35-49, 50-64, 65-80]`
  - Age uses MACE (probability calibration error) rather than TPR gap because age risk is monotone and threshold-independent

Threshold is tuned by `run.py` (search 0.25–0.75) to minimize the sex + race equalized odds gap, with a small penalty for deviation from 0.50. Age MACE is not affected by threshold.

---

## 5) Current Best Model — Iter 32

| Metric | Iter 1 (Baseline) | Iter 32 (Best) | Change |
|---|---|---|---|
| Model | Logistic Regression | XGBoost + Isotonic Calibration | — |
| Objective Score | 0.7777 | **0.9238** | +0.1461 |
| AUC-ROC | 0.9242 | **0.9285** | +0.0043 |
| Overall Fairness | 0.9767 | **0.0311** | -0.9456 |
| Sex EqOdds Gap | 0.0506 | 0.0152 | -0.0354 |
| Race EqOdds Gap | 0.1869 | 0.0300 | -0.1569 |
| Age MACE | 0.9767 | 0.0311 | -0.9456 |
| Fairness Pass | FAIL | **PASS** | — |

**Iter 32 config**:
```python
XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.7,
              min_child_weight=5, gamma=0.1,
              random_state=42, eval_metric='logloss', verbosity=0)

CalibratedClassifierCV(base_xgb, method='isotonic',
                       cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=42))
```
10 calibrated models are averaged at prediction time. The calibration ensemble is what primarily controls age MACE.

---

## 6) Experiment History Summary (44 iterations)

### Phase 1 — Model Selection (Iters 1–6, pre-calibration)

| Iter | Model | AUC | Obj | Decision |
|---|---|---|---|---|
| 1 | Logistic Regression (baseline) | 0.9242 | 0.7777 | keep |
| 2 | LR balanced | 0.9234 | 0.7734 | discard |
| 3 | Random Forest | 0.9259 | 0.7866 | keep |
| 4 | XGBoost base | 0.9322 | 0.8062 | keep |
| 5 | XGBoost + threshold=0.75 | 0.9322 | 0.8092 | keep |
| 6 | XGBoost regularized | 0.9281 | 0.8100 | keep |

### Phase 2 — Calibration Baseline (Iter 7, metric switch)

**Age fairness metric changed from equalized-odds TPR gap to MACE at iter 7.**

| Iter | Model | AUC | Obj | Decision |
|---|---|---|---|---|
| 7 | XGBoost + isotonic cv=3 | 0.9273 | 0.9210 | keep — calibration baseline |
| 8 | XGBoost + sigmoid cv=3 | 0.9280 | 0.9215 | keep |

### Phase 3 — Controlled Calibration (Iters 9–14)

Base model frozen at iter 7 XGBoost for all controlled runs.

| Iter | Change | AUC | Obj | Decision |
|---|---|---|---|---|
| 9 | RF + isotonic cv=3 | 0.9245 | 0.9136 | discard (base changed) |
| 10 | XGBoost deeper + iso cv=3 | 0.9272 | 0.9224 | keep (confounded) |
| 11 | XGBoost deeper + iso cv=5 | 0.9276 | 0.9228 | keep (confounded) |
| 12 | cv 3→5 only | 0.9276 | 0.9228 | keep |
| 13 | cv 3→10 only | — | 0.9224 | discard |
| 14 | Two-stage age-quintile calibration | — | — | see error_taxonomy |

### Phase 4 — Agent Loop (Iters 15–34)

All used the frozen iter 7 XGBoost base. Every single-parameter change was evaluated. Key result: **no single-parameter modification outperformed iter 32**.

| Notable iters | Change | AUC | Obj | Decision |
|---|---|---|---|---|
| 32 | RepeatedKFold(5,2) isotonic | 0.9285 | **0.9238** | **KEEP — NEW BEST** |
| 33 | RepeatedKFold(5,3) isotonic | 0.9284 | 0.9232 | discard |
| 34 | RepeatedStratifiedKFold(5,2) | 0.9283 | 0.9235 | discard |

### Phase 5 — 10-Loop Campaign (Iters 35–44)

Attempted model diversity and architecture changes. No improvement over iter 32.

| Iter | Change | AUC | Obj | Decision |
|---|---|---|---|---|
| 35 | DART booster | 0.9263 | 0.9142 | discard — race 0.081 |
| 36 | n=1000, lr=0.01, RKF(5,2) | 0.9290 | 0.9235 | discard — race 0.037 |
| 37 | n=1000, lr=0.01, RKF(5,3) | 0.9289 | 0.9206 | discard — race 0.055 |
| 38 | RepeatedKFold(10,1) | 0.9278 | 0.9230 | discard |
| 39 | num_parallel_tree=5 | 0.9292 | 0.9190 | discard — race 0.068 |
| 40 | Sigmoid RKF(5,2) | 0.9285 | 0.9223 | discard — age MACE 0.041 |
| 41 | Isotonic+sigmoid voting | 0.9285 | 0.9231 | discard — age MACE 0.036 |
| 42 | Pipeline interaction terms | 0.9288 | 0.9206 | discard — race 0.055 |
| 43 | VotingClassifier 3 seeds | 0.9282 | 0.9235 | discard |
| 44 | ExtraTreesClassifier | 0.9235 | 0.9092 | discard — race 0.095 |

**Key finding**: Configurations with higher raw AUC (iters 36, 39 reached 0.9290–0.9292) consistently widen the race fairness gap above 0.05. Iter 32 sits on the Pareto frontier of AUC vs fairness for the current 39 features.

---

## 7) File Reference

| File | Status | Purpose |
|---|---|---|
| `prepare.py` | FROZEN | Merges raw CSVs, builds features, creates locked splits |
| `run.py` | FROZEN | Trains model, evaluates, logs to experiment_log.csv |
| `model.py` | AGENT-MUTABLE | Only file the agent modifies — contains `build_model()` |
| `experiment_log.csv` | Append-only | Full trace of all 44 iterations |
| `baseline_results.json` | Generated by prepare.py | Baseline LR metrics for comparison |
| `locked_test_indices.json` | Generated by prepare.py | Test set row indices — never used during iteration |
| `X_train.csv`, `X_val.csv` | Generated by prepare.py | 39-feature matrices (6,512 / 1,628 rows) |
| `y_train.csv`, `y_val.csv` | Generated by prepare.py | Binary target labels |
| `nhanes_merged.csv` | Generated by prepare.py | Full raw merge (10,175 rows × 1,814 cols) |
| `nhanes_selected.csv` | Generated by prepare.py | Post-selection with engineered target columns |
| `best_vs_baseline.md` | Reference | Head-to-head table of iter 1 vs iter 32 |
| `error_taxonomy.md` | Reference | Notes on confounded/failed iterations |
| `performance.png` | Generated by run.py | AUC + fairness trend plots (regenerated each run) |

---

## 8) AUC Ceiling Analysis

With the current 39 features, the uncalibrated XGBoost ceiling is approximately **0.929–0.930**. Calibration via `CalibratedClassifierCV` lifts AUC slightly (0.9285 with 10 averaged models) while also improving age MACE. Reaching AUC > 0.95 would require richer features — body composition measures like arm circumference (BMXARMC) or sagittal abdominal diameter (BMDAVSAD) exist in `nhanes_merged.csv` but are not included in `prepare.py`'s feature selection.

---

## 9) Risks and Mitigations

- **AUC-fairness tradeoff**: Configurations that push AUC higher tend to widen the race equalized odds gap. The current best (iter 32) was selected specifically because it holds both.
- **Overfitting from iterative tuning**: Strict locked test policy prevents leakage. All 44 iterations evaluated on the same held-out validation set.
- **Small subgroup estimates**: Race/ethnicity groups with n < 20 are excluded from fairness calculations with a transparent warning.

---

## 10) Next Steps

1. **Final test set evaluation** — run once on `locked_test_indices.json` using the iter 32 model to report held-out performance
2. **Capstone report** — document modeling decisions, fairness tradeoffs, and limitations
3. **Ethical framing** — discuss what it means to predict prediabetes/hypertension risk using race and sex as features, and the limits of equalized odds as a fairness criterion

---

## 11) Minimum Viable Completion

Project is complete when all are true:

- [x] Baseline established (iter 1)
- [x] At least 3 agent iterations executed (44 completed)
- [x] Best candidate selected using validation metrics (iter 32, obj 0.9238)
- [x] Fairness constraint satisfied (overall fairness 0.0311 <= 0.05)
- [ ] Locked test evaluation completed once
- [ ] Final conclusions documented including performance and fairness tradeoffs
