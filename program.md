# Program Plan: NHANES Chronic Disease Risk Prediction

## 1) Project Intent

Build an auto-researcher workflow that predicts chronic disease risk from NHANES participant data while explicitly tracking fairness across demographic subgroups.

Core question:

> Can wearable/activity-linked and health survey features predict prediabetes or hypertension risk accurately **and** fairly across sex, age, and race/ethnicity groups?

---

## 2) Problem Definition

- **Dataset**: NHANES 2011-2014 (merged demographic, questionnaire, exam, labs, medication tables)
- **Modeling table**: `nhanes_clean.csv`
- **Target**: `high_risk` (binary)
  - 1 if HbA1c >= 5.7 (`prediabetes_risk`) OR avg blood pressure >= 130/80 (`hypertension_risk`)
  - 0 otherwise
- **Observed prevalence**: about 31.6% positive class

This framing supports early risk flagging, not diagnosis.

---

## 3) Fixed Rules (Do Not Change Mid-Loop)

To keep experiments comparable, freeze these items:

1. Target definition and clinical thresholds
2. Locked test set (`locked_test_indices.json`) used only for final evaluation
3. Primary metric and fairness metric definitions
4. Core feature table (`nhanes_clean.csv`) and baseline preprocessing assumptions

Any change to a frozen component starts a new experiment track and must be labeled as such.

---

## 4) Success Criteria

### Predictive Performance

- Beat logistic baseline by at least **+0.03 AUC-ROC**

### Fairness

- Equalized odds gap <= **0.05** across tracked subgroups
- No subgroup TPR more than **0.10** below overall TPR

### Practical Constraints

- Runtime per iteration should remain manageable for notebook workflows
- Experiment decisions should be traceable in `experiment_log.csv`

---

## 5) Evaluation Protocol

## Split Strategy

- Train/validation split is used for agent iteration
- Locked test set is untouched until model selection is complete

## Metrics Per Iteration

- `auc_roc`
- `overall_tpr`
- `eq_odds_gap`
- subgroup TPRs by:
  - sex (`RIAGENDR`)
  - race/ethnicity (`RIDRETH3`)
  - age bucket (`RIDAGEYR` binned to 18-34, 35-49, 50-64, 65-80)

## Logging

Every run logs:

- model name
- changes made
- metrics
- runtime
- fairness pass/fail
- notes/hypothesis

---

## 6) Auto-Researcher Loop Specification

For each iteration:

1. **Propose change** (single clear hypothesis)
2. **Train model** on train split
3. **Evaluate** on validation split
4. **Audit fairness** across required subgroups
5. **Log result** to `experiment_log.csv`
6. **Decide next step**:
   - keep/expand if AUC improves without fairness collapse
   - rollback or modify if fairness gap worsens

Stop when improvements plateau or constraints are violated.

---

## 7) Experiment Priority Queue

### Stage A: Strong Baselines

- Logistic regression (already established)
- XGBoost default
- XGBoost + class weighting
- XGBoost + SMOTE
- Tuned XGBoost (depth/trees/lr/subsample)

### Stage B: Fairness-Aware Refinement

- Threshold tuning by validation objective (AUC vs fairness tradeoff)
- Feature ablation on potentially disparity-amplifying predictors
- Reweighting or balanced training strategies per subgroup

### Stage C: Robustness Checks

- Stability across random seeds
- Calibration checks by subgroup
- Error analysis of false negatives in high-risk populations

---

## 8) Risks and Mitigations

- **Risk: fairness gap remains high despite better AUC**
  - Mitigation: prioritize thresholding, subgroup-aware reweighting, and ablation studies
- **Risk: overfitting from iterative tuning**
  - Mitigation: strict locked test policy and concise hypothesis-driven changes
- **Risk: noisy subgroup estimates for small groups**
  - Mitigation: minimum support checks and transparent reporting of excluded cells

---

## 9) Deliverables

1. Reproducible data pipeline outputs:
   - `nhanes_merged.csv`
   - `nhanes_selected.csv`
   - `nhanes_clean.csv`
2. Baseline metrics (`baseline_results.json`)
3. Agent experiment trace (`experiment_log.csv`)
4. Final selected model and locked-test evaluation
5. Capstone report with:
   - modeling performance
   - subgroup fairness outcomes
   - limitations and ethical considerations

---

## 10) Minimum Viable Completion

Project is considered complete when all are true:

- baseline + at least 3 agent iterations executed
- best candidate selected using validation metrics
- locked test evaluation completed once
- fairness table reported for required demographic groups
- final conclusions include both performance and fairness tradeoffs
