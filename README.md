# Can Your Step Count Predict Your Health?
### Using Fitness Data to Predict Chronic Disease Risk Across Demographic Groups

**Data Science Capstone — Spring 2026**

---

## What This Project Is

This project builds an AI-driven modeling pipeline that predicts whether someone is at risk for **prediabetes or high blood pressure** using physical activity and health data from a large national survey (NHANES). 

What makes it different from a standard prediction task is the **fairness angle** — the goal isn't just to build the most accurate model, but to make sure the model performs equally well across different groups of people (by sex, age, and race/ethnicity). A model that works well on average but fails for a specific group isn't a success here.

The modeling pipeline follows **Andrej Karpathy's auto-researcher framework**, where an AI agent automatically proposes experiments, runs them, evaluates the results, and decides what to try next — rather than manually testing one model at a time.

---

## Research Question

> Can wearable activity data be used to accurately predict chronic disease risk while maintaining fair performance across demographic subgroups?

---

## Data

- **Source:** [NHANES 2011–2014](https://wwwn.cdc.gov/nchs/nhanes/) (National Health and Nutrition Examination Survey, CDC), accessed via [Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey)
- **Size:** ~10,175 participants, 53 features after cleaning
- **Files used:** demographic, diet, examination, labs, medications, questionnaire (merged on participant ID)
- **Target variable:** `high_risk` — 1 if HbA1c ≥ 5.7% (prediabetes) OR average BP ≥ 130/80 mmHg (hypertension). 31.6% of participants are flagged as high risk.

---

## What Success Looks Like

The project is successful if the final model:

1. **Beats the baseline** — AUC-ROC improves by at least +0.03 over logistic regression
2. **Is fair** — equalized odds gap between subgroups stays ≤ 0.05
3. **Doesn't leave anyone behind** — no single subgroup has a true positive rate more than 10 points below the overall rate

---

## Project Structure

```
├── merge_nhanes.py        # merges the 6 raw NHANES CSV files into one dataset
├── select_columns.py      # selects relevant columns + engineers target variables
├── clean_data.py          # handles missing values, creates total_active_min_per_week
│
├── 01_eda.ipynb           # exploratory data analysis — distributions, subgroup charts
├── 02_baseline.ipynb      # logistic regression baseline + subgroup performance table
├── 03_agent_loop.ipynb    # auto-researcher agent loop (XGBoost, neural net, fairness)
├── 04_evaluation.ipynb    # final model evaluation — AUC-ROC, equalized odds, results
│
├── nhanes_merged.csv      # all 6 files joined on SEQN
├── nhanes_selected.csv    # trimmed to 52 relevant columns + target variables
├── nhanes_clean.csv       # final cleaned dataset ready for modeling
│
├── requirements.txt       # python dependencies
└── README.md              # you are here
```

---

## How to Run It

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare the data** (run in order)
```bash
python merge_nhanes.py
python select_columns.py
python clean_data.py
```

**3. Open the notebooks in order**
```
01_eda.ipynb → 02_baseline.ipynb → 03_agent_loop.ipynb → 04_evaluation.ipynb
```

> Note: The raw NHANES CSV files are not included in this repo due to size. Download them from [Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey) and place them in the root folder before running the scripts.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
jupyter
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Target variable | `high_risk` (binary) | Combines prediabetes + hypertension into one actionable label |
| Primary metric | AUC-ROC | Handles class imbalance better than accuracy |
| Fairness metric | Equalized odds gap | Measures whether TPR is consistent across groups |
| Frozen components | Train/test split, target definition, clinical thresholds | Must stay fixed so agent iterations are comparable |
| Minimum viable project | Baseline + XGBoost + fairness table | Delivers a real result even if advanced models don't improve things |

---

*Capstone project — Spring 2026*
