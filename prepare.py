"""
prepare.py  —  FROZEN
Do not modify this file. It defines the data pipeline, train/val/test split,
and baseline evaluation. The agent should only modify model.py.

Run once before starting the agent loop:
    python prepare.py
"""

import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ── 1. MERGE RAW FILES ────────────────────────────────────────────
print("Loading and merging raw NHANES files...")

demographic   = pd.read_csv('demographic.csv')
diet          = pd.read_csv('diet.csv')
examination   = pd.read_csv('examination.csv')
labs          = pd.read_csv('labs.csv')
medications   = pd.read_csv('medications.csv', encoding='latin1')
questionnaire = pd.read_csv('questionnaire.csv', encoding='latin1')

med_agg = medications.groupby('SEQN').agg(
    num_medications   = ('RXDDRUG', 'count'),
    on_any_medication = ('RXDUSE', 'first')
).reset_index()

merged = demographic.copy()
for df in [diet, examination, labs, med_agg, questionnaire]:
    merged = merged.merge(df, on='SEQN', how='left', suffixes=('', '_dup'))

merged.to_csv('nhanes_merged.csv', index=False)
print(f"  Merged shape: {merged.shape}")

# ── 2. CLEAN ──────────────────────────────────────────────────────
print("Cleaning...")

df = pd.read_csv('nhanes_merged.csv')

activity_minutes = ['PAD615', 'PAD630', 'PAD645', 'PAD660', 'PAD675']
activity_days    = ['PAQ610', 'PAQ625', 'PAQ640', 'PAQ655', 'PAQ670']
df[activity_minutes + activity_days] = df[activity_minutes + activity_days].fillna(0)

df['total_active_min_per_week'] = (
    (df['PAD615'] * df['PAQ610']) +
    (df['PAD630'] * df['PAQ625']) +
    (df['PAD645'] * df['PAQ640']) +
    (df['PAD660'] * df['PAQ655']) +
    (df['PAD675'] * df['PAQ670'])
)

df['SMD030'] = df['SMD030'].fillna(0)
df['ALQ110'] = df['ALQ110'].fillna(0)
df['ALQ101'] = df['ALQ101'].fillna(0)

median_cols = [
    'BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST',
    'LBXTC', 'LBDLDL', 'LBDHDD', 'LBXTR',
    'LBXSGL', 'LBXIN', 'LBXGH',
    'avg_systolic', 'avg_diastolic',
    'PAD680', 'INDFMPIR', 'DBD900', 'DBD910',
]
for col in median_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

cat_cols = ['DMDEDUC2', 'SMQ020', 'DIQ010', 'BPQ020', 'on_any_medication']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df['num_medications'] = df['num_medications'].fillna(0)

# ── 3. SELECT COLUMNS + ENGINEER TARGETS ─────────────────────────
print("Selecting features and engineering targets...")

demographics   = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'DMDEDUC2']
activity       = ['PAQ605', 'PAQ610', 'PAD615', 'PAQ620', 'PAQ625', 'PAD630',
                  'PAQ635', 'PAQ640', 'PAD645', 'PAQ650', 'PAQ655', 'PAD660',
                  'PAQ665', 'PAQ670', 'PAD675', 'PAD680']
body_measures  = ['BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST']
lab_results    = ['LBXGH', 'LBXTC', 'LBDLDL', 'LBDHDD', 'LBXTR', 'LBXSGL', 'LBXIN']
blood_pressure = ['BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2']
lifestyle      = ['SMQ020', 'SMD030', 'ALQ101', 'ALQ110', 'DBD900', 'DBD910']
medications_   = ['num_medications', 'on_any_medication']
self_reported  = ['DIQ010', 'BPQ020']

all_cols = (demographics + activity + body_measures + lab_results +
            blood_pressure + lifestyle + medications_ + self_reported)
df_selected = df[[c for c in all_cols if c in df.columns]].copy()

df_selected['avg_systolic']  = df_selected[['BPXSY1', 'BPXSY2']].mean(axis=1)
df_selected['avg_diastolic'] = df_selected[['BPXDI1', 'BPXDI2']].mean(axis=1)

df_selected['prediabetes_risk'] = (df_selected['LBXGH'] >= 5.7).astype(int)
df_selected['hypertension_risk'] = (
    (df_selected['avg_systolic'] >= 130) |
    (df_selected['avg_diastolic'] >= 80)
).astype(int)
df_selected['high_risk'] = (
    (df_selected['prediabetes_risk'] == 1) |
    (df_selected['hypertension_risk'] == 1)
).astype(int)

df_selected.to_csv('nhanes_selected.csv', index=False)
print(f"  Selected shape: {df_selected.shape}")
print(f"  High risk prevalence: {df_selected['high_risk'].mean():.1%}")

# ── 4. FEATURE MATRIX ────────────────────────────────────────────
drop_cols = [
    'SEQN', 'high_risk', 'prediabetes_risk', 'hypertension_risk',
    'DIQ010', 'BPQ020',
    'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2',
    'LBXGH', 'avg_systolic', 'avg_diastolic',
]
features = [c for c in df_selected.columns if c not in drop_cols]
X = df_selected[features].copy()
y = df_selected['high_risk'].copy()

activity_flags = ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']
for col in activity_flags:
    if col in X.columns:
        X[col] = X[col].fillna(0)

# ── 5. LOCKED TRAIN / VAL / TEST SPLIT ───────────────────────────
# FROZEN — these splits never change across any experiment
print("Creating locked train/val/test split...")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)

test_indices = X_test.index.tolist()
with open('locked_test_indices.json', 'w') as f:
    json.dump(test_indices, f)

# save split data for run.py to load
X_train.to_csv('X_train.csv', index=True)
X_val.to_csv('X_val.csv', index=True)
y_train.to_csv('y_train.csv', index=True)
y_val.to_csv('y_val.csv', index=True)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test (locked): {len(X_test):,}")
print("  Splits saved: X_train.csv, X_val.csv, y_train.csv, y_val.csv")
print("  Test set locked → locked_test_indices.json")

# ── 6. BASELINE MODEL ────────────────────────────────────────────
print("\nFitting baseline logistic regression...")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

start = time.time()
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_s, y_train)
elapsed = round(time.time() - start, 2)

y_pred      = baseline_model.predict(X_val_s)
y_pred_prob = baseline_model.predict_proba(X_val_s)[:, 1]

auc = roc_auc_score(y_val, y_pred_prob)
cm  = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
overall_tpr = tp / (tp + fn)

print(f"\n── BASELINE RESULTS (validation set) ──────────────")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Overall TPR: {overall_tpr:.4f}")
print(f"  Runtime:     {elapsed}s")
print(f"\n{classification_report(y_val, y_pred)}")

# subgroup fairness
def subgroup_tpr(df, group_col, group_labels):
    results = []
    for code, label in group_labels.items():
        sub = df[df[group_col] == code]
        if len(sub) < 20:
            continue
        sub_cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
        if sub_cm.shape == (2, 2):
            _, _, fn_s, tp_s = sub_cm.ravel()
            tpr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
        else:
            tpr = 0
        results.append({'group': label, 'n': len(sub), 'tpr': round(tpr, 4)})
    return pd.DataFrame(results)

val_df = X_val.copy()
val_df['y_true'] = y_val.values
val_df['y_pred'] = y_pred
val_df['age_group'] = pd.cut(val_df['RIDAGEYR'],
    bins=[0, 17, 34, 49, 64, 80],
    labels=['0-17', '18-34', '35-49', '50-64', '65-80'])

sex_table  = subgroup_tpr(val_df, 'RIAGENDR', {1: 'Male', 2: 'Female'})
race_table = subgroup_tpr(val_df, 'RIDRETH3', {
    1: 'Mexican American', 2: 'Other Hispanic',
    3: 'NH White', 4: 'NH Black', 6: 'NH Asian'})

age_results = []
for grp in ['0-17', '18-34', '35-49', '50-64', '65-80']:
    sub = val_df[val_df['age_group'] == grp]
    if len(sub) < 20 or sub['y_true'].nunique() < 2:
        continue
    sub_cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
    if sub_cm.shape == (2, 2):
        _, _, fn_s, tp_s = sub_cm.ravel()
        tpr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    else:
        tpr = 0
    age_results.append({'group': grp, 'n': len(sub), 'tpr': round(tpr, 4)})
age_table = pd.DataFrame(age_results)

all_tprs = list(sex_table['tpr']) + list(race_table['tpr']) + list(age_table['tpr'])
eq_odds_gap = round(max(all_tprs) - min(all_tprs), 4)

print("── SUBGROUP FAIRNESS (Sex) ─────────────────────────")
print(sex_table.to_string(index=False))
print("\n── SUBGROUP FAIRNESS (Race/Ethnicity) ──────────────")
print(race_table.to_string(index=False))
print("\n── SUBGROUP FAIRNESS (Age Group) ───────────────────")
print(age_table.to_string(index=False))
print(f"\n── EQUALIZED ODDS GAP: {eq_odds_gap}")
print(f"   Target: ≤ 0.05 | {'PASS ✓' if eq_odds_gap <= 0.05 else 'FAIL ✗ — agent needs to close this gap'}")

# save baseline for run.py to compare against
baseline_results = {
    "model": "logistic_regression",
    "auc_roc": round(auc, 4),
    "overall_tpr": round(overall_tpr, 4),
    "equalized_odds_gap": eq_odds_gap,
    "runtime_seconds": elapsed,
    "n_features": len(features),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "features": features,
}
with open('baseline_results.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)

print("\nBaseline saved → baseline_results.json")

# ── 7. GENERATE PERFORMANCE PLOT ──────────────────────────────────
def plot_performance():
    import os
    if not os.path.exists('results.tsv'):
        print("No results.tsv yet — run run.py first to generate experiments.")
        return

    results = pd.read_csv('results.tsv', sep='\t')
    if results.empty:
        return

    for col in ['auc_roc', 'eq_odds_gap']:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors='coerce')

    results = results.dropna(subset=['auc_roc', 'eq_odds_gap'])
    if results.empty:
        return

    results['iteration'] = range(1, len(results) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # AUC vs fairness gap scatter
    ax = axes[0]
    ax.scatter(results['eq_odds_gap'], results['auc_roc'], alpha=0.8, zorder=3)
    for _, row in results.iterrows():
        ax.annotate(int(row['iteration']),
                    (row['eq_odds_gap'], row['auc_roc']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_title('AUC vs Equalized Odds Gap')
    ax.set_xlabel('Equalized Odds Gap (lower = fairer)')
    ax.set_ylabel('AUC-ROC (higher = better)')
    ax.grid(alpha=0.3)

    # metric trend
    ax2 = axes[1]
    ax2.plot(results['iteration'], results['auc_roc'], marker='o', label='AUC-ROC')
    ax2.plot(results['iteration'], results['eq_odds_gap'], marker='o', label='EqOdds Gap')
    ax2.set_title('Metric Trend by Iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance.png', dpi=150)
    plt.show()
    print("Saved → performance.png")

plot_performance()
print("\nprepare.py complete. Now run: python run.py")
