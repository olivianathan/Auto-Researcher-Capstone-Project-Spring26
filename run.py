"""
run.py  --  FROZEN
Do not modify this file. It loads the prepared data, calls build_model()
from model.py, evaluates results, and appends a row to experiment_log.csv.

Usage:
    python run.py

Fairness metric (iters 7+):
  - Sex:  equalized odds (TPR gap across Male/Female)
  - Race: equalized odds (TPR gap across race groups)
  - Age:  mean absolute calibration error (MACE) across age bins
  Overall fairness = max(sex_gap, race_gap, age_mace)
  Threshold search (fairness_tuned) minimizes sex+race equalized odds only,
  since age calibration error is threshold-independent (uses y_prob directly).
"""

import csv
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from model import build_model, DESCRIPTION, THRESHOLD_MODE

# -- 1. LOAD PREPARED DATA -----------------------------------------
print("Loading prepared data...")

X_train = pd.read_csv('X_train.csv', index_col=0)
X_val   = pd.read_csv('X_val.csv',   index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0).squeeze()
y_val   = pd.read_csv('y_val.csv',   index_col=0).squeeze()

with open('baseline_results.json') as f:
    baseline = json.load(f)

with open('locked_test_indices.json') as f:
    locked_idx = set(json.load(f))

assert len(set(X_train.index) & locked_idx) == 0, "LEAK: test indices found in train set!"
assert len(set(X_val.index)   & locked_idx) == 0, "LEAK: test indices found in val set!"
print("  Test set isolation confirmed -- no leakage.")
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

# -- 2. BUILD + FIT MODEL ------------------------------------------
print(f"\nBuilding model: {DESCRIPTION}")

model = build_model()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

start = time.time()
try:
    model.fit(X_train_s, y_train)
    X_eval = X_val_s
except Exception:
    model = build_model()
    model.fit(X_train, y_train)
    X_eval = X_val.values
elapsed = round(time.time() - start, 2)

y_prob = model.predict_proba(X_eval)[:, 1]

# -- 3. FAIRNESS HELPERS -------------------------------------------
subgroup_cols = X_val[['RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]

def _sex_race_eq_odds(y_true, y_pred_bin, subgroup_df):
    """Equalized odds (TPR gap) for sex and race axes."""
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred_bin)
    gaps = {}
    for axis_name, series in [('sex', subgroup_df['RIAGENDR']),
                               ('race', subgroup_df['RIDRETH3'])]:
        tprs = []
        for val in series.dropna().unique():
            mask = series.values == val
            yt, yp = y_true_np[mask], y_pred_np[mask]
            if len(yt) < 20 or np.unique(yt).size < 2:
                continue
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            _, _, fn_s, tp_s = cm.ravel()
            tpr = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else np.nan
            if not np.isnan(tpr):
                tprs.append(tpr)
        gaps[axis_name] = round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else np.nan
    return gaps

def _age_calibration_error(y_true, y_prob, age_series):
    """
    Mean absolute calibration error (MACE) across age bins.
    For each bin: |mean(predicted prob) - actual positive rate|.
    Threshold-independent -- uses raw probabilities.
    """
    age_bucket = pd.cut(age_series, bins=[0, 17, 34, 49, 64, 80],
                        labels=['0-17', '18-34', '35-49', '50-64', '65-80'])
    y_true_np = np.asarray(y_true)
    errors = []
    for grp in ['0-17', '18-34', '35-49', '50-64', '65-80']:
        mask = (age_bucket == grp).values
        if mask.sum() < 20:
            continue
        actual_rate    = float(y_true_np[mask].mean())
        predicted_rate = float(y_prob[mask].mean())
        errors.append(abs(actual_rate - predicted_rate))
    return round(float(np.mean(errors)), 4) if errors else np.nan

# -- 4. THRESHOLD --------------------------------------------------
# Threshold search only minimizes sex+race equalized odds.
# Age calibration error is threshold-independent and not included here.
if THRESHOLD_MODE == 'fairness_tuned':
    print("  Searching fairness-optimal threshold for sex/race (0.25-0.75)...")
    best_thr, best_loss = 0.5, np.inf
    for thr in np.linspace(0.25, 0.75, 51):
        y_tmp  = (y_prob >= thr).astype(int)
        sr     = _sex_race_eq_odds(y_val.values, y_tmp, subgroup_cols)
        sr_max = max(v for v in sr.values() if not np.isnan(v))
        loss   = sr_max + 0.02 * abs(thr - 0.5)
        if loss < best_loss:
            best_loss, best_thr = loss, float(thr)
    print(f"  Best threshold: {best_thr:.3f}")
else:
    best_thr = 0.5

y_pred = (y_prob >= best_thr).astype(int)

# -- 5. EVALUATE ---------------------------------------------------
auc         = round(roc_auc_score(y_val, y_prob), 4)
cm          = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
overall_tpr = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0

sr_gaps  = _sex_race_eq_odds(y_val.values, y_pred, subgroup_cols)
age_mace = _age_calibration_error(y_val.values, y_prob, subgroup_cols['RIDAGEYR'])

sex_gap  = sr_gaps.get('sex',  np.nan)
race_gap = sr_gaps.get('race', np.nan)
all_metrics     = [v for v in [sex_gap, race_gap, age_mace] if not np.isnan(v)]
overall_fairness = round(max(all_metrics), 4) if all_metrics else 1.0

objective    = round(auc - 0.15 * overall_fairness, 4)
baseline_obj = round(baseline['auc_roc'] - 0.15 * baseline['equalized_odds_gap'], 4)
improved     = "YES" if objective > baseline_obj else "NO"

# -- 6. PRINT RESULTS ---------------------------------------------
print(f"\n-- RESULTS -----------------------------------------")
print(f"  AUC-ROC:          {auc}  (baseline: {baseline['auc_roc']})")
print(f"  Overall TPR:      {overall_tpr}")
print(f"  Threshold:        {best_thr:.3f}")
print(f"  Overall fairness: {overall_fairness}  (max of sex/race EqOdds + age MACE)")
print(f"    -> sex  (EqOdds):  {sex_gap}")
print(f"    -> race (EqOdds):  {race_gap}")
print(f"    -> age  (MACE):    {age_mace}")
print(f"  Objective score:  {objective}  (baseline: {baseline_obj})")
print(f"  Runtime:          {elapsed}s")
print(f"  Improved over baseline: {improved}")

# -- 7. APPEND TO experiment_log.csv ------------------------------
log_path = 'experiment_log.csv'
log_columns = [
    'timestamp', 'iteration',
    'model', 'experiment_name', 'model_type',
    'changes_made', 'notes',
    'threshold',
    'auc_roc', 'overall_tpr',
    'eq_odds_gap', 'eq_odds_gap_overall',
    'eq_odds_gap_sex', 'eq_odds_gap_age', 'eq_odds_gap_race',
    'runtime_seconds', 'fairness_pass', 'objective_score',
    'discard_or_keep',
]

if os.path.exists(log_path):
    prev_log  = pd.read_csv(log_path)
    iteration = int(pd.to_numeric(prev_log['iteration'], errors='coerce').dropna().max()) + 1 \
        if 'iteration' in prev_log.columns and not prev_log.empty else 1

    discard_or_keep = 'keep'
    if 'discard_or_keep' in prev_log.columns and 'objective_score' in prev_log.columns:
        kept = prev_log[prev_log['discard_or_keep'] == 'keep']
        if not kept.empty:
            last_kept_obj = pd.to_numeric(kept['objective_score'], errors='coerce').dropna().iloc[-1]
            discard_or_keep = 'keep' if objective > last_kept_obj else 'discard'
else:
    iteration       = 1
    discard_or_keep = 'keep'

fairness_pass = 'PASS' if overall_fairness <= 0.05 else 'FAIL'

write_header = not os.path.exists(log_path)
with open(log_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=log_columns)
    if write_header:
        writer.writeheader()
    writer.writerow({
        'timestamp':           pd.Timestamp.now().isoformat(timespec='seconds'),
        'iteration':           iteration,
        'model':               DESCRIPTION,
        'experiment_name':     f'iter_{iteration}',
        'model_type':          DESCRIPTION.split(':')[0].strip() if ':' in DESCRIPTION else DESCRIPTION,
        'changes_made':        DESCRIPTION,
        'notes':               'age fairness = MACE (iters 7+); sex/race = equalized odds',
        'threshold':           round(best_thr, 3),
        'auc_roc':             auc,
        'overall_tpr':         overall_tpr,
        'eq_odds_gap':         overall_fairness,
        'eq_odds_gap_overall': overall_fairness,
        'eq_odds_gap_sex':     sex_gap,
        'eq_odds_gap_age':     age_mace,
        'eq_odds_gap_race':    race_gap,
        'runtime_seconds':     elapsed,
        'fairness_pass':       fairness_pass,
        'objective_score':     objective,
        'discard_or_keep':     discard_or_keep,
    })

print(f"\n  decision: {discard_or_keep.upper()}")
print(f"Logged -> {log_path} (iteration {iteration})")

# -- 8. SUMMARY TABLE ---------------------------------------------
try:
    summary = pd.read_csv(log_path)
    for col in ['iteration', 'auc_roc', 'eq_odds_gap_overall', 'objective_score']:
        summary[col] = pd.to_numeric(summary[col], errors='coerce')
    summary = summary.dropna(subset=['auc_roc', 'objective_score']) \
                     .sort_values('objective_score', ascending=False)

    cols = ['iteration', 'experiment_name', 'auc_roc',
            'eq_odds_gap_overall', 'threshold', 'objective_score', 'discard_or_keep']
    cols = [c for c in cols if c in summary.columns]

    print(f"\n-- ALL RESULTS (sorted by objective) ---------------")
    print(summary[cols].to_string(index=False))
except Exception:
    pass

# -- 9. REGENERATE PERFORMANCE PLOT -------------------------------
try:
    import matplotlib.pyplot as plt

    plot_df = pd.read_csv(log_path)
    for col in ['auc_roc', 'eq_odds_gap_overall', 'iteration']:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna(subset=['auc_roc', 'eq_odds_gap_overall']).sort_values('iteration')

    # mark where the fairness metric definition changed
    switch_iter = 7

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    pre  = plot_df[plot_df['iteration'] < switch_iter]
    post = plot_df[plot_df['iteration'] >= switch_iter]
    ax.scatter(pre['eq_odds_gap_overall'],  pre['auc_roc'],  alpha=0.8, zorder=3, label='iters 1-6 (EqOdds)')
    ax.scatter(post['eq_odds_gap_overall'], post['auc_roc'], alpha=0.8, zorder=3, marker='D', label='iters 7+ (hybrid)')
    for _, row in plot_df.iterrows():
        ax.annotate(int(row['iteration']),
                    (row['eq_odds_gap_overall'], row['auc_roc']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_title('AUC vs Fairness Metric')
    ax.set_xlabel('Fairness metric (EqOdds iters 1-6 / hybrid iters 7+)')
    ax.set_ylabel('AUC-ROC (higher = better)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(plot_df['iteration'], plot_df['auc_roc'], marker='o', label='AUC-ROC')
    ax2.plot(plot_df['iteration'], plot_df['eq_odds_gap_overall'], marker='o', label='Fairness metric')
    ax2.axvline(x=switch_iter - 0.5, color='gray', linestyle='--', linewidth=1, label='metric switch')
    ax2.set_title('Metric Trend by Iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance.png', dpi=150)
    plt.close()
    print("Updated -> performance.png")
except Exception as e:
    print(f"Could not update performance.png: {e}")
