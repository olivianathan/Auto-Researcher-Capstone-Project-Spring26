import pandas as pd
import numpy as np
import json, time, csv, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── SETUP ─────────────────────────────────────────────────────────

df = pd.read_csv('nhanes_clean.csv')

drop_cols = ['SEQN', 'high_risk', 'prediabetes_risk', 'hypertension_risk',
             'DIQ010', 'BPQ020', 'BPXSY1', 'BPXDI1', 'BPXSY2', 'BPXDI2',
             'LBXGH', 'avg_systolic', 'avg_diastolic']
features = [c for c in df.columns if c not in drop_cols]

activity_flags = ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']
for col in activity_flags:
    if col in df.columns:
        df[col] = df[col].fillna(0)

X = df[features]
y = df['high_risk']

# ── LOCKED TEST SET (never used during agent loop) ────────────────
with open('locked_test_indices.json') as f:
    test_idx = json.load(f)

X_test_locked = X.loc[test_idx]
y_test_locked = y.loc[test_idx]

# ── TRAIN / VALIDATION SPLIT (what the agent actually sees) ───────
X_remaining = X.drop(index=test_idx)
y_remaining = y.drop(index=test_idx)

X_train, X_val, y_train, y_val = train_test_split(
    X_remaining, y_remaining,
    test_size=0.2, random_state=42, stratify=y_remaining
)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test (locked): {len(X_test_locked):,}")
print(f"Agent will ONLY evaluate on val set. Test set stays locked.\n")

# ── LOAD BASELINE TO BEAT ─────────────────────────────────────────
with open('baseline_results.json') as f:
    baseline = json.load(f)

best_auc = baseline['auc_roc']
best_gap = baseline['equalized_odds_gap']
print(f"Baseline to beat → AUC: {best_auc} | Fairness gap: {best_gap}\n")

# ── FAIRNESS EVALUATION FUNCTION ──────────────────────────────────
def fairness_eval(X_eval, y_true, y_pred, y_prob):
    eval_df = X_eval.copy()
    eval_df['y_true'] = y_true.values
    eval_df['y_pred'] = y_pred
    eval_df['y_prob'] = y_prob

    # Age groups
    eval_df['age_group'] = pd.cut(eval_df['RIDAGEYR'],
        bins=[17, 34, 49, 64, 80],   # exclude under-18 (too few positive cases)
        labels=['18-34', '35-49', '50-64', '65-80'])

    all_tprs = []
    subgroup_results = {}

    for group_col, group_map in [
        ('RIAGENDR', {1: 'Male', 2: 'Female'}),
        ('RIDRETH3', {1: 'Mexican American', 2: 'Other Hispanic',
                      3: 'NH White', 4: 'NH Black', 6: 'NH Asian'}),
        ('age_group', {'18-34': '18-34', '35-49': '35-49',
                       '50-64': '50-64', '65-80': '65-80'})
    ]:
        for code, label in group_map.items():
            sub = eval_df[eval_df[group_col] == code]
            if len(sub) < 20 or sub['y_true'].nunique() < 2:
                continue
            cm = confusion_matrix(sub['y_true'], sub['y_pred'], labels=[0, 1])
            if cm.shape == (2, 2):
                _, _, fn_s, tp_s = cm.ravel()
                tpr = round(tp_s / (tp_s + fn_s), 4) if (tp_s + fn_s) > 0 else 0
            else:
                tpr = 0
            all_tprs.append(tpr)
            subgroup_results[label] = tpr

    eq_gap = round(max(all_tprs) - min(all_tprs), 4) if all_tprs else 1.0
    return eq_gap, subgroup_results

# ── EXPERIMENT LOG ────────────────────────────────────────────────
LOG_FILE = 'experiment_log.csv'
log_fields = ['iteration', 'model', 'changes_made', 'auc_roc',
              'overall_tpr', 'eq_odds_gap', 'runtime_seconds',
              'fairness_pass', 'notes']

def log_result(row):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  → Logged to {LOG_FILE}")

# ── AGENT EVALUATION FUNCTION ─────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_v, y_v, iteration, model_name, changes, notes=""):
    start = time.time()
    model.fit(X_tr, y_tr)
    elapsed = round(time.time() - start, 2)

    y_pred = model.predict(X_v)
    y_prob = model.predict_proba(X_v)[:, 1]

    auc = round(roc_auc_score(y_v, y_prob), 4)
    cm  = confusion_matrix(y_v, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = round(tp / (tp + fn), 4)

    eq_gap, subgroups = fairness_eval(X_v, y_v, y_pred, y_prob)
    fairness_pass = "PASS" if eq_gap <= 0.05 else "FAIL"

    print(f"\n{'─'*52}")
    print(f"  Iteration {iteration}: {model_name}")
    print(f"  AUC-ROC:        {auc}  (baseline: {best_auc})")
    print(f"  Overall TPR:    {tpr}")
    print(f"  Fairness gap:   {eq_gap}  ({'PASS ✓' if fairness_pass == 'PASS' else 'FAIL ✗'})")
    print(f"  Runtime:        {elapsed}s")
    print(f"\n  Subgroup TPRs:")
    for g, t in subgroups.items():
        print(f"    {g:<22} {t}")

    log_result({
        'iteration': iteration, 'model': model_name,
        'changes_made': changes, 'auc_roc': auc,
        'overall_tpr': tpr, 'eq_odds_gap': eq_gap,
        'runtime_seconds': elapsed, 'fairness_pass': fairness_pass,
        'notes': notes
    })

    return auc, eq_gap, tpr

# ═══════════════════════════════════════════════════════════════════
# AGENT LOOP — each iteration proposes and evaluates a change
# ═══════════════════════════════════════════════════════════════════

print("=" * 52)
print("  AUTO-RESEARCHER AGENT LOOP STARTING")
print("=" * 52)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# ── ITERATION 1: XGBoost, no class balancing ──────────────────────
print("\n[Iteration 1] Trying XGBoost — no class balancing yet")
xgb1 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=42, eval_metric='logloss', verbosity=0)
auc1, gap1, tpr1 = evaluate(
    xgb1, X_train, y_train, X_val, y_val,
    iteration=1, model_name='xgboost_base',
    changes='switched from logistic regression to XGBoost (100 trees, depth 4)',
    notes='Testing if XGBoost improves over LR before adding complexity'
)

# ── ITERATION 2: XGBoost + class weights ─────────────────────────
print("\n[Iteration 2] XGBoost + class weights to address imbalance")
scale_pos = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
xgb2 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      scale_pos_weight=scale_pos,
                      random_state=42, eval_metric='logloss', verbosity=0)
auc2, gap2, tpr2 = evaluate(
    xgb2, X_train, y_train, X_val, y_val,
    iteration=2, model_name='xgboost_weighted',
    changes=f'added scale_pos_weight={scale_pos} to handle class imbalance',
    notes='Class weight should improve recall on minority class'
)

# ── ITERATION 3: XGBoost + SMOTE ─────────────────────────────────
print("\n[Iteration 3] XGBoost + SMOTE oversampling")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
xgb3 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=42, eval_metric='logloss', verbosity=0)
auc3, gap3, tpr3 = evaluate(
    xgb3, X_train_sm, y_train_sm, X_val, y_val,
    iteration=3, model_name='xgboost_smote',
    changes='replaced class weights with SMOTE oversampling',
    notes='SMOTE creates synthetic minority samples instead of reweighting'
)

# ── ITERATION 4: XGBoost + more trees + tuned depth ──────────────
print("\n[Iteration 4] XGBoost + tuned hyperparameters")
xgb4 = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=scale_pos,
                      random_state=42, eval_metric='logloss', verbosity=0)
auc4, gap4, tpr4 = evaluate(
    xgb4, X_train, y_train, X_val, y_val,
    iteration=4, model_name='xgboost_tuned',
    changes='300 trees, depth 5, lr 0.05, subsample 0.8, colsample 0.8',
    notes='Slower learning rate + more trees often improves generalization'
)

# ── AGENT DECISION: pick best model ──────────────────────────────
print(f"\n{'='*52}")
print("  AGENT DECISION SUMMARY")
print(f"{'='*52}")
results = [
    ('Baseline (LR)',       best_auc,  best_gap),
    ('Iteration 1 (XGB)',   auc1,      gap1),
    ('Iteration 2 (XGB+W)', auc2,      gap2),
    ('Iteration 3 (SMOTE)', auc3,      gap3),
    ('Iteration 4 (Tuned)', auc4,      gap4),
]
print(f"\n  {'Model':<24} {'AUC':>6}  {'Gap':>7}  {'Better?'}")
print(f"  {'─'*24} {'─'*6}  {'─'*7}  {'─'*7}")
for name, auc, gap in results:
    better = "✓" if auc > best_auc else "✗"
    print(f"  {name:<24} {auc:>6}  {gap:>7}  {better}")

# Pick iteration with best AUC (agent's primary objective)
best_iter = max([(auc1,1),(auc2,2),(auc3,3),(auc4,4)], key=lambda x: x[0])
print(f"\n  Best AUC achieved: Iteration {best_iter[1]} ({best_iter[0]})")
print(f"\n  NOTE: Fairness gap is driven mainly by age.")
print(f"  Run 04_evaluation.py to test the best model on the locked test set.")
print(f"{'='*52}")
