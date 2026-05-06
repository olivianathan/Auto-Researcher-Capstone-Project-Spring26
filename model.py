"""
model.py  —  AGENT-MUTABLE
This is the only file the agent is allowed to modify.

The agent's job is to rewrite build_model() to return a better sklearn estimator.
Everything else (data loading, evaluation, logging) is handled by run.py and prepare.py.

── EXPERIMENT HISTORY ────────────────────────────────────────────────────────
Iter 1  LR baseline            AUC 0.9242  obj 0.7777  → keep
Iter 2  LR balanced            AUC 0.9234  obj 0.7734  → discard (worse than iter 1)
Iter 3  Random Forest          AUC 0.9259  obj 0.7866  → keep
Iter 4  XGBoost base           AUC 0.9322  obj 0.8062  → keep
Iter 5  XGBoost threshold=0.75 AUC 0.9322  obj 0.8092  → keep  ← current best

Fairness note: all iterations fail the ≤0.05 equalized-odds gap target.
The age-group TPR gap is the dominant driver (iter 5 age gap = 0.8202).
Iter 6 applies stronger regularization (min_child_weight, gamma) to reduce the
model's reliance on age as a decision boundary.
──────────────────────────────────────────────────────────────────────────────

Current experiment: iter 6 — XGBoost regularized for age-group fairness
"""

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DESCRIPTION = (
    "Iter 6: XGBoost n=300 depth=4 lr=0.05 min_child_weight=5 gamma=0.1 "
    "colsample=0.7 — regularized to reduce age-driven decision boundary"
)

# 'fixed_0.50' — use threshold=0.5 (default)
# 'fairness_tuned' — run.py searches 0.25–0.75 for the threshold that minimizes equalized-odds gap
THRESHOLD_MODE = 'fairness_tuned'


# ── PAST ITERATIONS (frozen — do not modify) ──────────────────────

def build_iter1():
    """Iter 1: LR baseline. AUC 0.9242, obj 0.7777 → keep"""
    return LogisticRegression(max_iter=1000, random_state=42)


def build_iter2():
    """Iter 2: LR with class_weight=balanced. AUC 0.9234, obj 0.7734 → discard"""
    return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)


def build_iter3():
    """Iter 3: Random Forest balanced subsample. AUC 0.9259, obj 0.7866 → keep"""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
    )


def build_iter4():
    """Iter 4: XGBoost base. AUC 0.9322, obj 0.8062 → keep"""
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
    return build_iter3()


def build_iter5():
    """
    Iter 5: XGBoost with fairness-aware threshold tuning. AUC 0.9322, obj 0.8092 → keep
    Note: threshold=0.75 was applied externally in the notebook; this returns
    the same base estimator. Use predict_proba(...) >= 0.75 to reproduce iter 5 exactly.
    """
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
    return build_iter3()


# ── CURRENT EXPERIMENT ────────────────────────────────────────────

def build_model():
    """
    Return an sklearn-compatible estimator.
    Must implement .fit(X, y) and .predict_proba(X).

    Iter 6 strategy: heavier regularization (min_child_weight=5, gamma=0.1)
    discourages the model from splitting aggressively on age-correlated features,
    which drove the large age-group TPR gap in iters 4-5.
    Falls back to Random Forest if XGBoost is unavailable.
    """
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced_subsample',
        random_state=42,
    )
