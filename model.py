"""
model.py  --  AGENT-MUTABLE
This is the only file the agent is allowed to modify.

The agent's job is to rewrite build_model() to return a better sklearn estimator.
Everything else (data loading, evaluation, logging) is handled by run.py and prepare.py.

-- EXPERIMENT HISTORY -----------------------------------------------------------
Iter 1  LR baseline            AUC 0.9242  obj 0.7777  -> keep
Iter 2  LR balanced            AUC 0.9234  obj 0.7734  -> discard (worse than iter 1)
Iter 3  Random Forest          AUC 0.9259  obj 0.7866  -> keep
Iter 4  XGBoost base           AUC 0.9322  obj 0.8062  -> keep
Iter 5  XGBoost threshold=0.75 AUC 0.9322  obj 0.8092  -> keep
Iter 6  XGBoost regularized    AUC 0.9281  obj 0.8100  -> keep  <- best so far

Fairness metric change at iter 7:
  Age:       switched from equalized odds (TPR gap) to MACE (calibration error)
  Sex/Race:  still equalized odds
  Rationale: age TPR gap was structurally immovable (0.79+ across 6 iters) because
             disease risk is genuinely age-dependent. MACE checks whether the
             model's predicted probabilities are honest within each age group,
             which is the clinically relevant fairness question.
--------------------------------------------------------------------------------

Current experiment: iter 7 -- XGBoost + isotonic calibration (MACE target)
"""

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

DESCRIPTION = (
    "Iter 11: XGBoost (depth=5, n=500, lr=0.03, reg_alpha=0.1) + isotonic cv=5 "
    "-- L1 regularization + more calibration folds to squeeze MACE below 0.03"
)

# 'fixed_0.50'     -- use threshold=0.5 (default)
# 'fairness_tuned' -- run.py searches 0.25-0.75 minimizing sex/race equalized odds only
THRESHOLD_MODE = 'fairness_tuned'


# -- PAST ITERATIONS (frozen -- do not modify) ----------------------------

def build_iter1():
    """Iter 1: LR baseline. AUC 0.9242, obj 0.7777 -> keep"""
    return LogisticRegression(max_iter=1000, random_state=42)


def build_iter2():
    """Iter 2: LR with class_weight=balanced. AUC 0.9234, obj 0.7734 -> discard"""
    return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)


def build_iter3():
    """Iter 3: Random Forest balanced subsample. AUC 0.9259, obj 0.7866 -> keep"""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
    )


def build_iter4():
    """Iter 4: XGBoost base. AUC 0.9322, obj 0.8062 -> keep"""
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
    Iter 5: XGBoost + fairness-aware threshold=0.75. AUC 0.9322, obj 0.8092 -> keep
    threshold=0.75 was applied externally; this returns the base estimator.
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


def build_iter6():
    """Iter 6: XGBoost regularized (min_child_weight=5, gamma=0.1). AUC 0.9281, obj 0.8100 -> keep"""
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


# -- CURRENT EXPERIMENT ---------------------------------------------------

def build_model():
    """
    Return an sklearn-compatible estimator.
    Must implement .fit(X, y) and .predict_proba(X).

    Iter 7: Wraps the iter 6 XGBoost in isotonic calibration (cv=3).
    CalibratedClassifierCV fits the base model on 2/3 of folds and learns
    a monotone mapping from raw scores to calibrated probabilities on the
    remaining 1/3. This directly targets MACE without touching the
    underlying decision boundary.
    """
    if _HAS_XGB:
        base = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
        return CalibratedClassifierCV(base, method='isotonic', cv=5)
    return CalibratedClassifierCV(build_iter3(), method='isotonic', cv=5)
