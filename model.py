"""
model.py  --  AGENT-MUTABLE
This is the only file the agent is allowed to modify.

The agent's job is to rewrite build_model() to return a better sklearn estimator.
Everything else (data loading, evaluation, logging) is handled by run.py and prepare.py.

-- EXPERIMENT HISTORY -----------------------------------------------------------
Iter 1   LR baseline             AUC 0.9242  obj 0.7777  -> keep
Iter 2   LR balanced             AUC 0.9234  obj 0.7734  -> discard
Iter 3   Random Forest           AUC 0.9259  obj 0.7866  -> keep
Iter 4   XGBoost base            AUC 0.9322  obj 0.8062  -> keep
Iter 5   XGBoost threshold=0.75  AUC 0.9322  obj 0.8092  -> keep
Iter 6   XGBoost regularized     AUC 0.9281  obj 0.8100  -> keep

[Metric switch at iter 7: age fairness = MACE instead of equalized odds]

Iter 7   XGBoost + isotonic cv=3  AUC 0.9273  obj 0.9210  -> keep  <- calibration baseline
Iter 8   XGBoost + sigmoid  cv=3  AUC 0.9280  obj 0.9215  -> keep
Iter 9   RF + isotonic cv=3       AUC 0.9245  obj 0.9136  -> discard [CONFOUNDED: base model changed]
Iter 10  XGBoost deeper + iso cv=3 AUC 0.9272 obj 0.9224 -> keep   [CONFOUNDED: xgb params changed]
Iter 11  XGBoost deeper + iso cv=5 AUC 0.9276 obj 0.9228 -> keep   [CONFOUNDED: params + cv changed]

Controlled re-runs (iters 12-14): base model frozen at iter 7 XGBoost, one variable at a time.
  Iter 12: cv 3->5  (only cv changes)
  Iter 13: cv 3->10 (only cv changes)
  Iter 14: two-stage calibration -- global isotonic cv=3 + per-age-quintile isotonic second pass
--------------------------------------------------------------------------------

Current best: iter 12 (cv=5). Iters 13-14 were discarded.
"""

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

DESCRIPTION = (
    "Iter 12: XGBoost (iter7 base, frozen) + CalibratedClassifierCV isotonic cv=5 "
    "-- only change from iter7: cv 3->5 [BEST KEPT: obj=0.9237]"
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
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric='logloss', verbosity=0,
        )
    return build_iter3()


def build_iter5():
    """Iter 5: XGBoost + threshold=0.75. AUC 0.9322, obj 0.8092 -> keep"""
    return build_iter4()


def build_iter6():
    """Iter 6: XGBoost regularized. AUC 0.9281, obj 0.8100 -> keep"""
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, gamma=0.1,
            random_state=42, eval_metric='logloss', verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight='balanced_subsample', random_state=42,
    )


def _base_xgb():
    """Frozen base XGBoost used as the fixed model for iters 7-14."""
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, gamma=0.1,
            random_state=42, eval_metric='logloss', verbosity=0,
        )
    return build_iter3()


def build_iter7():
    """Iter 7: _base_xgb + isotonic cv=3. AUC 0.9273, obj 0.9210 -> keep (calibration baseline)"""
    return CalibratedClassifierCV(_base_xgb(), method='isotonic', cv=3)


def build_iter8():
    """Iter 8: _base_xgb + sigmoid cv=3. AUC 0.9280, obj 0.9215 -> keep"""
    return CalibratedClassifierCV(_base_xgb(), method='sigmoid', cv=3)


# -- TWO-STAGE AGE-QUINTILE CALIBRATOR (for iter 14) ---------------------

class AgeQuintileCalibrator:
    """
    Stage 1: CalibratedClassifierCV(isotonic, cv=3) applied globally.
    Stage 2: per-age-quintile IsotonicRegression applied on top of stage-1 probs.

    Assumes RIDAGEYR is at column index 1 in the (scaled) feature array,
    consistent with prepare.py's feature ordering: [RIAGENDR, RIDAGEYR, RIDRETH3, ...].
    Quintile boundaries are learned from training data so no raw age values needed.
    """

    RIDAGEYR_COL = 1  # column index in scaled feature array
    N_QUINTILES  = 5  # one per age band

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def _age_col(self, X):
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.RIDAGEYR_COL].values
        return X[:, self.RIDAGEYR_COL]

    def fit(self, X, y):
        # Stage 1
        self.stage1_ = CalibratedClassifierCV(
            self.base_estimator, method='isotonic', cv=3
        )
        self.stage1_.fit(X, y)

        # Compute quintile bin edges from training ages (scaled but order-preserving)
        age = self._age_col(X)
        pcts = np.linspace(0, 100, self.N_QUINTILES + 1)
        edges = np.percentile(age, pcts)
        edges[0]  -= 1e-6
        edges[-1] += 1e-6
        self.bin_edges_ = edges

        # Stage 2: fit per-quintile isotonic on stage-1 outputs
        s1_probs = self.stage1_.predict_proba(X)[:, 1]
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        self.stage2_ = {}
        for i in range(self.N_QUINTILES):
            mask = (age >= edges[i]) & (age < edges[i + 1])
            if mask.sum() < 20:
                self.stage2_[i] = None
                continue
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(s1_probs[mask], y_arr[mask])
            self.stage2_[i] = ir
        return self

    def predict_proba(self, X):
        probs = self.stage1_.predict_proba(X)[:, 1].copy()
        age   = self._age_col(X)
        for i in range(self.N_QUINTILES):
            mask = (age >= self.bin_edges_[i]) & (age < self.bin_edges_[i + 1])
            if self.stage2_.get(i) is None or mask.sum() == 0:
                continue
            probs[mask] = self.stage2_[i].predict(probs[mask])
        return np.column_stack([1 - probs, probs])


# -- CURRENT EXPERIMENT ---------------------------------------------------

def build_model():
    """
    Best kept model: iter 12 -- frozen _base_xgb + CalibratedClassifierCV(isotonic, cv=5).
    Iters 13 (cv=10, tied) and 14 (two-stage, regressed) were both discarded.
    """
    return CalibratedClassifierCV(_base_xgb(), method='isotonic', cv=5)
