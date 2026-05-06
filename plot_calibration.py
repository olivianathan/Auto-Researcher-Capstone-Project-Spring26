"""
plot_calibration.py
Generates a reliability diagram (calibration curve) per age group for the
current model in model.py. Run after prepare.py has been run at least once.

Usage:
    python plot_calibration.py

Output:
    calibration.png
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from model import build_model, DESCRIPTION, THRESHOLD_MODE

# -- 1. LOAD DATA ---------------------------------------------------------
print(f"Model: {DESCRIPTION}")
print("Loading prepared data...")

X_train = pd.read_csv('X_train.csv', index_col=0)
X_val   = pd.read_csv('X_val.csv',   index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0).squeeze()
y_val   = pd.read_csv('y_val.csv',   index_col=0).squeeze()

# -- 2. FIT MODEL ---------------------------------------------------------
model = build_model()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

try:
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_val_s)[:, 1]
except Exception:
    model = build_model()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val.values)[:, 1]

print("Model fitted. Generating calibration curves...")

# -- 3. AGE BINS ----------------------------------------------------------
age_bins   = [0, 17, 34, 49, 64, 80]
age_labels = ['0-17', '18-34', '35-49', '50-64', '65-80']

age_bucket = pd.cut(X_val['RIDAGEYR'], bins=age_bins, labels=age_labels)

# -- 4. PLOT --------------------------------------------------------------
n_bins = 8  # calibration curve bins

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: reliability diagram per age group
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

mace_per_group = {}
for grp in age_labels:
    mask = (age_bucket == grp).values
    if mask.sum() < 30 or y_val.values[mask].sum() < 5:
        continue
    frac_pos, mean_pred = calibration_curve(
        y_val.values[mask], y_prob[mask],
        n_bins=n_bins, strategy='uniform'
    )
    mace = float(np.mean(np.abs(frac_pos - mean_pred)))
    mace_per_group[grp] = mace
    ax.plot(mean_pred, frac_pos, marker='o', markersize=4,
            label=f'{grp}  (MACE={mace:.3f})')

ax.set_title('Reliability Diagram by Age Group')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Actual positive rate')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Panel 2: overall reliability diagram (all ages combined)
ax2 = axes[1]
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

frac_pos_all, mean_pred_all = calibration_curve(
    y_val.values, y_prob, n_bins=n_bins, strategy='uniform'
)
overall_mace = float(np.mean(np.abs(frac_pos_all - mean_pred_all)))
ax2.plot(mean_pred_all, frac_pos_all, marker='o', color='steelblue',
         label=f'Overall  (MACE={overall_mace:.3f})')

# shade the gap between curve and diagonal
ax2.fill_between(mean_pred_all,
                 mean_pred_all, frac_pos_all,
                 alpha=0.15, color='steelblue', label='calibration gap')

ax2.set_title('Overall Reliability Diagram')
ax2.set_xlabel('Mean predicted probability')
ax2.set_ylabel('Actual positive rate')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(alpha=0.3)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)

fig.suptitle(f'Calibration -- {DESCRIPTION}', fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved -> calibration.png")
print("\nMACE per age group:")
for grp, mace in mace_per_group.items():
    bar = '#' * int(mace * 40)
    print(f"  {grp:<6}  {mace:.4f}  {bar}")
print(f"\n  Overall MACE: {overall_mace:.4f}")
