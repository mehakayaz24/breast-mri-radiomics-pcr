"""
Hyperparameter tuning done the correct way: NESTED cross-validation on the 126 I-SPY2/ACRIN patients.

Outer loop : repeated stratified 5-fold CV. The outer test fold is never used for tuning.
Inner loop : 5-fold grid search on the outer TRAINING fold only (the whole pipeline is tuned: number of selected
             features k and the model settings). The best settings are refitted on the outer training fold and
             evaluated once on the outer test fold.
Result     : the out-of-fold AUC of the TUNED models, compared with the fixed (untuned) settings of train_models.py.

Put this file in the same folder as train_models.py (retrain_126), then run:  python train_nested_tuning.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

import train_models as tm

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\results_nested_126"
N_REPEATS = 3          # outer repeats (more repeats = slower)
N_JOBS = -1            # use all CPU cores for the inner grid search
# ====================================================================

SEED = tm.SEED
GRIDS = {
    "RandomForest": {"select__k": [10, 20],
                     "model__max_depth": [2, 3, 5],
                     "model__min_samples_leaf": [2, 3, 5]},
    "LogisticRegression": {"select__k": [10, 20],
                           "model__C": [0.01, 0.1, 1.0]},
    "XGBoost": {"select__k": [10, 20],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.05, 0.1]},
}
os.makedirs(OUT_DIR, exist_ok=True)
lines = []


def say(t=""):
    print(t)
    lines.append(t)


data = tm.build_dataset()
data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
feat_cols = [c for c in data.columns if c.startswith("original_")]
X = data[feat_cols].to_numpy(dtype=float)
y = data["pCR"].to_numpy()
say("Patients: %d (pCR=1: %d), features: %d" % (len(y), y.sum(), len(feat_cols)))

rows = []
for name, grid in GRIDS.items():
  try:
      cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
      oof = np.zeros((N_REPEATS, len(y)))
      chosen = []
      for n, (tr, te) in enumerate(cv.split(X, y)):
          gs = GridSearchCV(tm.make_pipeline(name, 20), grid, scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=N_JOBS, refit=True)
          gs.fit(X[tr], y[tr])
          oof[n // 5, te] = gs.predict_proba(X[te])[:, 1]
          chosen.append(str(gs.best_params_))
      aucs = np.array([roc_auc_score(y, oof[r]) for r in range(N_REPEATS)])
      pooled = oof.mean(axis=0)
      lo, hi = tm.bootstrap_ci(y, pooled)
      # same outer folds with the FIXED settings, for a fair comparison
      fixed_aucs, fixed_oof = tm.repeated_cv(tm.make_pipeline(name, 20), X, y, y, N_REPEATS)
      top = pd.Series(chosen).value_counts().head(3)
      say("\n%s" % name)
      say("  tuned (nested CV):  AUC %.3f +/- %.3f   pooled %.3f  95%% CI [%.2f, %.2f]" % (aucs.mean(), aucs.std(), roc_auc_score(y, pooled), lo, hi))
      say("  fixed settings:     AUC %.3f +/- %.3f" % (fixed_aucs.mean(), fixed_aucs.std()))
      say("  most often chosen settings:")
      for s, c in top.items():
          say("     %2d of %d folds: %s" % (c, len(chosen), s))
      rows.append({"model": name, "tuned_AUC": aucs.mean(), "tuned_sd": aucs.std(), "tuned_CI_low": lo, "tuned_CI_high": hi,
                   "fixed_AUC": fixed_aucs.mean(), "fixed_sd": fixed_aucs.std()})
  except Exception as e:
    say("\n%s: skipped (%s)" % (name, e))

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "nested_tuning_results.csv"), index=False)
with open(os.path.join(OUT_DIR, "nested_tuning_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
say("\nSaved in " + OUT_DIR)
