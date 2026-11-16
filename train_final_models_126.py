"""
Train the FIVE final models for the prototype and save them in one file (final_models.joblib).

Models (the five best on the 126 development patients in cross-validation, chosen BEFORE looking at the 12 test patients):
  Logistic Regression (10 features), Elastic-Net Logistic Regression (10), Extra Trees (20), Random Forest (20),
  and an Ensemble of LR + SVM + RF + XGBoost (10).  XGBoost and Histogram Gradient Boosting (CV AUC 0.57-0.62) were dropped. Their cross-validated AUC (repeated 5-fold, patient level, all steps inside the
folds) is printed and stored, so the app can show the honest number.

Put this file in the same folder as train_models.py, then run it.  Run it on the laptop that runs the app, so the
saved models match the installed scikit-learn / xgboost versions.
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_models as tm

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
OUT_FILE = r"C:\Users\ntech\Desktop\retrain_126\final_models_126.joblib"
# ====================================================================

SEED = 42
K_OF = {"LogisticRegression": 10, "ElasticNetLR": 10, "ExtraTrees": 20, "RandomForest": 20, "Ensemble": 10}
MODEL_NAMES = ["LogisticRegression", "ElasticNetLR", "ExtraTrees", "RandomForest", "Ensemble"]


def base_model(name):
    if name == "LogisticRegression":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    if name == "ElasticNetLR":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                                  class_weight="balanced", max_iter=5000, random_state=SEED)
    if name == "RandomForest":
        return tm.make_model("RandomForest")
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3, class_weight="balanced",
                                    random_state=SEED, n_jobs=1)
    if name == "Ensemble":
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
        return VotingClassifier([("lr", base_model("LogisticRegression")), ("svm", svm),
                                 ("rf", tm.make_model("RandomForest")), ("xgb", tm.make_model("XGBoost"))], voting="soft")
    raise ValueError(name)


def make_pipeline(name):
    return Pipeline([
        ("var", VarianceThreshold(1e-8)),
        ("corr", tm.CorrelationFilter(tm.CORR_LIMIT)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=K_OF[name])),
        ("model", base_model(name)),
    ])


def selected(pipe, feat_cols):
    names = np.array(feat_cols)
    names = names[pipe.named_steps["var"].get_support()]
    names = names[pipe.named_steps["corr"].keep_]
    return list(names[pipe.named_steps["select"].get_support()])


def main():
    data = tm.build_dataset()
    data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
    feat_cols = [c for c in data.columns if c.startswith("original_")]
    X = data[feat_cols].to_numpy(float)
    y = data["pCR"].to_numpy()
    print("Training patients: %d (pCR=1: %d)" % (len(y), y.sum()))

    models, cv_auc, sel = {}, {}, {}
    for name in MODEL_NAMES:
        pipe = make_pipeline(name)
        aucs, _ = tm.repeated_cv(pipe, X, y, y, 10)
        cv_auc[name] = float(aucs.mean())
        models[name] = clone(pipe).fit(X, y)
        sel[name] = selected(models[name], feat_cols)
        print("%-22s cross-validated AUC %.3f" % (name, cv_auc[name]))

    stats = pd.DataFrame({"mean": data[feat_cols].mean(), "std": data[feat_cols].std(),
                          "min": data[feat_cols].min(), "max": data[feat_cols].max()})
    bundle = {"models": models, "cv_auc": cv_auc, "feature_names": feat_cols, "selected_features": sel,
              "feature_stats": stats, "n_patients": len(y), "trained_on": "I-SPY2/ACRIN-6698 (TNBC and HER2+), baseline MRI"}
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    joblib.dump(bundle, OUT_FILE)
    print("\nSaved:", OUT_FILE)


if __name__ == "__main__":
    main()
