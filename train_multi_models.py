"""
Six classical machine-learning models on the radiomics features (I-SPY2/ACRIN, 50 patients), compared fairly.

Models: Logistic Regression, SVM (RBF), K-Nearest Neighbours, Random Forest, XGBoost, and a soft-voting Ensemble
(LR + SVM + RF + XGBoost).  Each model is run with 10 and with 20 selected features.

Validation is identical for all models: repeated stratified 5-fold cross-validation, patient-level, and every
learning step (variance filter, correlation filter, scaling, feature selection, model) is fitted on the
training fold only.

DATA AUGMENTATION (optional, correct way):
If the file made by robustness_extract.py exists (AUGMENT_CSV), its perturbed-mask features (shrunk, grown,
shifted, randomly wobbled and solid-filled tumor masks) are added as EXTRA TRAINING ROWS of the training patients.
The test fold always contains ONLY the original rows of unseen patients. Augmented copies of a test patient are
never in the training set, so there is no leakage.

Put this file in the same folder as train_models.py (it re-uses its paths and helper functions).
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import train_models as tm      # re-uses build_dataset(), CorrelationFilter, bootstrap_ci(), paths

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
AUGMENT_CSV = r"C:\no_augmentation.csv"   # from robustness_extract.py (optional)
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\results_multi_126"
# ====================================================================

SEED = 42
N_REPEATS = 10
K_VALUES = [10, 20]
MODEL_NAMES = ["LogisticRegression", "SVM", "KNN", "RandomForest", "XGBoost", "Ensemble"]

os.makedirs(OUT_DIR, exist_ok=True)
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def base_model(name):
    if name == "LogisticRegression":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3,
                                      class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=5.0, min_child_weight=2, eval_metric="logloss", random_state=SEED,
                             n_jobs=1, verbosity=0)
    if name == "Ensemble":
        return VotingClassifier([("lr", base_model("LogisticRegression")), ("svm", base_model("SVM")),
                                 ("rf", base_model("RandomForest")), ("xgb", base_model("XGBoost"))], voting="soft")
    raise ValueError(name)


def make_pipeline(name, k):
    return Pipeline([
        ("var", VarianceThreshold(1e-8)),
        ("corr", tm.CorrelationFilter(tm.CORR_LIMIT)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", base_model(name)),
    ])


def cv_auc(pipe, X, y, strata, ids, aug, n_repeats):
    """Repeated stratified 5-fold CV. aug = {patient_id: array of augmented feature rows} or None."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    oof = np.zeros((n_repeats, len(y)))
    for n, (tr, te) in enumerate(cv.split(X, strata)):
        Xtr, ytr = X[tr], y[tr]
        if aug:
            extra_x, extra_y = [], []
            for i in tr:                                   # augmented copies of TRAINING patients only
                rows = aug.get(ids[i])
                if rows is not None and len(rows):
                    extra_x.append(rows)
                    extra_y.append(np.full(len(rows), y[i]))
            if extra_x:
                Xtr = np.vstack([Xtr] + extra_x)
                ytr = np.concatenate([ytr] + extra_y)
        p = clone(pipe).fit(Xtr, ytr)
        oof[n // 5, te] = p.predict_proba(X[te])[:, 1]      # test fold: original rows only
    aucs = np.array([roc_auc_score(y, oof[r]) for r in range(n_repeats)])
    return aucs, oof.mean(axis=0)


def load_augmentation(feat_cols, ids):
    if not os.path.exists(AUGMENT_CSV):
        return None
    a = pd.read_csv(AUGMENT_CSV)
    a = a[a["Patient_ID"].isin(set(ids))]
    missing = [c for c in feat_cols if c not in a.columns]
    if missing or a.empty:
        return None
    return {pid: g[feat_cols].to_numpy(dtype=float) for pid, g in a.groupby("Patient_ID")}


def main():
    data = tm.build_dataset()
    data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
    feat_cols = [c for c in data.columns if c.startswith("original_")]
    X = data[feat_cols].to_numpy(dtype=float)
    y = data["pCR"].to_numpy()
    ids = data["Patient_ID"].to_numpy()
    strata = y
    say("Analysis set: %d I-SPY2/ACRIN patients, pCR=1: %d, pCR=0: %d, %d features" % (len(y), y.sum(), (1 - y).sum(), len(feat_cols)))

    aug = load_augmentation(feat_cols, ids)
    settings = [("no augmentation", None)]
    if aug:
        n_rows = sum(len(v) for v in aug.values())
        say("Augmentation file found: %d extra rows from %d patients (perturbed tumor masks)" % (n_rows, len(aug)))
        settings.append(("with mask-perturbation augmentation (training folds only)", aug))
    else:
        say("No augmentation file found (run robustness_extract.py first) -> only the no-augmentation comparison.")

    rows, oofs = [], {}
    for label, a in settings:
        say("\n=== %s: repeated stratified 5-fold CV, %d repeats ===" % (label, N_REPEATS))
        for name in MODEL_NAMES:
            for k in K_VALUES:
                aucs, oof = cv_auc(make_pipeline(name, k), X, y, strata, ids, a, N_REPEATS)
                lo, hi = tm.bootstrap_ci(y, oof)
                rows.append({"setting": label, "model": name, "k_features": k, "AUC_mean": aucs.mean(),
                             "AUC_sd_over_repeats": aucs.std(), "AUC_pooled_oof": roc_auc_score(y, oof),
                             "CI95_low": lo, "CI95_high": hi})
                oofs[(label, name, k)] = oof
                say("%-18s k=%-2d  AUC %.3f +/- %.3f   pooled out-of-fold %.3f  95%% CI [%.2f, %.2f]" % (
                    name, k, aucs.mean(), aucs.std(), roc_auc_score(y, oof), lo, hi))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "multi_model_results.csv"), index=False)

    say("\n=== SUMMARY: best k per model ===")
    for label, _ in settings:
        say("[%s]" % label)
        sub = res[res["setting"] == label]
        for name in MODEL_NAMES:
            b = sub[sub["model"] == name].sort_values("AUC_mean", ascending=False).iloc[0]
            say("  %-18s AUC %.3f (k=%d)" % (name, b["AUC_mean"], b["k_features"]))
    say("\nNote: %d model/feature-count combinations were compared; the best one is slightly optimistic." % (len(MODEL_NAMES) * len(K_VALUES)))

    # ROC curves, one line per model (best k), first setting
    for label, _ in settings:
        sub = res[res["setting"] == label]
        plt.figure(figsize=(6, 5.5))
        for name in MODEL_NAMES:
            b = sub[sub["model"] == name].sort_values("AUC_mean", ascending=False).iloc[0]
            fpr, tpr, _ = roc_curve(y, oofs[(label, name, int(b["k_features"]))])
            plt.plot(fpr, tpr, label="%s (AUC %.2f)" % (name, b["AUC_pooled_oof"]))
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("Out-of-fold ROC, n=%d\n%s" % (len(y), label), fontsize=9)
        plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "roc_all_models_%s.png" % ("aug" if "augment" in label and "no " not in label else "noaug")), dpi=200)
        plt.close()

    with open(os.path.join(OUT_DIR, "multi_model_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nSaved in", OUT_DIR)


if __name__ == "__main__":
    main()
