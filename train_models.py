"""
pCR prediction from baseline breast-MRI radiomics (I-SPY1 + I-SPY2/ACRIN-6698).

USE_COHORT = "ISPY2": primary analysis on the 50 I-SPY2/ACRIN-6698 patients (25 pCR / 25 non-pCR); the 19 I-SPY1
patients are kept aside as a small external test set. USE_COHORT = "ALL": pooled analysis of all 69 patients.

Models (decided in advance): Random Forest and XGBoost, 10 or 20 selected features.
Hyper-parameters are fixed in advance (no tuning on 69 patients), all four configurations are reported.

Everything that learns from the data (variance filter, correlation filter, scaling, feature selection, model)
sits INSIDE the cross-validation pipeline, so the test fold is never seen during selection.

What this script produces (in OUT_DIR):
  merged_data_all.csv              features + pCR label + subtype for all patients with a label
  results_summary.csv              repeated-CV AUC for every configuration
  validation_report.txt            all numbers (AUC, bootstrap CI, permutation test, baselines, cross-cohort ...)
  roc_best_model.png, auc_comparison.png
  selected_features.csv            how often each feature was picked inside the CV folds (best model)
  final_model.joblib               best configuration fitted on the analysed patients (for the Streamlit prototype)
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
FEATURES_CSV = r"C:\Users\ntech\Desktop\retrain_126\radiomics_features_126.csv"
ISPY1_XLSX = r"C:\Users\ntech\Desktop\slicer\I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx"
ISPY2_XLSX = r"C:\Users\ntech\Desktop\ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\results_126"
USE_COHORT = "ISPY2"     # "ISPY2" (50 patients, primary) or "ALL" (pooled 69 patients)
# ====================================================================

SEED = 42
N_REPEATS = 10          # repeated stratified 5-fold CV
N_PERMUTATIONS = 100    # label-permutation test
N_BOOTSTRAP = 2000
K_VALUES = [10, 20]
CORR_LIMIT = 0.90

os.makedirs(OUT_DIR, exist_ok=True)
report_lines = []


def say(text=""):
    print(text)
    report_lines.append(text)


# ----------------------------------------------------------------- data
def build_dataset():
    f = pd.read_csv(FEATURES_CSV)
    f["Case_ID"] = f["Case_ID"].astype(str)

    c1 = pd.read_excel(ISPY1_XLSX, sheet_name="TCIA Patient Clinical Subset")
    o1 = pd.read_excel(ISPY1_XLSX, sheet_name="TCIA Outcomes Subset")
    m1 = c1[["SUBJECTID", "HR_HER2_STATUS"]].merge(o1[["SUBJECTID", "PCR"]], on="SUBJECTID")
    m1["Case_ID"] = m1["SUBJECTID"].astype(str)
    m1["subtype"] = m1["HR_HER2_STATUS"].map({"TripleNeg": "TNBC", "HER2pos": "HER2+"}).fillna("other")
    m1 = m1.rename(columns={"PCR": "pCR"})[["Case_ID", "pCR", "subtype"]]
    a = f[f["Dataset"] == "ISPY1"].merge(m1, on="Case_ID", how="left")

    c2 = pd.read_excel(ISPY2_XLSX)
    c2["Case_ID"] = c2["Patient_ID"].astype(str)
    c2["subtype"] = np.where((c2["HR"] == 0) & (c2["HER2"] == 0), "TNBC",
                             np.where(c2["HER2"] == 1, "HER2+", "other"))
    b = f[f["Dataset"] != "ISPY1"].merge(c2[["Case_ID", "pCR", "subtype"]], on="Case_ID", how="left")

    d = pd.concat([a, b], ignore_index=True)
    n_before = len(d)
    d = d[d["pCR"].notna()].copy()
    d["pCR"] = d["pCR"].astype(int)
    d["cohort"] = np.where(d["Dataset"] == "ISPY1", "ISPY1", "ISPY2")   # ACRIN-6698 patients belong to I-SPY2 imaging cohort 1
    say("Patients with features: %d, with a pCR label: %d (dropped %d without label)" % (n_before, len(d), n_before - len(d)))
    return d


# ------------------------------------------------------------ pipeline
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature of every pair with |correlation| above the limit (decided on the training data only)."""

    def __init__(self, limit=0.9):
        self.limit = limit

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr)
        keep = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if not keep[i]:
                continue
            for j in range(i + 1, X.shape[1]):
                if keep[j] and corr[i, j] > self.limit:
                    keep[j] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_model(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3,
                                      class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=2,
                             eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    if name == "LogisticRegression":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    raise ValueError(name)


def make_pipeline(model_name, k):
    return Pipeline([
        ("var", VarianceThreshold(1e-8)),
        ("corr", CorrelationFilter(CORR_LIMIT)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", make_model(model_name)),
    ])


def proba(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def repeated_cv(pipe, X, y, strata, n_repeats, seed=SEED):
    """Returns per-repeat pooled out-of-fold AUCs and the out-of-fold probability averaged over repeats."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(y)))
    for n, (tr, te) in enumerate(cv.split(X, strata)):
        r = n // 5
        p = clone(pipe).fit(X[tr], y[tr])
        oof[r, te] = proba(p, X[te])
    aucs = np.array([roc_auc_score(y, oof[r]) for r in range(n_repeats)])
    return aucs, oof.mean(axis=0)


def bootstrap_ci(y, p, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def make_strata(y, cohort):
    """Stratify on label (and on cohort too when more than one cohort is analysed)."""
    if len(set(cohort)) == 1:
        return np.asarray(y)
    return np.array(["%d_%s" % (a, b) for a, b in zip(y, cohort)])


def main():
    data_all = build_dataset()
    data_all.to_csv(os.path.join(OUT_DIR, "merged_data_all.csv"), index=False)
    if USE_COHORT == "ALL":
        data, ext = data_all.reset_index(drop=True), None
    else:
        data = data_all[data_all["cohort"] == USE_COHORT].reset_index(drop=True)
        ext = data_all[data_all["cohort"] != USE_COHORT].reset_index(drop=True)

    feat_cols = [c for c in data.columns if c.startswith("original_")]
    X = data[feat_cols].to_numpy(dtype=float)
    y = data["pCR"].to_numpy()
    cohort = data["cohort"].to_numpy()
    strata = make_strata(y, cohort)

    say("\n=== DATA (analysis set: %s) ===" % USE_COHORT)
    say("Patients: %d   pCR=1: %d   pCR=0: %d   features: %d" % (len(y), y.sum(), (1 - y).sum(), len(feat_cols)))
    say(pd.crosstab([data["cohort"], data["subtype"]], data["pCR"]).to_string())
    if ext is not None and len(ext):
        say("Kept aside as external test set: %d patients (%s), pCR=%d" % (len(ext), ", ".join(sorted(set(ext["cohort"]))), ext["pCR"].sum()))

    # ------------------------------------------------ main experiment
    say("\n=== REPEATED STRATIFIED 5-FOLD CV (%d repeats), all steps inside the folds ===" % N_REPEATS)
    rows, oof_store = [], {}
    for model_name in ["RandomForest", "XGBoost"]:
        for k in K_VALUES:
            aucs, oof = repeated_cv(make_pipeline(model_name, k), X, y, strata, N_REPEATS)
            lo, hi = bootstrap_ci(y, oof)
            rows.append({"model": model_name, "k_features": k, "AUC_mean": aucs.mean(), "AUC_sd_over_repeats": aucs.std(),
                         "AUC_pooled_oof": roc_auc_score(y, oof), "CI95_low": lo, "CI95_high": hi})
            oof_store[(model_name, k)] = oof
            say("%-13s k=%-2d  AUC %.3f +/- %.3f (sd over repeats)   pooled out-of-fold AUC %.3f  95%% CI [%.2f, %.2f]" % (
                model_name, k, aucs.mean(), aucs.std(), roc_auc_score(y, oof), lo, hi))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)

    best = res.sort_values("AUC_mean", ascending=False).iloc[0]
    best_key = (best["model"], int(best["k_features"]))
    best_oof = oof_store[best_key]
    say("\nBest configuration by mean CV AUC: %s with %d features (AUC %.3f)." % (best_key[0], best_key[1], best["AUC_mean"]))
    say("Note: 4 configurations were compared, so the best one is slightly optimistic.")

    # ---------------------------------------------- permutation test
    say("\n=== PERMUTATION TEST (labels shuffled, same pipeline, %d permutations) ===" % N_PERMUTATIONS)
    rng = np.random.default_rng(SEED)
    null = []
    pipe_best = make_pipeline(*best_key)
    for i in range(N_PERMUTATIONS):
        yp = rng.permutation(y)
        a_, _ = repeated_cv(pipe_best, X, yp, make_strata(yp, cohort), 2, seed=1000 + i)
        null.append(a_.mean())
    null = np.array(null)
    pval = (np.sum(null >= best["AUC_mean"]) + 1) / (len(null) + 1)
    say("Observed AUC %.3f | chance-level AUC from shuffled labels: mean %.3f, 95th percentile %.3f | p = %.3f" % (
        best["AUC_mean"], null.mean(), np.percentile(null, 95), pval))

    # --------------------------------------------------- baselines
    say("\n=== BASELINES AND SHORTCUT CHECKS ===")
    sub = pd.get_dummies(data["subtype"]).to_numpy(dtype=float)
    pipe_lr = Pipeline([("scale", StandardScaler()), ("model", make_model("LogisticRegression"))])
    a_sub, _ = repeated_cv(pipe_lr, sub, y, strata, N_REPEATS)
    say("Subtype only (TNBC vs HER2+) -> pCR:          AUC %.3f  (is radiomics better than just knowing the subtype?)" % a_sub.mean())
    vol_auc = roc_auc_score(y, -data["original_shape_MeshVolume"].to_numpy())
    say("Tumor volume only (bigger tumor -> less pCR):  AUC %.3f  (direction fixed in advance, no model fitted)" % vol_auc)
    bar_labels, bar_vals = ["Subtype only", "Tumor volume only"], [a_sub.mean(), vol_auc]
    if len(set(cohort)) > 1:
        coh = (cohort == "ISPY1").astype(float).reshape(-1, 1)
        a_coh, _ = repeated_cv(pipe_lr, coh, y, strata, N_REPEATS)
        say("Cohort only (ISPY1 vs ISPY2) -> pCR:          AUC %.3f  (should be near 0.5, otherwise cohort is a shortcut)" % a_coh.mean())
        strata_c = np.array(["%s_%d" % (a, b) for a, b in zip(cohort, y)])
        a_b, _ = repeated_cv(make_pipeline("RandomForest", 10), X, (cohort == "ISPY1").astype(int), strata_c, 3)
        say("Radiomics -> which cohort (ISPY1 vs ISPY2):   AUC %.3f  (near 1.0 = strong scanner/protocol difference in the features)" % a_b.mean())
        bar_labels.append("Cohort only"); bar_vals.append(a_coh.mean())

    # ------------------------------------------ within-group AUC
    say("\n=== AUC INSIDE SUBGROUPS (best model, out-of-fold predictions) ===")
    groups = [(c, cohort == c) for c in sorted(set(cohort))] if len(set(cohort)) > 1 else []
    groups += [(s_, (data["subtype"] == s_).to_numpy()) for s_ in ["TNBC", "HER2+"]]
    for name, m in groups:
        if len(set(y[m])) == 2:
            say("%s only (n=%d, pCR=%d): AUC %.3f" % (name, m.sum(), y[m].sum(), roc_auc_score(y[m], best_oof[m])))

    # ---------------------------------------- external test set
    if ext is not None and len(ext):
        say("\n=== TRAIN ON THE %d ANALYSED PATIENTS, TEST ON THE %d KEPT-ASIDE ISPY1 PATIENTS ===" % (len(y), len(ext)))
        p = clone(pipe_best).fit(X, y)
        ye = ext["pCR"].to_numpy()
        pe = proba(p, ext[feat_cols].to_numpy(dtype=float))
        say("%s k=%d: AUC on ISPY1 = %.3f (only %d test patients, so very uncertain)" % (best_key[0], best_key[1], roc_auc_score(ye, pe), len(ext)))
    elif len(set(cohort)) > 1:
        say("\n=== TRAIN ON ISPY2/ACRIN (n=%d), TEST ON ISPY1 (n=%d) ===" % ((cohort == "ISPY2").sum(), (cohort == "ISPY1").sum()))
        tr_m, te_m = cohort == "ISPY2", cohort == "ISPY1"
        p = clone(pipe_best).fit(X[tr_m], y[tr_m])
        say("%s k=%d: AUC on ISPY1 = %.3f (only %d test patients, so very uncertain)" % (
            best_key[0], best_key[1], roc_auc_score(y[te_m], proba(p, X[te_m])), te_m.sum()))

    # ------------------------------------------- feature stability
    sel_count = pd.Series(0, index=feat_cols, dtype=float)
    n_fit = 0
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    for tr_i, _ in skf.split(X, strata):
        p = clone(pipe_best).fit(X[tr_i], y[tr_i])
        cols = np.array(feat_cols)
        cols = cols[p.named_steps["var"].get_support()]
        cols = cols[p.named_steps["corr"].keep_]
        cols = cols[p.named_steps["select"].get_support()]
        sel_count[cols] += 1
        n_fit += 1
    stab = (sel_count / n_fit).sort_values(ascending=False)
    stab.rename("selected_in_fraction_of_folds").to_csv(os.path.join(OUT_DIR, "selected_features.csv"))
    say("\n=== MOST STABLE FEATURES (picked in the largest share of CV folds) ===")
    for name, v in stab.head(10).items():
        say("%.2f  %s" % (v, name))

    # ------------------------------------------------- plots
    fpr, tpr, _ = roc_curve(y, best_oof)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="%s, %d features\nAUC %.2f (95%% CI %.2f-%.2f)" % (best_key[0], best_key[1], best["AUC_pooled_oof"], best["CI95_low"], best["CI95_high"]))
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Out-of-fold ROC (repeated 5-fold CV, n=%d)" % len(y)); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "roc_best_model.png"), dpi=200); plt.close()

    labels = ["%s k=%d" % (r.model, r.k_features) for r in res.itertuples()] + bar_labels
    vals = list(res["AUC_mean"]) + bar_vals
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(labels, vals, color=["#4c78a8"] * len(res) + ["#bbbbbb"] * len(bar_labels))
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1)
    plt.ylim(0.3, 1.0); plt.ylabel("Mean CV AUC"); plt.xticks(rotation=25, ha="right")
    plt.title("Radiomics models vs baselines (n=%d)" % len(y)); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "auc_comparison.png"), dpi=200); plt.close()

    # --------------------------------------------- final model
    final = clone(pipe_best).fit(X, y)
    joblib.dump({"pipeline": final, "feature_names": feat_cols, "config": best_key, "trained_on": USE_COHORT, "n_patients": len(y)},
                os.path.join(OUT_DIR, "final_model.joblib"))
    say("\nFinal model (%s, %d features) fitted on all %d analysed patients and saved as final_model.joblib" % (best_key[0], best_key[1], len(y)))

    with open(os.path.join(OUT_DIR, "validation_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    print("\nAll results saved in", OUT_DIR)


if __name__ == "__main__":
    main()
