"""
Segmentation robustness, step 2: how stable are the radiomics features and the predictions when the tumor
segmentation changes?  (Objective iii and research question ii of the project.)

Input : robustness_features.csv (made by robustness_extract.py) and merged_data_all.csv (reference features,
        made by train_models.py), I-SPY2/ACRIN patients only.
Part A: feature robustness. ICC(2,1) (two-way random, absolute agreement) per feature
          - overall: reference + the 5 perturbed masks as 6 "raters"
          - per perturbation: reference vs that perturbed mask
        ICC >= 0.90 excellent, 0.75 to 0.90 good, below 0.75 poor. Summaries per feature class.
Part B: effect on the predictive models. Train on the reference masks (training patients), then predict the
        held-out patients from their reference features AND from each perturbed mask: AUC and prediction stability.
Part C: model that uses only the robust features (overall ICC >= 0.75) versus the model with all features.

Put this file in the same folder as train_models.py.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import train_models as tm

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
REFERENCE_CSV = r"C:\Users\ntech\Desktop\results\merged_data_all.csv"
ROBUSTNESS_CSV = r"C:\Users\ntech\Desktop\robustness\robustness_features.csv"
OUT_DIR = r"C:\Users\ntech\Desktop\results_robustness"
# ====================================================================

PERTURBATIONS = ["shrink_25pct", "grow_25pct", "shift_2mm", "random_contour", "solid_filled"]
SEED = 42
N_REPEATS = 5
ICC_GOOD, ICC_EXCELLENT = 0.75, 0.90

os.makedirs(OUT_DIR, exist_ok=True)
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def icc_2_1(M):
    """ICC(2,1), absolute agreement. M: subjects x raters."""
    n, k = M.shape
    grand = M.mean()
    ms_r = k * ((M.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((M.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else np.nan


def feature_class(name):
    return name.split("_")[1]


def grade(v):
    return "excellent" if v >= ICC_EXCELLENT else ("good" if v >= ICC_GOOD else "poor")


def main():
    ref = pd.read_csv(REFERENCE_CSV)
    ref = ref[ref["cohort"] == "ISPY2"].sort_values("Patient_ID").reset_index(drop=True)
    rob = pd.read_csv(ROBUSTNESS_CSV)
    feats = [c for c in ref.columns if c.startswith("original_")]
    ids = ref["Patient_ID"].to_numpy()
    y = ref["pCR"].to_numpy()

    versions = {"reference": ref[feats].to_numpy(float)}
    for p in PERTURBATIONS:
        sub = rob[rob["perturbation"] == p].set_index("Patient_ID").reindex(ids)
        versions[p] = sub[feats].to_numpy(float)
    ok = ~np.isnan(np.stack(list(versions.values()))).any(axis=(0, 2))
    say("Patients with all 6 versions: %d of %d, features: %d" % (ok.sum(), len(ids), len(feats)))
    ids, y = ids[ok], y[ok]
    versions = {k: v[ok] for k, v in versions.items()}

    # ------------------------------------------------------------------ A
    say("\n=== A) FEATURE ROBUSTNESS (ICC 2,1) ===")
    rows = []
    for j, f in enumerate(feats):
        stack = np.column_stack([versions[v][:, j] for v in versions])
        row = {"feature": f, "class": feature_class(f), "ICC_overall": icc_2_1(stack)}
        for p in PERTURBATIONS:
            row["ICC_" + p] = icc_2_1(np.column_stack([versions["reference"][:, j], versions[p][:, j]]))
        rows.append(row)
    icc = pd.DataFrame(rows).fillna(-1.0)       # constant features give no ICC; treat as not robust
    icc["grade"] = icc["ICC_overall"].apply(grade)
    icc.to_csv(os.path.join(OUT_DIR, "icc_per_feature.csv"), index=False)

    n = len(icc)
    say("Overall (all 6 mask versions): excellent (ICC>=0.90): %d (%.0f%%), good (0.75-0.90): %d (%.0f%%), poor (<0.75): %d (%.0f%%)" % (
        (icc.grade == "excellent").sum(), 100 * (icc.grade == "excellent").mean(),
        (icc.grade == "good").sum(), 100 * (icc.grade == "good").mean(),
        (icc.grade == "poor").sum(), 100 * (icc.grade == "poor").mean()))
    say("\nBy feature class (overall ICC):")
    cls = icc.groupby("class").agg(n_features=("feature", "size"), median_ICC=("ICC_overall", "median"),
                                   pct_ICC_ge_075=("ICC_overall", lambda s: 100 * (s >= ICC_GOOD).mean()),
                                   pct_ICC_ge_090=("ICC_overall", lambda s: 100 * (s >= ICC_EXCELLENT).mean()))
    say(cls.round(2).to_string())
    cls.to_csv(os.path.join(OUT_DIR, "icc_by_feature_class.csv"))

    say("\nBy perturbation (share of the 107 features with ICC >= 0.75 against the reference mask):")
    per = {p: 100 * (icc["ICC_" + p] >= ICC_GOOD).mean() for p in PERTURBATIONS}
    for p, v in per.items():
        say("  %-15s %.0f%%" % (p, v))

    order = sorted(icc["class"].unique())
    plt.figure(figsize=(8, 4.5))
    plt.boxplot([icc[icc["class"] == c]["ICC_overall"].clip(lower=-0.2) for c in order], labels=order)
    plt.axhline(ICC_GOOD, color="orange", linestyle="--", linewidth=1)
    plt.axhline(ICC_EXCELLENT, color="green", linestyle="--", linewidth=1)
    plt.ylabel("ICC (2,1), reference + 5 perturbed masks"); plt.title("Feature robustness by feature class")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "icc_by_class.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(list(per.keys()), list(per.values()), color="#4c78a8")
    plt.ylabel("% of features with ICC >= 0.75"); plt.xticks(rotation=20, ha="right")
    plt.title("Feature robustness per segmentation change"); plt.ylim(0, 100)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "robust_features_per_perturbation.png"), dpi=200); plt.close()

    # ------------------------------------------------------------------ B
    say("\n=== B) EFFECT ON THE PREDICTIVE MODELS ===")
    say("Models are trained on reference masks of the training patients only. Held-out patients are predicted from their")
    say("reference features and from each perturbed mask. Repeated stratified 5-fold CV (%d repeats), patient level." % N_REPEATS)
    names = list(versions.keys())
    res_b = []
    for model_name in ["RandomForest", "XGBoost"]:
        pipe = tm.make_pipeline(model_name, 20)
        oof = {v: np.zeros((N_REPEATS, len(y))) for v in names}
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
        for k, (tr, te) in enumerate(cv.split(versions["reference"], y)):
            fitted = clone(pipe).fit(versions["reference"][tr], y[tr])
            for v in names:
                oof[v][k // 5, te] = fitted.predict_proba(versions[v][te])[:, 1]
        say("\n%s (20 features)" % model_name)
        auc_ref = np.mean([roc_auc_score(y, oof["reference"][r]) for r in range(N_REPEATS)])
        say("  %-15s AUC %.3f" % ("reference mask", auc_ref))
        res_b.append({"model": model_name, "version": "reference", "AUC": auc_ref, "spearman_with_reference": 1.0})
        for p in PERTURBATIONS:
            a = np.mean([roc_auc_score(y, oof[p][r]) for r in range(N_REPEATS)])
            rho = np.mean([spearmanr(oof["reference"][r], oof[p][r])[0] for r in range(N_REPEATS)])
            say("  %-15s AUC %.3f (change %+.3f)   rank agreement of the patient scores with the reference: %.2f" % (p, a, a - auc_ref, rho))
            res_b.append({"model": model_name, "version": p, "AUC": a, "spearman_with_reference": rho})
    res_b = pd.DataFrame(res_b)
    res_b.to_csv(os.path.join(OUT_DIR, "model_performance_under_perturbation.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.38
    xs = np.arange(len(names))
    for i, m in enumerate(["RandomForest", "XGBoost"]):
        sub = res_b[res_b["model"] == m].set_index("version").reindex(names)
        ax.bar(xs + (i - 0.5) * w, sub["AUC"], w, label=m)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right"); ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("AUC (model trained on reference masks)"); ax.legend()
    ax.set_title("Prediction performance when the test segmentation changes")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "auc_under_perturbation.png"), dpi=200); plt.close()

    # ------------------------------------------------------------------ C
    say("\n=== C) ONLY ROBUST FEATURES (overall ICC >= %.2f) versus ALL FEATURES ===" % ICC_GOOD)
    robust = icc[icc["ICC_overall"] >= ICC_GOOD]["feature"].tolist()
    say("Robust features: %d of %d" % (len(robust), len(feats)))
    cols = [feats.index(f) for f in robust]
    Xall, Xrob = versions["reference"], versions["reference"][:, cols]
    for model_name in ["RandomForest", "XGBoost"]:
        for k in [10, 20]:
            a_all, _ = tm.repeated_cv(tm.make_pipeline(model_name, k), Xall, y, y, 10)
            a_rob, _ = tm.repeated_cv(tm.make_pipeline(model_name, min(k, len(robust))), Xrob, y, y, 10)
            say("  %-13s k=%-2d  all features AUC %.3f | robust features only AUC %.3f" % (model_name, k, a_all.mean(), a_rob.mean()))

    with open(os.path.join(OUT_DIR, "robustness_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nSaved in", OUT_DIR)


if __name__ == "__main__":
    main()
