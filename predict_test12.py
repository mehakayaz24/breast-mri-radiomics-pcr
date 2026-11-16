"""
FINAL TEST of the models trained on 70 patients (final_models_126.joblib) on the 12 untouched patients. RUN ONLY ONCE, using their own tumor masks.
Put this file in the 'prototype' folder (next to app_mri.py, autoseg.py, dicom_loader.py, final_models.joblib).

Run (cmd):
  cd /d C:\\Users\\ntech\\Desktop\\prototype
  python predict_new_patients.py

It uses exactly the same code as the app (analyse() in app_mri.py): same features, same five models, same majority vote.
Output: predictions_new_patients.csv and a printed summary (accuracy, confusion matrix, AUC).
The true outcomes (test_labels.csv) are only read at the end, to compare. They are never used for the prediction.
"""
import os
import sys
import glob
import logging
import warnings

NRRD_DIR = r"C:\Users\ntech\Desktop\new_test12\nrrd"
LABELS = r"C:\Users\ntech\Desktop\new_test12\test12_labels.csv"
OUT_CSV = r"C:\Users\ntech\Desktop\new_test12\predictions_test12.csv"

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
import app_mri as A

MODELS_126 = r"C:\Users\ntech\Desktop\retrain_126\final_models_126.joblib"   # trained on the 126 development patients only
bundle = A.load_bundle(MODELS_126)
params = A.load_params()
pre_files = sorted(glob.glob(os.path.join(NRRD_DIR, "*_pre.nrrd")))
print("Patients found:", len(pre_files))
rows = []
for i, pf in enumerate(pre_files, 1):
    pid = os.path.basename(pf)[:-len("_pre.nrrd")]
    try:
        pre = sitk.ReadImage(pf)
        post = sitk.ReadImage(os.path.join(NRRD_DIR, pid + "_image.nrrd"))
        mask = sitk.ReadImage(os.path.join(NRRD_DIR, pid + "_mask.nrrd"))
        r = A.analyse(pre, post, mask, bundle, params)
        cls, votes, n, mean = A.predicted_class(r["probs"])
        row = {"PatientID": pid, "Predicted_class": cls, "Votes_for_pCR": "%d of %d" % (votes, n), "Mean_score": round(mean, 3),
               "Tumor_volume_mL": round(r["info"]["tumor_volume_ml"], 1),
               "Outside_training_range": "; ".join(r["checks"].loc[r["checks"]["Status"] != "within range", "Property"])}
        row.update({"Score_" + k: round(v, 3) for k, v in r["probs"].items()})
        rows.append(row)
        print("[%d/%d] %s -> class %d (%d of %d models, mean score %.2f)" % (i, len(pre_files), pid, cls, votes, n, mean))
    except Exception as e:
        print("[%d/%d] %s -> FAILED: %s" % (i, len(pre_files), pid, e))
        rows.append({"PatientID": pid, "Predicted_class": np.nan, "Error": str(e)})

df = pd.DataFrame(rows)
lab = pd.read_csv(LABELS)[["PatientID", "subtype", "pCR"]].rename(columns={"pCR": "True_class"})
df = df.merge(lab, on="PatientID", how="left")
df["Correct"] = np.where(df["Predicted_class"].isna(), "", np.where(df["Predicted_class"] == df["True_class"], "yes", "no"))
df.to_csv(OUT_CSV, index=False)

d = df.dropna(subset=["Predicted_class", "True_class"])
if len(d):
    y, p = d["True_class"].astype(int), d["Predicted_class"].astype(int)
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    print("\n===== SUMMARY (%d patients) =====" % len(d))
    print("Accuracy: %d/%d = %.0f%%" % (tp + tn, len(d), 100.0 * (tp + tn) / len(d)))
    print("Confusion matrix: TP %d  TN %d  FP %d  FN %d" % (tp, tn, fp, fn))
    if y.nunique() == 2:
        print("AUC of the mean model score: %.3f" % roc_auc_score(y, d["Mean_score"]))
        for c in [c for c in d.columns if c.startswith("Score_")]:
            print("   AUC %s: %.3f" % (c[6:], roc_auc_score(y, d[c])))
print("\nSaved:", OUT_CSV)
