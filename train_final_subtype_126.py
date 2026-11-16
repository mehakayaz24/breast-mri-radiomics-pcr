"""
Final models for the prototype, trained SEPARATELY for HER2+ and TNBC (126 development patients only; the 12 test
patients are not used). Same five models and the same settings as train_final_models_126.py.
Output (in retrain_126):  final_models_her2.joblib  and  final_models_tnbc.joblib
Put this file next to train_final_models_126.py and run:  python train_final_subtype_126.py
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
import train_models as tm
import train_final_models_126 as fm

warnings.filterwarnings("ignore")
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126"
FILES = {"HER2+": "final_models_her2.joblib", "TNBC": "final_models_tnbc.joblib"}

data = tm.build_dataset()
data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
feat_cols = [c for c in data.columns if c.startswith("original_")]
for sub, fname in FILES.items():
    d = data[data["subtype"] == sub].reset_index(drop=True)
    X = d[feat_cols].to_numpy(float)
    y = d["pCR"].to_numpy()
    print("\n%s: %d patients (pCR=%d)" % (sub, len(y), y.sum()))
    models, cv_auc, sel = {}, {}, {}
    for name in fm.MODEL_NAMES:
        pipe = fm.make_pipeline(name)
        aucs, _ = tm.repeated_cv(pipe, X, y, y, 10)
        cv_auc[name] = float(aucs.mean())
        models[name] = clone(pipe).fit(X, y)
        sel[name] = fm.selected(models[name], feat_cols)
        print("  %-20s cross-validated AUC %.3f" % (name, cv_auc[name]))
    stats = pd.DataFrame({"mean": d[feat_cols].mean(), "std": d[feat_cols].std(),
                          "min": d[feat_cols].min(), "max": d[feat_cols].max()})
    bundle = {"models": models, "cv_auc": cv_auc, "feature_names": feat_cols, "selected_features": sel,
              "feature_stats": stats, "n_patients": len(y), "subtype": sub,
              "trained_on": "I-SPY2/ACRIN-6698, %s only, baseline MRI" % sub}
    joblib.dump(bundle, os.path.join(OUT_DIR, fname))
    print("  saved:", os.path.join(OUT_DIR, fname))
