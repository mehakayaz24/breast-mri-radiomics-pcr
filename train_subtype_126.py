"""
Separate models for HER2+ and TNBC (as in Hatamikia et al. 2024), on the 126 DEVELOPMENT patients only.
The 12 test patients are NOT used. Same five final models, same validation (repeated stratified 5-fold CV, 10 repeats,
every step inside the folds). EXPLORATORY: only 61 (HER2+) and 65 (TNBC) patients per model, so the AUC is very uncertain.

Put in retrain_126 (next to train_final_models_126.py) and run:  python train_subtype_126.py
"""
import os
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
import train_models as tm
import train_final_models_126 as fm

warnings.filterwarnings("ignore")
OUT = r"C:\Users\ntech\Desktop\retrain_126\results_subtype_126"
os.makedirs(OUT, exist_ok=True)

data = tm.build_dataset()
data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
feat_cols = [c for c in data.columns if c.startswith("original_")]
lines = []
for sub in ["HER2+", "TNBC"]:
    d = data[data["subtype"] == sub].reset_index(drop=True)
    X = d[feat_cols].to_numpy(float)
    y = d["pCR"].to_numpy()
    lines.append("\n=== %s only: n=%d, pCR=%d, non-pCR=%d ===" % (sub, len(y), y.sum(), (1 - y).sum()))
    for name in fm.MODEL_NAMES:
        aucs, oof = tm.repeated_cv(fm.make_pipeline(name), X, y, y, 10)
        lo, hi = tm.bootstrap_ci(y, oof)
        lines.append("%-20s AUC %.3f +/- %.3f   pooled %.3f  95%% CI [%.2f, %.2f]" % (name, aucs.mean(), aucs.std(), roc_auc_score(y, oof), lo, hi))
    vol = d["original_shape_MeshVolume"].to_numpy(float)
    lines.append("%-20s AUC %.3f  (tumor volume only, direction fixed in advance)" % ("Volume only", roc_auc_score(y, -vol)))
    print("\n".join(lines[-(len(fm.MODEL_NAMES) + 2):]))
with open(os.path.join(OUT, "subtype_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved in", OUT)
