"""
Does removing the features that are NOT stable against tumor-mask noise help?

The stability of every feature was measured earlier WITHOUT using the pCR labels (robustness analysis: mask shrunk,
grown, shifted 2 mm, wobbled, filled; ICC over the six mask versions, 50 patients). Here only the stable features are kept
and the SAME five final models are cross-validated again on the 126 development patients (same folds, same settings).

Settings compared: all 107 features | ICC >= 0.75 (71 features) | ICC >= 0.90 (31 features).
This is an extra analysis on the development patients only. The 12 test patients are NOT used here and must not be
used again (they have already been used once).

Put this file in retrain_126 (next to train_final_models_126.py) and run:  python train_robust_126.py
"""
import os
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
import train_models as tm
import train_final_models_126 as fm

warnings.filterwarnings("ignore")
OUT = r"C:\Users\ntech\Desktop\retrain_126\results_robust_126"
os.makedirs(OUT, exist_ok=True)

ROBUST_075 = ['original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_VoxelVolume', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_Contrast', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_SumAverage', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Strength']
ROBUST_090 = ['original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_SurfaceArea', 'original_shape_VoxelVolume', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Maximum', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Range', 'original_firstorder_RootMeanSquared', 'original_firstorder_TotalEnergy', 'original_gldm_DependenceNonUniformity', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_ngtdm_Busyness']

data = tm.build_dataset()
data = data[data["cohort"] == "ISPY2"].reset_index(drop=True)
all_cols = [c for c in data.columns if c.startswith("original_")]
y = data["pCR"].to_numpy()
settings = {"all 107 features": all_cols,
            "ICC>=0.75 (%d)" % len(ROBUST_075): [c for c in all_cols if c in set(ROBUST_075)],
            "ICC>=0.90 (%d)" % len(ROBUST_090): [c for c in all_cols if c in set(ROBUST_090)]}
lines = []
for label, cols in settings.items():
    X = data[cols].to_numpy(float)
    lines.append("\n=== %s ===" % label)
    for name in fm.MODEL_NAMES:
        aucs, oof = tm.repeated_cv(fm.make_pipeline(name), X, y, y, 10)
        lo, hi = tm.bootstrap_ci(y, oof)
        lines.append("%-20s AUC %.3f +/- %.3f   pooled %.3f  95%% CI [%.2f, %.2f]" % (name, aucs.mean(), aucs.std(), roc_auc_score(y, oof), lo, hi))
    print("\n".join(lines[-(len(fm.MODEL_NAMES) + 1):]))
with open(os.path.join(OUT, "robust_feature_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved in", OUT)
