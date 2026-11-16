"""
PyRadiomics feature extraction for the 56 NEW I-SPY2/ACRIN patients, with exactly the SAME settings as the first 50.

Why re-run ISPY1 too: the two datasets have different scanners, voxel sizes and intensity scales, so every
patient must go through exactly the same preprocessing:
  - normalize=True       -> intensities are z-scored per image, so scanner scale does not leak into the features
  - binCount=32          -> a fixed number of grey levels for the texture features (instead of a fixed bin width)
  - resample to 1x1x1 mm -> voxel size differences between patients do not leak into the features

Output: one CSV with one row per patient (107 features + Patient_ID, Dataset, Case_ID + a few QC columns).
"""
import os
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ============================ EDIT THESE ============================
FOLDERS = [
    r"C:\Users\ntech\Desktop\new_dev56\nrrd",   # the 56 new development patients (made by ispy2_to_nrrd_dev56.py)
]
OUT_CSV = r"C:\Users\ntech\Desktop\retrain_126\features_dev56.csv"
LABEL = 255                                             # tumor value in the masks
# ====================================================================

SETTINGS = {
    "label": LABEL,
    "normalize": True,
    "normalizeScale": 100,
    "binCount": 32,
    "resampledPixelSpacing": [1.0, 1.0, 1.0],
    "interpolator": "sitkBSpline",
    "preCrop": True,
    "correctMask": True,
    "padDistance": 5,
}

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
for key, value in SETTINGS.items():
    extractor.settings[key] = value


def dataset_name(patient_id):
    if patient_id.startswith("ISPY1"):
        return "ISPY1"
    if patient_id.startswith("ACRIN"):
        return "ACRIN6698"
    return "ISPY2"


def case_number(patient_id):
    return patient_id.replace("_", "-").split("-")[-1]


jobs = []
for folder in FOLDERS:
    if not os.path.isdir(folder):
        print("Folder not found, skipped:", folder)
        continue
    for f in sorted(os.listdir(folder)):
        if f.endswith("_image.nrrd"):
            pid = f[:-len("_image.nrrd")]
            fixed = os.path.join(folder, pid + "_mask_fixed.nrrd")   # ISPY1: use the "fixed" mask when it exists
            plain = os.path.join(folder, pid + "_mask.nrrd")
            jobs.append((pid, os.path.join(folder, f), fixed if os.path.exists(fixed) else plain))

ids = [j[0] for j in jobs]
dups = sorted({i for i in ids if ids.count(i) > 1})
if dups:
    print("WARNING: same patient found in more than one folder:", dups)
print("Found %d image files" % len(jobs))

rows, failed = [], []
for n, (pid, image_path, mask_path) in enumerate(jobs, 1):
    if not os.path.exists(mask_path):
        print("[%d/%d] %s: mask missing" % (n, len(jobs), pid))
        failed.append((pid, "mask missing"))
        continue
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        # put the mask on the image grid (nearest neighbour), same as the old extract_all.py
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = resampler.Execute(mask)

        marr = sitk.GetArrayFromImage(mask) == LABEL
        iarr = sitk.GetArrayFromImage(image).astype(float)
        if not marr.any():
            raise RuntimeError("mask is empty after resampling onto the image")
        nonzero = iarr[iarr > 0]

        result = extractor.execute(image, mask)
        features = {k: v for k, v in result.items() if not k.startswith("diagnostics")}
        features["Patient_ID"] = pid
        features["Dataset"] = dataset_name(pid)
        features["Case_ID"] = case_number(pid)
        features["QC_tumor_voxels"] = int(marr.sum())
        features["QC_tumor_mean_over_image_median"] = float(iarr[marr].mean() / np.median(nonzero))
        features["QC_spacing"] = str(tuple(round(s, 3) for s in image.GetSpacing()))
        features["QC_mask_file"] = os.path.basename(mask_path)
        rows.append(features)
        print("[%d/%d] %s done" % (n, len(jobs), pid))
    except Exception as e:
        traceback.print_exc()
        print("[%d/%d] %s ERROR: %s" % (n, len(jobs), pid, e))
        failed.append((pid, str(e)))

df = pd.DataFrame(rows)
if df.empty:
    print("\nNo patient was processed.")
else:
    first = ["Patient_ID", "Dataset", "Case_ID"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(OUT_CSV, index=False)
    print("\nDone: %d patients saved to %s" % (len(df), OUT_CSV))
    print(df.groupby("Dataset").size().to_string())

    r = df["QC_tumor_mean_over_image_median"]
    med = r.median()
    odd = df[(r < 0.5 * med) | (r > 2.0 * med)]
    print("\nQC: tumor mean / image median, typical value %.2f" % med)
    if len(odd):
        print("Patients with an unusual value (check their image phase / mask):")
        print(odd[["Patient_ID", "QC_tumor_mean_over_image_median"]].to_string(index=False))
    else:
        print("No unusual patients.")

if failed:
    print("\nFailed patients:", failed)
