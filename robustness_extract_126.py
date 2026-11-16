"""
Segmentation robustness, step 1: extract radiomics features from deliberately perturbed tumor masks.

Objective iii of the project: "assess the robustness of radiomic features under variations in tumor segmentation".
For every I-SPY2/ACRIN patient (the 126 development patients; the 12 test patients are never used) the reference tumor mask is changed in
five realistic ways and the same 107 features are extracted again with EXACTLY the same settings as extract_all_70.py:

  shrink_25pct   mask loses its 25% most peripheral voxels (volume x 0.75)
  grow_25pct     mask grows outwards until its volume is 25% larger (volume x 1.25)
  shift_2mm      whole mask moved 2 mm in x and 2 mm in y (a mis-placed contour)
  random_contour boundary randomly wobbled by about 1.5 mm (smooth random contour noise, fixed random seed)
  solid_filled   gaps closed (3 mm) and holes filled: a compact, "hand-drawn looking" lesion instead of the
                 speckled enhancement mask

(Shrinking and growing is defined by volume, not by a fixed number of millimetres, because the enhancement
masks are speckled with many small gaps: a fixed 1 mm erosion would delete 95% of them.)

Output: one long CSV (one row per patient and perturbation) -> step 2 (robustness_analysis.py) works on it.
The script can be stopped and started again: patients that are already finished are skipped.
"""
import os
import glob
import shutil
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ============================ EDIT THESE ============================
# the 126 development patients live in three folders (the 12 test patients are NOT here, on purpose)
NRRD_DIRS = [
    r"C:\Users\ntech\Desktop\Dataset ISPY2\nrrd",            # the first 50
    r"C:\Users\ntech\Desktop\new_test_patients\nrrd",        # the 20 added later
    r"C:\Users\ntech\Desktop\new_dev56\nrrd",                # the 56 added last
]
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\robustness126"    # results go here (a sub folder per patient, plus the final CSV)
OLD_PER_PATIENT = r"C:\Users\ntech\Desktop\robustness\per_patient"   # finished patients of the first run are re-used (same settings)
LABEL = 255
# ====================================================================

# identical to extract_all_70.py (do not change, otherwise the reference features no longer match)
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

PERTURBATIONS = ["shrink_25pct", "grow_25pct", "shift_2mm", "random_contour", "solid_filled"]
VOLUME_CHANGE = 0.25
CLOSING_MM = 3.0
RANDOM_AMPLITUDE_MM = 1.5
RANDOM_SMOOTH_MM = 3.0
SHIFT_MM = (2.0, 2.0, 0.0)
SEED = 2026


def signed_distance(binary_mask):
    """Distance to the mask border in mm: negative inside the tumor, positive outside."""
    return sitk.SignedMaurerDistanceMap(binary_mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)


def volume_target_mask(dist, n_voxels, seed):
    """Keep exactly n_voxels voxels: the ones with the smallest signed distance (deepest inside first).
    Many voxels have identical distances, so ties are broken with a tiny fixed random jitter."""
    arr = sitk.GetArrayFromImage(dist).astype(np.float64)
    n_voxels = int(max(1, min(n_voxels, arr.size - 1)))
    jitter = np.random.default_rng(seed).uniform(0.0, 1e-4, arr.shape)
    flat = (arr + jitter).ravel()
    keep = np.zeros(flat.size, dtype=np.uint8)
    keep[np.argpartition(flat, n_voxels - 1)[:n_voxels]] = 1
    out = sitk.GetImageFromArray(keep.reshape(arr.shape))
    out.CopyInformation(dist)
    return out


def perturb(binary_mask, kind, seed):
    """binary_mask: uint8 image with 0/1. Returns a perturbed uint8 0/1 image on the same grid."""
    if kind == "shift_2mm":
        tx = sitk.TranslationTransform(3, [-s for s in SHIFT_MM])      # moving the image content by +shift
        return sitk.Resample(binary_mask, binary_mask, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    n_ref = int(sitk.GetArrayFromImage(binary_mask).sum())
    dist = signed_distance(binary_mask)
    if kind == "shrink_25pct":
        out = volume_target_mask(dist, round(n_ref * (1 - VOLUME_CHANGE)), seed)
    elif kind == "grow_25pct":
        out = volume_target_mask(dist, round(n_ref * (1 + VOLUME_CHANGE)), seed)
    elif kind == "random_contour":
        rng = np.random.default_rng(seed)
        noise = sitk.GetImageFromArray(rng.standard_normal(sitk.GetArrayFromImage(binary_mask).shape).astype(np.float32))
        noise.CopyInformation(binary_mask)
        noise = sitk.SmoothingRecursiveGaussian(noise, RANDOM_SMOOTH_MM)     # sigma in mm
        arr = sitk.GetArrayFromImage(noise)
        arr = arr / (arr.std() + 1e-12) * RANDOM_AMPLITUDE_MM               # unit variance, then scaled to mm
        wob = sitk.GetImageFromArray(arr.astype(np.float32))
        wob.CopyInformation(binary_mask)
        out = (dist + wob) <= 0.0
    elif kind == "solid_filled":
        grown = sitk.Cast(dist <= CLOSING_MM, sitk.sitkUInt8)               # closing = grow, then shrink back
        closed = signed_distance(grown) <= -CLOSING_MM
        closed = sitk.Cast(closed, sitk.sitkUInt8) | binary_mask            # never lose original voxels
        out = sitk.BinaryFillhole(closed)
    else:
        raise ValueError(kind)
    return sitk.Cast(out, sitk.sitkUInt8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_dir = os.path.join(OUT_DIR, "per_patient")
    os.makedirs(patient_dir, exist_ok=True)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    for k, v in SETTINGS.items():
        extractor.settings[k] = v

    images = []
    for d in NRRD_DIRS:
        images += sorted(glob.glob(os.path.join(d, "*_image.nrrd")))
    print("Patients found: %d (each gets %d perturbations)" % (len(images), len(PERTURBATIONS)))
    failed = []

    for n, img_path in enumerate(images, 1):
        pid = os.path.basename(img_path)[:-len("_image.nrrd")]
        out_csv = os.path.join(patient_dir, pid + ".csv")
        if os.path.exists(out_csv):
            print("[%d/%d] %s already done, skipped" % (n, len(images), pid))
            continue
        old_csv = os.path.join(OLD_PER_PATIENT, pid + ".csv")
        if os.path.exists(old_csv):                      # already extracted in the first run with identical settings
            shutil.copy(old_csv, out_csv)
            print("[%d/%d] %s re-used from the first run" % (n, len(images), pid))
            continue
        mask_path = os.path.join(os.path.dirname(img_path), pid + "_mask.nrrd")
        if not os.path.exists(mask_path):
            print("[%d/%d] %s: mask missing" % (n, len(images), pid))
            failed.append((pid, "mask missing"))
            continue
        try:
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)
            binary = sitk.Cast(mask == LABEL, sitk.sitkUInt8)
            ref_voxels = int(sitk.GetArrayFromImage(binary).sum())
            rows = []
            for j, kind in enumerate(PERTURBATIONS):
                pm = perturb(binary, kind, SEED + 1000 * n + j)
                voxels = int(sitk.GetArrayFromImage(pm).sum())
                if voxels < 20:
                    print("   %s %s: mask became (almost) empty, skipped" % (pid, kind))
                    continue
                pmask = sitk.Cast(pm, sitk.sitkUInt8) * LABEL
                pmask.CopyInformation(image)
                result = extractor.execute(image, pmask)
                row = {k: v for k, v in result.items() if k.startswith("original_")}
                row.update({"Patient_ID": pid, "perturbation": kind, "mask_voxels": voxels,
                            "reference_mask_voxels": ref_voxels})
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print("[%d/%d] %s done (%d perturbations)" % (n, len(images), pid, len(rows)))
        except Exception as e:
            traceback.print_exc()
            print("[%d/%d] %s ERROR: %s" % (n, len(images), pid, e))
            failed.append((pid, str(e)))

    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(patient_dir, "*.csv")))]
    if parts:
        allf = pd.concat(parts, ignore_index=True)
        final = os.path.join(OUT_DIR, "robustness_features.csv")
        allf.to_csv(final, index=False)
        print("\nSaved %d rows (%d patients) to %s" % (len(allf), allf["Patient_ID"].nunique(), final))
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
