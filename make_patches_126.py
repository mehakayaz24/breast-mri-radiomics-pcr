"""
CNN preparation (run on the laptop): cut small tumor-centred image patches from the baseline MRI.

For each of the 126 development I-SPY2/ACRIN patients (image + mask NRRD made by ispy2_to_nrrd.py):
  1. put the volume in a standard orientation (LPS, so slices are axial) and resample to 1 x 1 x 1 mm
  2. find the tumor centre and the axial slice with the largest tumor area (the mask is used ONLY to centre the crop)
  3. cut a 64 x 64 mm window around the tumor centre, in 5 axial positions (-6, -3, 0, +3, +6 mm from that slice)
     Every sample has 3 channels = three neighbouring slices (3 mm apart), the "2.5D" input of a normal 2D CNN
  4. scale the grey values of each patient to 0..1 (1st to 99.5th percentile of the breast voxels)

Output: cnn_patches.npz (a few MB) -> upload it to Google Colab together with cnn_colab.py
        patch_preview.png -> look at it: every row is one patient, tumor should be in the middle.
"""
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ============================ EDIT THESE ============================
NRRD_DIRS = [
    r"C:\Users\ntech\Desktop\Dataset ISPY2\nrrd",            # the first 50
    r"C:\Users\ntech\Desktop\new_test_patients\nrrd",        # the 20 added later
    r"C:\Users\ntech\Desktop\new_dev56\nrrd",                # the 56 added last (the 12 test patients are NOT included)
]
LABELS_CSV = r"C:\Users\ntech\Desktop\retrain_126\results_126\merged_data_all.csv"     # made by train_models.py (has Patient_ID and pCR)
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\cnn126"
LABEL = 255
# ====================================================================

PATCH_MM = 64
OFFSETS_MM = (-6, -3, 0, 3, 6)
CHANNEL_STEP_MM = 3


def to_lps_isotropic(image, interpolator):
    image = sitk.DICOMOrient(image, "LPS")
    spacing = image.GetSpacing()
    size = [int(round(sz * sp)) for sz, sp in zip(image.GetSize(), spacing)]      # 1 mm voxels
    return sitk.Resample(image, size, sitk.Transform(), interpolator, image.GetOrigin(), (1.0, 1.0, 1.0),
                         image.GetDirection(), 0, image.GetPixelID())


def crop_window(slice_2d, cy, cx, half):
    out = np.zeros((2 * half, 2 * half), dtype=np.float32)
    y0, x0 = cy - half, cx - half
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + 2 * half, slice_2d.shape[0]), min(x0 + 2 * half, slice_2d.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = slice_2d[ys:ye, xs:xe]
    return out


def make_patient_patches(image_path, mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    r = sitk.ResampleImageFilter()                                  # mask onto the image grid (nearest neighbour)
    r.SetReferenceImage(image)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    mask = r.Execute(mask)
    mask = sitk.Cast(mask == LABEL, sitk.sitkUInt8)

    img = sitk.GetArrayFromImage(to_lps_isotropic(sitk.Cast(image, sitk.sitkFloat32), sitk.sitkLinear))   # (z,y,x)
    msk = sitk.GetArrayFromImage(to_lps_isotropic(mask, sitk.sitkNearestNeighbor)) > 0
    if not msk.any():
        raise RuntimeError("mask is empty after resampling")

    breast = img[img > 0]
    lo, hi = np.percentile(breast, 1), np.percentile(breast, 99.5)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    idx = np.argwhere(msk)
    cy, cx = [int(round(v)) for v in idx.mean(axis=0)[1:]]
    zc = int(np.argmax(msk.sum(axis=(1, 2))))
    half = PATCH_MM // 2
    nz = img.shape[0]
    patches = []
    for off in OFFSETS_MM:
        chans = []
        for c in (-CHANNEL_STEP_MM, 0, CHANNEL_STEP_MM):
            z = int(min(max(zc + off + c, 0), nz - 1))
            chans.append(crop_window(img[z], cy, cx, half))
        patches.append(np.stack(chans))
    return np.stack(patches)                                        # (5, 3, 64, 64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels["cohort"] == "ISPY2"].set_index("Patient_ID")["pCR"]

    patches, ids, ys = [], [], []
    all_paths = []
    for d in NRRD_DIRS:
        all_paths += sorted(glob.glob(os.path.join(d, "*_image.nrrd")))
    for path in all_paths:
        pid = os.path.basename(path)[:-len("_image.nrrd")]
        if pid not in labels.index:
            continue
        try:
            p = make_patient_patches(path, os.path.join(os.path.dirname(path), pid + "_mask.nrrd"))
        except Exception as e:
            print("%s ERROR: %s" % (pid, e))
            continue
        patches.append(p)
        ids.append(pid)
        ys.append(int(labels[pid]))
        print("%s done" % pid)

    patches = np.stack(patches).astype(np.float16)                  # (patients, 5, 3, 64, 64)
    np.savez_compressed(os.path.join(OUT_DIR, "cnn_patches.npz"), patches=patches, patient_ids=np.array(ids),
                        labels=np.array(ys), offsets_mm=np.array(OFFSETS_MM))
    print("\nSaved %d patients, array shape %s, pCR=1: %d" % (len(ids), patches.shape, sum(ys)))

    # quick visual check: centre slice of the middle position, one row per patient
    tiles = [(patches[i, 2, 1].astype(np.float32) * 255).astype(np.uint8) for i in range(len(ids))]
    ncol = 10
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + ncol]) for r in range(0, len(tiles), ncol)]
    sitk.WriteImage(sitk.GetImageFromArray(np.vstack(rows)), os.path.join(OUT_DIR, "patch_preview.png"))
    print("Preview picture saved: patch_preview.png (tumor should sit in the middle of each tile)")


if __name__ == "__main__":
    main()
