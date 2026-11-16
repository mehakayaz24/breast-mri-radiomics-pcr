"""
Tries better automatic segmentation rules on all 50 patients and compares them (Dice against the reference mask).
Put in the prototype folder next to autoseg.py and run:   python autoseg_tune.py        (about 20-40 minutes, prints progress)
Results: C:\\Users\\ntech\\Desktop\\autoseg\\autoseg_tune_report.txt and autoseg_tune_per_patient.csv  -> send the report.

What the diagnosis showed: with the simple rule (PE >= 70%) the "tumor" is the whole enhancing breast (100-500 mL against
2-70 mL in the reference), because the reference mask is limited to a box (VOI) drawn by a radiologist around the tumor.
Rules tried here:
  open     : PE >= threshold, then a morphological opening removes thin enhancing structures (vessels, glandular
             speckle); the component with the largest total enhancement is kept
  hotspot  : the centre of the densest enhancing region is found (smoothed map) and only voxels within a sphere of
             radius R around it are kept
  box_upper_bound : (not automatic!) the same PE rule inside a box around the reference tumor. It shows what is possible if
             the user marks a rough region, as the I-SPY radiologists did.
"""
import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
import autoseg

NRRD_DIR = r"C:\Users\ntech\Desktop\Dataset ISPY2\nrrd"
OUT_DIR = r"C:\Users\ntech\Desktop\autoseg"
LABEL = 255
BOX_MARGIN_MM = 5.0

CONFIGS = [
    ("open_bg15_pe70_o2",   dict(method="open", bg=0.15, pe=70.0,  open_mm=2.0)),
    ("open_bg35_pe70_o2",   dict(method="open", bg=0.35, pe=70.0,  open_mm=2.0)),
    ("open_bg15_pe120_o2",  dict(method="open", bg=0.15, pe=120.0, open_mm=2.0)),
    ("open_bg35_pe120_o2",  dict(method="open", bg=0.35, pe=120.0, open_mm=2.0)),
    ("hotspot_pe70_R25",    dict(method="hotspot", bg=0.25, pe=70.0,  radius_mm=25.0)),
    ("hotspot_pe70_R35",    dict(method="hotspot", bg=0.25, pe=70.0,  radius_mm=35.0)),
    ("hotspot_pe120_R25",   dict(method="hotspot", bg=0.25, pe=120.0, radius_mm=25.0)),
    ("hotspot_pe120_R35",   dict(method="hotspot", bg=0.25, pe=120.0, radius_mm=35.0)),
]


def pe_map(pre, post, bg_fraction):
    a_pre = sitk.GetArrayFromImage(sitk.Cast(pre, sitk.sitkFloat32)).astype(np.float32)
    a_post = sitk.GetArrayFromImage(sitk.Cast(post, sitk.sitkFloat32)).astype(np.float32)
    background = bg_fraction * np.percentile(a_pre[a_pre > 0], 99.5)
    valid = a_pre > background
    pe = np.zeros_like(a_pre)
    pe[valid] = (a_post[valid] - a_pre[valid]) / a_pre[valid] * 100.0
    return pe


def segment(pe, spacing, cfg):
    """pe: array (z,y,x); spacing (x,y,z) in mm. Returns uint8 array."""
    cand = pe >= cfg["pe"]
    if not cand.any():
        return np.zeros(pe.shape, np.uint8)
    if cfg["method"] == "open":
        img = sitk.GetImageFromArray(cand.astype(np.uint8))
        img.SetSpacing(spacing)
        r = [max(1, int(round(cfg["open_mm"] / s))) for s in spacing]
        opened = sitk.BinaryMorphologicalOpening(img, r)
        lab = sitk.GetArrayFromImage(sitk.ConnectedComponent(opened))
        if lab.max() == 0:
            return np.zeros(pe.shape, np.uint8)
        score = np.bincount(lab.ravel(), weights=np.clip(pe, 0, 300).ravel().astype(np.float64))
        score[0] = 0
        region = sitk.GetImageFromArray((lab == int(score.argmax())).astype(np.uint8))
        region.SetSpacing(spacing)
        dist = sitk.SignedMaurerDistanceMap(region, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        return (cand & (sitk.GetArrayFromImage(dist) <= 3.0)).astype(np.uint8)
    if cfg["method"] == "hotspot":
        sig = [6.0 / spacing[2], 6.0 / spacing[1], 6.0 / spacing[0]]            # 6 mm, array order z,y,x
        dens = ndi.gaussian_filter(cand.astype(np.float32), sigma=sig)
        cz, cy, cx = np.unravel_index(int(dens.argmax()), dens.shape)
        nz, ny, nx = pe.shape
        dz = (np.arange(nz) - cz) * spacing[2]; dy = (np.arange(ny) - cy) * spacing[1]; dx = (np.arange(nx) - cx) * spacing[0]
        r2 = dz[:, None, None] ** 2 + dy[None, :, None] ** 2 + dx[None, None, :] ** 2
        return (cand & (r2 <= cfg["radius_mm"] ** 2)).astype(np.uint8)
    raise ValueError(cfg["method"])


def box_upper_bound(pe, ref, spacing, pe_thr=70.0):
    zz, yy, xx = np.nonzero(ref)
    m = [int(round(BOX_MARGIN_MM / s)) for s in (spacing[2], spacing[1], spacing[0])]
    box = np.zeros(ref.shape, bool)
    box[max(zz.min() - m[0], 0):zz.max() + m[0] + 1, max(yy.min() - m[1], 0):yy.max() + m[1] + 1,
        max(xx.min() - m[2], 0):xx.max() + m[2] + 1] = True
    return ((pe >= pe_thr) & box).astype(np.uint8)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * (a.astype(bool) & b.astype(bool)).sum() / s) if s else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = sorted(f[:-len("_image.nrrd")] for f in os.listdir(NRRD_DIR) if f.endswith("_image.nrrd") and not f.startswith("ISPY1"))
    ids = [i for i in ids if all(os.path.exists(os.path.join(NRRD_DIR, i + s)) for s in ("_pre.nrrd", "_mask.nrrd"))]
    print("Patients: %d, rules tried: %d (+ the box upper bound)" % (len(ids), len(CONFIGS)))
    rows, t0 = [], time.time()
    for n, pid in enumerate(ids, 1):
        pre = sitk.ReadImage(os.path.join(NRRD_DIR, pid + "_pre.nrrd"))
        post = sitk.ReadImage(os.path.join(NRRD_DIR, pid + "_image.nrrd"))
        rs = sitk.ResampleImageFilter(); rs.SetReferenceImage(post); rs.SetInterpolator(sitk.sitkNearestNeighbor)
        ref = (sitk.GetArrayFromImage(rs.Execute(sitk.ReadImage(os.path.join(NRRD_DIR, pid + "_mask.nrrd")))) == LABEL)
        sp = post.GetSpacing()
        a_pre = sitk.GetArrayFromImage(pre).astype(float); a_post = sitk.GetArrayFromImage(post).astype(float)
        row = {"Patient_ID": pid, "reference_ml": ref.sum() * sp[0] * sp[1] * sp[2] / 1000.0,
               "tumor_post_over_pre": float(np.median(a_post[ref]) / max(np.median(a_pre[ref]), 1e-6))}
        cache = {}
        for name, cfg in CONFIGS:
            try:
                if cfg["bg"] not in cache:
                    cache = {cfg["bg"]: pe_map(pre, post, cfg["bg"])}
                m = segment(cache[cfg["bg"]], sp, cfg)
                row[name] = dice(m, ref)
                row[name + "_ml"] = m.sum() * sp[0] * sp[1] * sp[2] / 1000.0
            except Exception as e:
                row[name] = 0.0; row[name + "_ml"] = np.nan
                print("   %s %s failed: %s" % (pid, name, e))
        row["box_upper_bound"] = dice(box_upper_bound(pe_map(pre, post, 0.15), ref, sp), ref)
        rows.append(row)
        print("[%d/%d] %s  best rule Dice %.2f | box %.2f | post/pre in tumor %.2f | %.0f min" % (
            n, len(ids), pid, max(row[c] for c, _ in CONFIGS), row["box_upper_bound"], row["tumor_post_over_pre"], (time.time() - t0) / 60))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "autoseg_tune_per_patient.csv"), index=False)
    lines = ["=== Dice against the reference mask, 50 patients ===", "rule                         mean   median  >0.5 in n patients   median auto volume (mL)"]
    for name, _ in CONFIGS + [("box_upper_bound", None)]:
        v = df[name]
        vol = df[name + "_ml"].median() if name + "_ml" in df else float("nan")
        lines.append("%-28s %.3f  %.3f   %2d                   %.1f" % (name, v.mean(), v.median(), (v > 0.5).sum(), vol))
    lines.append("\nMedian reference volume %.1f mL" % df["reference_ml"].median())
    odd = df[df["tumor_post_over_pre"] < 1.3]
    lines.append("Patients whose reference tumor does NOT enhance (post/pre < 1.3): %d %s" % (len(odd), list(odd["Patient_ID"])))
    print("\n" + "\n".join(lines))
    with open(os.path.join(OUT_DIR, "autoseg_tune_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
