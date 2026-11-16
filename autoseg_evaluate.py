"""
How good is the automatic segmentation (autoseg.py) compared with the reference FTV masks, and what does it do to
the predictions?   (Part of objective iii / research question ii: segmentation variability.)

Needs, for each of the 50 I-SPY2/ACRIN patients in NRRD_DIR:
   <id>_pre.nrrd   pre-contrast phase   (made by the updated ispy2_to_nrrd.py, SAVE_PRE = True)
   <id>_image.nrrd first post-contrast phase
   <id>_mask.nrrd  reference tumor mask (value 255)

Step 1  calibrate the one free parameter (background level) on a small grid, report Dice against the reference
        mask, save the automatic masks and autoseg_params.json (used by the app).
Step 2  extract the 107 radiomics features from the automatic masks (same settings as everywhere else).
Step 3  model performance: train on reference-mask features (training patients), predict the held-out patients from
        (a) their reference features and (b) their automatic-mask features. Also train AND test on automatic masks.

Note: the background level is tuned on the same 50 patients, so Dice is slightly optimistic. Say so in the thesis.
Put this file in the same folder as train_models.py and autoseg.py.
"""
import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import autoseg
import train_models as tm

warnings.filterwarnings("ignore")

# ============================ EDIT THESE ============================
NRRD_DIR = r"C:\Users\ntech\Desktop\Dataset ISPY2\nrrd"
REFERENCE_CSV = r"C:\Users\ntech\Desktop\results\merged_data_all.csv"
OUT_DIR = r"C:\Users\ntech\Desktop\autoseg"
RUN_EXTRACTION = True            # steps 2 and 3 need PyRadiomics; set False to do only the Dice part
# ====================================================================

LABEL = 255
BG_GRID = [0.25, 0.35, 0.45]
SETTINGS = {                      # identical to extract_all_70.py
    "label": LABEL, "normalize": True, "normalizeScale": 100, "binCount": 32,
    "resampledPixelSpacing": [1.0, 1.0, 1.0], "interpolator": "sitkBSpline",
    "preCrop": True, "correctMask": True, "padDistance": 5,
}
N_REPEATS = 5
SEED = 42
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def load_cases(ids):
    cases = {}
    for pid in ids:
        paths = [os.path.join(NRRD_DIR, pid + s) for s in ("_pre.nrrd", "_image.nrrd", "_mask.nrrd")]
        if all(os.path.exists(p) for p in paths):
            cases[pid] = paths
        else:
            say("%s: pre/post/mask file missing, skipped" % pid)
    return cases


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = pd.read_csv(REFERENCE_CSV)
    ref = ref[ref["cohort"] == "ISPY2"].sort_values("Patient_ID").reset_index(drop=True)
    cases = load_cases(list(ref["Patient_ID"]))
    say("Patients with pre, post and reference mask: %d of %d" % (len(cases), len(ref)))

    # ------------------------------------------------------------ step 1
    say("\n=== STEP 1: Dice of the automatic mask against the reference mask ===")
    imgs = {}
    for pid, (pre_p, post_p, mask_p) in cases.items():
        post = sitk.ReadImage(post_p)
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(post)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        ref_mask = sitk.Cast(r.Execute(sitk.ReadImage(mask_p)) == LABEL, sitk.sitkUInt8)
        imgs[pid] = (sitk.ReadImage(pre_p), post, ref_mask)

    table = {}
    for bg in BG_GRID:
        d = []
        for pid, (pre, post, ref_mask) in imgs.items():
            try:
                m, _ = autoseg.auto_segment(pre, post, bg_fraction=bg)
                d.append(autoseg.dice(m, ref_mask))
            except Exception:
                d.append(0.0)
        table[bg] = np.array(d)
        say("background level %.2f of the 99.5th percentile:  mean Dice %.3f  median %.3f  (Dice > 0.5 in %d of %d patients)" % (
            bg, table[bg].mean(), np.median(table[bg]), (table[bg] > 0.5).sum(), len(d)))
    best_bg = max(table, key=lambda b: table[b].mean())
    say("Chosen background level: %.2f" % best_bg)

    rows, auto_masks = [], {}
    os.makedirs(os.path.join(OUT_DIR, "automasks"), exist_ok=True)
    for pid, (pre, post, ref_mask) in imgs.items():
        row = {"Patient_ID": pid, "dice": 0.0, "auto_volume_ml": np.nan, "reference_volume_ml": np.nan, "note": ""}
        sp = post.GetSpacing()
        row["reference_volume_ml"] = float(sitk.GetArrayFromImage(ref_mask).sum() * sp[0] * sp[1] * sp[2] / 1000.0)
        try:
            m, info = autoseg.auto_segment(pre, post, bg_fraction=best_bg)
            row.update({"dice": autoseg.dice(m, ref_mask), "auto_volume_ml": info["tumor_volume_ml"],
                        "note": "; ".join(info["warnings"])})
            auto_masks[pid] = m
            out = sitk.Cast(m, sitk.sitkUInt8) * LABEL
            out.CopyInformation(post)
            sitk.WriteImage(out, os.path.join(OUT_DIR, "automasks", pid + "_mask.nrrd"), True)
        except Exception as e:
            row["note"] = "FAILED: %s" % e
        rows.append(row)
    dd = pd.DataFrame(rows)
    dd.to_csv(os.path.join(OUT_DIR, "dice_per_patient.csv"), index=False)
    say("\nAt the chosen level: mean Dice %.3f, median %.3f, range %.2f to %.2f; failed patients: %d" % (
        dd["dice"].mean(), dd["dice"].median(), dd["dice"].min(), dd["dice"].max(), (dd["note"].str.startswith("FAILED")).sum()))
    ok = dd.dropna(subset=["auto_volume_ml"])
    if len(ok) > 2:
        say("Tumor volume, automatic versus reference: Spearman correlation %.2f" % spearmanr(ok["auto_volume_ml"], ok["reference_volume_ml"])[0])
    with open(os.path.join(OUT_DIR, "autoseg_params.json"), "w") as fh:
        json.dump({"bg_fraction": best_bg, "pe_threshold": autoseg.DEFAULTS["pe_threshold"],
                   "close_mm": autoseg.DEFAULTS["close_mm"], "grow_mm": autoseg.DEFAULTS["grow_mm"],
                   "mean_dice": float(dd["dice"].mean())}, fh, indent=2)

    if not RUN_EXTRACTION:
        with open(os.path.join(OUT_DIR, "autoseg_report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return

    # ------------------------------------------------------------ step 2
    say("\n=== STEP 2: radiomics features from the automatic masks ===")
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    for k, v in SETTINGS.items():
        extractor.settings[k] = v
    feats = [c for c in ref.columns if c.startswith("original_")]
    frows = []
    for n, pid in enumerate(auto_masks, 1):
        post = imgs[pid][1]
        out = sitk.Cast(auto_masks[pid], sitk.sitkUInt8) * LABEL
        out.CopyInformation(post)
        try:
            res = extractor.execute(post, out)
            row = {k: float(v) for k, v in res.items() if k.startswith("original_")}
            row["Patient_ID"] = pid
            frows.append(row)
            print("[%d/%d] %s done" % (n, len(auto_masks), pid))
        except Exception as e:
            print("[%d/%d] %s ERROR: %s" % (n, len(auto_masks), pid, e))
    af = pd.DataFrame(frows)
    af.to_csv(os.path.join(OUT_DIR, "autoseg_features.csv"), index=False)

    # ------------------------------------------------------------ step 3
    say("\n=== STEP 3: predictions with automatic masks (repeated 5-fold CV, %d repeats, patient level) ===" % N_REPEATS)
    ref = ref[ref["Patient_ID"].isin(af["Patient_ID"])].sort_values("Patient_ID").reset_index(drop=True)
    af = af.set_index("Patient_ID").loc[ref["Patient_ID"]].reset_index()
    y = ref["pCR"].to_numpy()
    Xref, Xauto = ref[feats].to_numpy(float), af[feats].to_numpy(float)
    for model_name in ["RandomForest", "XGBoost"]:
        pipe = tm.make_pipeline(model_name, 20)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
        o_ref, o_auto, o_auto_auto = (np.zeros((N_REPEATS, len(y))) for _ in range(3))
        for k, (tr, te) in enumerate(cv.split(Xref, y)):
            f = clone(pipe).fit(Xref[tr], y[tr])
            o_ref[k // 5, te] = f.predict_proba(Xref[te])[:, 1]
            o_auto[k // 5, te] = f.predict_proba(Xauto[te])[:, 1]
            g = clone(pipe).fit(Xauto[tr], y[tr])
            o_auto_auto[k // 5, te] = g.predict_proba(Xauto[te])[:, 1]
        a = lambda o: np.mean([roc_auc_score(y, o[r]) for r in range(N_REPEATS)])
        rho = np.mean([spearmanr(o_ref[r], o_auto[r])[0] for r in range(N_REPEATS)])
        say("%s (20 features)" % model_name)
        say("  train reference, test reference masks : AUC %.3f" % a(o_ref))
        say("  train reference, test AUTOMATIC masks : AUC %.3f   (rank agreement of patient scores with the reference: %.2f)" % (a(o_auto), rho))
        say("  train AUTOMATIC, test AUTOMATIC masks : AUC %.3f" % a(o_auto_auto))

    with open(os.path.join(OUT_DIR, "autoseg_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\nSaved in", OUT_DIR)


if __name__ == "__main__":
    main()
