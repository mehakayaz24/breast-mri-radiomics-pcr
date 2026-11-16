"""
Research prototype: pCR (1) vs non-pCR (0) from the baseline breast MRI (Streamlit app).

Input : the baseline DCE-MRI exam as DICOM (the .dcm files, or ZIP files of exam folders).
        One patient or several patients in one upload: for several patients the app can analyse them all.
        (Advanced: NRRD or NIfTI files, or your own tumor mask.)
Steps : automatic tumor segmentation (enhancement based, autoseg.py) -> radiomics features (PyRadiomics, same
        settings as in training) -> five models -> predicted class 1 = pCR, 0 = non-pCR.

Run (cmd), from any folder:
  python -m streamlit run "C:\\Users\\ntech\\Desktop\\prototype\\app_mri.py" --server.maxUploadSize 2000
Folder 'prototype' must contain: app_mri.py, autoseg.py, dicom_loader.py, final_models.joblib (optional: autoseg_params.json)

Proof of concept only. NOT a medical device and NOT for clinical decisions.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

import autoseg
import dicom_loader as dl

# ===================== EDIT THESE: title and team shown in the app =====================
PROJECT_TITLE = "Radiomic Features of Breast MRI as Predictors of Neoadjuvant Chemotherapy Efficacy in Locally Advanced Breast Cancer"
PROTOTYPE_SUBTITLE = "Prototype: predicts pCR (1) versus non-pCR (0) from the baseline DCE-MRI"
TEAM_MEMBERS = ["Mehak (22BM001)", "Ayesha (22BM011)"]
SUPERVISOR = "Engr. Rabia Chandio"
INSTITUTION = "Department of Biomedical Engineering, Mehran University of Engineering and Technology, Jamshoro"
# =======================================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "final_models_126.joblib")
SUBTYPE_FILES = {                                   # label shown in the app -> model file (all must be in the app folder)
    "Not specified (HER2+ and TNBC together)": "final_models_126.joblib",
    "HER2+": "final_models_her2.joblib",
    "TNBC": "final_models_tnbc.joblib",
}
PARAMS_FILE = os.path.join(HERE, "autoseg_params.json")
LABEL = 255

SETTINGS = {                      # identical to the settings used for the training features
    "normalize": True, "normalizeScale": 100, "binCount": 32, "resampledPixelSpacing": [1.0, 1.0, 1.0],
    "interpolator": "sitkBSpline", "preCrop": True, "correctMask": True, "padDistance": 5,
}

# Ranges of the 50 training scans (from conversion_log.csv, I-SPY2/ACRIN baseline DCE-MRI, first post-contrast phase)
TRAINING_RANGES = {
    "In-plane voxel size (mm)": (0.31, 1.41),
    "Slice thickness (mm)": (1.0, 2.5),
    "In-plane matrix (pixels)": (256, 384),
    "Number of slices": (64, 200),
    "Tumor volume (mL)": (2.3, 170.0),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"], button, input, textarea, select { font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif !important; }
.stApp { background: #f4f7fa; color: #0f172a; }
.block-container { max-width: 980px; padding-top: 1.4rem; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], #MainMenu, footer, header { display: none !important; }
.hero { background: linear-gradient(135deg, #0f766e 0%, #155e75 100%); color: #ffffff; border-radius: 18px; padding: 26px 30px; margin-bottom: 16px; }
.hero h1 { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.01em; line-height: 1.25; }
.hero p { color: #e6fffb; margin: 0; font-size: 1rem; }
.steps { display: flex; gap: 10px; margin: 14px 0 0 0; flex-wrap: wrap; }
.step { background: rgba(255,255,255,0.16); border-radius: 999px; padding: 6px 14px; font-size: 0.85rem; color: #ffffff; font-weight: 500; }
.card { background: #ffffff; border: 1px solid #e3e9f0; border-radius: 14px; padding: 16px 20px; margin: 10px 0; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }
.card h4 { margin: 0 0 6px 0; color: #0f172a; font-size: 1.02rem; }
.card p, .card li { color: #334155; font-size: 0.93rem; }
.result-card { background: #ffffff; border: 1px solid #dbe5ee; border-radius: 18px; padding: 22px 26px; margin: 12px 0; text-align: center; box-shadow: 0 2px 8px rgba(15,23,42,0.07); }
.result-label { color: #64748b; font-size: 0.9rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.result-score { font-size: 3.4rem; font-weight: 700; color: #0f766e; line-height: 1.1; margin: 4px 0 10px 0; }
.bar { height: 12px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin: 0 auto 12px auto; max-width: 520px; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #5eead4, #0f766e); border-radius: 999px; }
.verdict { display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.95rem; }
.verdict-pcr { background: #ccfbf1; color: #115e59; }
.verdict-no { background: #e2e8f0; color: #334155; }
.notice { background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; border-radius: 12px; padding: 10px 14px; margin: 8px 0; font-size: 0.9rem; }
.danger { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; border-radius: 12px; padding: 12px 16px; margin: 8px 0; font-weight: 600; }
.footer { background: #ffffff; border: 1px solid #e3e9f0; border-radius: 14px; padding: 16px 22px; margin: 22px 0 6px 0; text-align: center; }
.footer .t { color: #0f766e; font-weight: 700; margin-bottom: 4px; }
.footer .n { color: #334155; font-size: 0.93rem; line-height: 1.6; }
.stButton > button[kind="primary"] { background: #0f766e; border: 0; border-radius: 10px; padding: 0.55rem 1.4rem; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #115e59; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.98rem; }
.small { color: #64748b; font-size: 0.82rem; }
</style>
"""


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Same class as in train_models.py, needed to load the saved models."""

    def __init__(self, limit=0.9):
        self.limit = limit

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        keep = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if not keep[i]:
                continue
            for j in range(i + 1, X.shape[1]):
                if keep[j] and corr[i, j] > self.limit:
                    keep[j] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def load_bundle(path_or_file):
    fake = types.ModuleType("train_models")          # the models were pickled with this class path
    fake.CorrelationFilter = CorrelationFilter
    sys.modules["train_models"] = fake
    sys.modules["__main__"].CorrelationFilter = CorrelationFilter
    return joblib.load(path_or_file)


@st.cache_resource(show_spinner=False)
def get_bundle(path=None):
    return load_bundle(path or MODEL_FILE)


def get_bundles():
    """All model files that exist in the app folder: {label: bundle}. The pooled model must exist."""
    out = {}
    for label, fname in SUBTYPE_FILES.items():
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            out[label] = get_bundle(path)
    return out


def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as fh:
            p = json.load(fh)
        return {k: p[k] for k in ("bg_fraction", "pe_threshold", "close_mm", "grow_mm") if k in p}
    return {}


def needed_features(bundle):
    """Only the features that the models really use (the selected features of all five models)."""
    return sorted({f for fs in bundle["selected_features"].values() for f in fs})


@st.cache_resource(show_spinner=False)
def get_extractor(needed):
    """PyRadiomics extractor that computes only the needed features (faster than all 107). Falls back to all features."""
    from radiomics import featureextractor       # needs: pip install pyradiomics
    ex = featureextractor.RadiomicsFeatureExtractor()
    try:
        by_class = {}
        for f in needed:
            _, cls, name = f.split("_", 2)
            by_class.setdefault(cls, []).append(name)
        ex.disableAllFeatures()
        ex.enableFeaturesByName(**by_class)
    except Exception:
        ex = featureextractor.RadiomicsFeatureExtractor()
        ex.enableAllFeatures()
    for key, value in SETTINGS.items():
        ex.settings[key] = value
    ex.settings["label"] = LABEL
    ex.settings["additionalInfo"] = False
    return ex


def extract_features(image, mask, needed):
    result = get_extractor(tuple(needed)).execute(image, mask)
    return {k: float(v) for k, v in result.items() if k.startswith("original_")}


def analyse(pre, post, mask_img, bundle, params, side=None, report=None):
    """pre, post: SimpleITK images; mask_img: optional (>0 = tumor). Returns a result dict (arrays are not kept)."""
    say = report or (lambda text: None)
    if post.GetDimension() != 3 or pre.GetDimension() != 3:
        raise ValueError("Both images must be 3D volumes.")
    timings = {}

    t = time.time()
    say("Finding the tumor ...")
    if mask_img is not None:
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(post)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        mask01 = sitk.Cast(r.Execute(mask_img) > 0, sitk.sitkUInt8)
        sp = post.GetSpacing()
        n = int(sitk.GetArrayFromImage(mask01).sum())
        info = {"tumor_voxels": n, "tumor_volume_ml": n * sp[0] * sp[1] * sp[2] / 1000.0, "warnings": [],
                "source": "your mask"}
    else:
        mask01, info = autoseg.auto_segment(pre, post, side=side, **params)
        info["source"] = "automatic segmentation"
    if info["tumor_voxels"] == 0:
        raise ValueError("The tumor mask is empty.")
    mask255 = sitk.Cast(mask01, sitk.sitkUInt8) * LABEL
    mask255.CopyInformation(post)
    timings["segmentation"] = time.time() - t

    t = time.time()
    say("Computing the radiomics features (the slowest step) ...")
    needed = needed_features(bundle)
    feats = extract_features(post, mask255, needed)
    missing = [n for n in needed if n not in feats]
    if missing:
        raise ValueError("Missing features: %s" % missing[:5])
    timings["features"] = time.time() - t

    t = time.time()
    say("Running the five models ...")
    stats = bundle["feature_stats"]
    names = bundle["feature_names"]
    # the models only use the needed features; the other columns are filled with the training mean (no effect on the result)
    row = np.array([[feats[n] if n in feats else float(stats.loc[n, "mean"]) for n in names]], dtype=float)
    probs = {name: float(model.predict_proba(row)[0, 1]) for name, model in bundle["models"].items()}
    z = {f: (feats[f] - stats.loc[f, "mean"]) / (stats.loc[f, "std"] + 1e-12) for f in needed}
    outside = [f for f, v in z.items() if abs(v) > 4]
    timings["models"] = time.time() - t

    msk = sitk.GetArrayFromImage(mask01) > 0
    img = sitk.GetArrayFromImage(post)
    fig = overlay_figure(img, msk)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    tmp = tempfile.mkdtemp(prefix="mask_")
    try:
        sitk.WriteImage(mask255, os.path.join(tmp, "tumor_mask.nrrd"), True)
        with open(os.path.join(tmp, "tumor_mask.nrrd"), "rb") as fh:
            mask_bytes = fh.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"probs": probs, "info": info, "outside_features": outside, "overlay_png": buf.getvalue(),
            "mask_bytes": mask_bytes, "checks": input_checks(post, info), "timings": timings}


def read_image_file(path):
    return sitk.ReadImage(path)


def input_checks(post, info):
    """Compare the uploaded exam with the range of the training scans. Returns a DataFrame."""
    sx, sy, sz = post.GetSpacing()
    nx, ny, nz = post.GetSize()
    values = {
        "In-plane voxel size (mm)": (min(sx, sy), max(sx, sy), "%.2f x %.2f" % (sx, sy)),
        "Slice thickness (mm)": (sz, sz, "%.2f" % sz),
        "In-plane matrix (pixels)": (min(nx, ny), max(nx, ny), "%d x %d" % (nx, ny)),
        "Number of slices": (nz, nz, "%d" % nz),
        "Tumor volume (mL)": (info["tumor_volume_ml"], info["tumor_volume_ml"], "%.1f" % info["tumor_volume_ml"]),
    }
    rows = []
    for name, (lo_v, hi_v, shown) in values.items():
        lo, hi = TRAINING_RANGES[name]
        inside = lo_v >= lo * 0.999 and hi_v <= hi * 1.001
        rows.append({"Property": name, "Your exam": shown, "Training range": "%g to %g" % (lo, hi),
                     "Status": "within range" if inside else "OUTSIDE training range"})
    return pd.DataFrame(rows)


def overlay_figure(img, msk):
    z = int(np.argmax(msk.sum(axis=(1, 2))))
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.imshow(img[z], cmap="gray")
    ax.contour(msk[z].astype(float), levels=[0.5], colors="#ef4444", linewidths=0.9)
    ax.set_title("Post-contrast slice %d (largest tumor area)" % z, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def predicted_class(probs):
    """Majority vote of the models (score >= 0.5 votes for 1 = pCR). Returns (class 0/1, votes for 1, n models, mean score)."""
    votes = sum(p >= 0.5 for p in probs.values())
    n = len(probs)
    return int(votes > n / 2), votes, n, float(np.mean(list(probs.values())))


def result_html(probs):
    cls, votes, n, mean_p = predicted_class(probs)
    agree = votes if cls == 1 else n - votes
    name = "pCR" if cls == 1 else "non-pCR"
    style = "verdict-pcr" if cls == 1 else "verdict-no"
    return ('<div class="result-card"><div class="result-label">Predicted class</div>'
            '<div class="result-score">%d</div><div style="font-size:1.25rem;font-weight:600;margin-bottom:10px">'
            '%s <span class="small">(1 = pCR, 0 = non-pCR)</span></div>'
            '<span class="verdict %s">%d of %d models agree</span>'
            '<div class="bar" style="margin-top:14px"><div class="bar-fill" style="width:%d%%"></div></div>'
            '<div class="small">Average model score for pCR: %.2f (0 to 1, not a calibrated probability)</div></div>'
            % (cls, name, style, agree, n, int(round(100 * min(max(mean_p, 0.0), 1.0))), mean_p))


def show_item(item, bundle, key):
    """Detailed result of one patient."""
    n_out = int((item["checks"]["Status"] != "within range").sum())
    if n_out or len(item["outside_features"]) >= 3:
        st.markdown('<div class="danger">This exam differs from the training data (%d scan properties outside the training range, '
                    '%d model features far outside the training range). Do not trust the result below.</div>'
                    % (n_out, len(item["outside_features"])), unsafe_allow_html=True)
    if item["info"].get("source") == "automatic segmentation":
        st.markdown('<div class="notice"><b>Tumor found automatically (approximate): treat this prediction as a demonstration only.</b> '
                    'The automatic rule overlaps only weakly with the reference tumor masks used for training (Dice about 0.2 on the 50 training '
                    'patients). When the models were tested on automatic masks of these patients, the AUC fell from about 0.70 to about 0.53 (evaluated on the first 50 patients) '
                    '(close to chance). For a more reliable result, upload your own tumor mask under <i>Advanced options</i>.</div>', unsafe_allow_html=True)
    st.markdown(result_html(item["probs"]), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Model": list(item["probs"].keys()),
                               "Score for pCR": ["%.2f" % v for v in item["probs"].values()],
                               "Predicted class": ["1 (pCR)" if v >= 0.5 else "0 (non-pCR)" for v in item["probs"].values()],
                               "Cross-validated AUC": ["%.2f" % bundle["cv_auc"].get(k, float("nan")) for k in item["probs"]]}),
                 hide_index=True)
    for w in item["info"].get("warnings", []):
        st.markdown('<div class="notice">%s</div>' % w, unsafe_allow_html=True)
    left, right = st.columns(2)
    left.image(item["overlay_png"])
    right.write("**Check the tumor outline**")
    right.write("Mask source: %s" % item["info"].get("source"))
    right.write("Tumor volume: %.1f mL (%d voxels)" % (item["info"]["tumor_volume_ml"], item["info"]["tumor_voxels"]))
    if "n_enhancing_clusters" in item["info"]:
        right.write("Enhancing clusters found: %d" % item["info"]["n_enhancing_clusters"])
    right.download_button("Download the tumor mask (.nrrd)", item["mask_bytes"], file_name="tumor_mask_%s.nrrd" % key, key="dl_" + key)
    t = item["timings"]
    st.markdown('<div class="small">Always look at the red outline: if it is not on the tumor, the prediction is meaningless. '
                'Time: segmentation %.0f s, features %.0f s, models %.0f s.</div>' % (t["segmentation"], t["features"], t["models"]),
                unsafe_allow_html=True)
    st.write("**Input check against the training data**")
    st.dataframe(item["checks"], hide_index=True)


def show_results(results, bundle):
    items = results["items"]
    st.subheader("Result" if len(items) == 1 else "Results (%d patients)" % len(items))
    if len(items) == 1 and not results.get("errors"):
        show_item(items[0], bundle, "single")
    else:
        rows = []
        for it in items:
            cls, votes, n, mean_p = predicted_class(it["probs"])
            flagged = int((it["checks"]["Status"] != "within range").sum()) + (len(it["outside_features"]) >= 3)
            rows.append({"Patient": it["patient"], "Predicted class (1 = pCR, 0 = non-pCR)": cls,
                         "Models voting for pCR": "%d of %d" % (votes, n), "Average score": round(mean_p, 2),
                         "Tumor volume (mL)": round(it["info"]["tumor_volume_ml"], 1),
                         "Outside training range": "yes" if flagged else "no"})
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True)
        st.download_button("Download the table (.csv)", table.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", key="dl_csv")
        for patient, msg in results.get("errors", []):
            st.markdown('<div class="notice"><b>%s</b>: %s</div>' % (patient, msg), unsafe_allow_html=True)
        for i, it in enumerate(items):
            cls = predicted_class(it["probs"])[0]
            with st.expander("%s: predicted class %d (%s)" % (it["patient"], cls, "pCR" if cls == 1 else "non-pCR")):
                show_item(it, bundle, "p%d" % i)
    st.markdown('<div class="notice">Cross-validated AUC is about 0.62 to 0.68 depending on the model, on 126 development patients (TNBC and HER2+ only), so one '
                'prediction is uncertain. Research prototype, not for clinical use.</div>', unsafe_allow_html=True)


def scan_uploaded_exam(files):
    """Save the uploaded DICOM files once per upload and read the volumes (kept in session_state)."""
    key = tuple((f.name, f.size) for f in files)
    if st.session_state.get("exam_key") != key:
        old = st.session_state.get("exam_dir")
        if old:
            shutil.rmtree(old, ignore_errors=True)
        folder = tempfile.mkdtemp(prefix="exam_")
        dl.save_uploads(files, folder)
        volumes, notes = dl.scan_exam(folder)
        st.session_state.update(exam_key=key, exam_dir=folder, volumes=volumes, notes=notes, results=None)
    return st.session_state["volumes"], st.session_state["notes"]


def save_upload(uploaded, folder):
    path = os.path.join(folder, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


REQUIREMENTS_MD = """
#### What to upload
| Item | Requirement |
|---|---|
| Exam | **Baseline** (before chemotherapy) **dynamic contrast-enhanced (DCE) breast MRI**: T1-weighted, fat-suppressed, 3D, axial |
| Format | **DICOM**: select all the `.dcm` files of the exam, or upload **ZIP files** of the exam folders. The app finds the pre-contrast and the first post-contrast volume and asks you to confirm them |
| Several patients | You may upload several patients at once (for example one ZIP with five patients). Choose **one patient** or **all patients** |
| Advanced | NRRD or NIfTI volumes (pre and post), and optionally your own tumor mask |
| Patient | Intended for **TNBC or HER2-positive** tumors before neoadjuvant chemotherapy |

#### Not accepted
Photos, screenshots or PDF/JPG/PNG images of an MRI, single 2D slices, post-treatment scans, MRI without contrast, DWI or T2 only,
CT, mammography, ultrasound, and enhanced multi-frame DICOM (export classic DICOM, one file per slice).

#### Why not a JPG or PNG?
The model needs the full 3D volume with real signal values and the real voxel size. A picture keeps neither: the grey values
are rescaled, the voxel size is lost, and a pre- and a post-contrast volume cannot be compared. Every hospital can export DICOM.

#### The training scans
The model saw cropped single-breast volumes (matrix 256 to 384, 64 to 200 slices, voxels 0.31 to 1.41 mm in-plane, 1.0 to 2.5 mm slice thickness).
After you press Predict the app compares your exam with these ranges and warns you when it is outside.
For a **whole bilateral scan** choose the breast side under *Advanced options*; the result can still be less reliable.
"""

ABOUT_MD = """
#### Intended use
Research prototype (proof of concept) of a final-year project. It predicts whether a locally advanced breast tumor is likely to
reach a **pathological complete response (pCR = 1)** or **no pCR (non-pCR = 0)** after neoadjuvant chemotherapy, from the **baseline (pre-treatment)** MRI.
It is **not a medical device** and must not be used for clinical decisions.

#### How it works
1. The pre- and post-contrast volumes are read from the DICOM exam.
2. The tumor is found automatically: voxels whose signal rises by at least 70% after contrast (the I-SPY functional tumor volume idea), main lesion kept.
3. Radiomics features (shape, intensity, texture) are computed after resampling to 1 x 1 x 1 mm and intensity normalisation.
4. Five models give a score each (Logistic Regression, Elastic-Net Logistic Regression, Extra Trees, Random Forest, Ensemble); the majority vote is the predicted class.

#### Training data
- **126 patients** of the I-SPY2 trial (93 from the I-SPY2 collection, 33 from ACRIN-6698), baseline DCE-MRI.
- **Triple-negative and HER2-positive tumors only.** HR+/HER2- tumors were not studied.
- **53 pCR and 73 non-pCR** (HER2+ 61, TNBC 65), randomly drawn from the available patients; the training data may not reflect the real pCR rate.
- The tumor mask used for training is the I-SPY functional tumor volume mask.

#### How well does it work?
- Cross-validated **AUC about 0.62 to 0.68** depending on the model (best single model about 0.68, 95% confidence interval about 0.58 to 0.78): modest, a single prediction is uncertain.
- Tumor shape and size features are the most robust to segmentation changes, texture features less. A tumor mask shifted by 2 mm can lower the AUC clearly.
- **Automatic tumor segmentation is only approximate.** It is a simple enhancement rule (no deep learning) and it overlaps only weakly with the reference masks (Dice about 0.2 on the training patients, because the reference masks were limited to a region drawn by a radiologist). When the models were tested on automatic masks, the AUC fell from about 0.70 to about 0.53 (evaluated on the first 50 patients), close to chance. For a more reliable prediction upload your own tumor mask (.nrrd or .nii) under *Advanced options*; a solid mask drawn around the tumor lowered the AUC only slightly in our robustness test, whereas a mask shifted by 2 mm lowered it clearly.

#### Not suitable for
HR-positive/HER2-negative tumors, scans during or after treatment, scanners or protocols outside the training ranges, other cancers, and any clinical use.
"""


def footer_html():
    team = "<br>".join(TEAM_MEMBERS)
    parts = ['<div class="footer"><div class="t">Developed by</div><div class="n">%s</div>' % team]
    if SUPERVISOR:
        parts.append('<div class="n" style="margin-top:6px"><b>Supervisor:</b> %s</div>' % SUPERVISOR)
    if INSTITUTION:
        parts.append('<div class="n">%s</div>' % INSTITUTION)
    parts.append('<div class="small" style="margin-top:8px">Final year project, research prototype. Not a medical device.</div></div>')
    return "".join(parts)


def main():
    st.set_page_config(page_title="Breast MRI pCR prototype", page_icon="🩻", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>%s</h1><p>%s</p>'
        '<div class="steps"><span class="step">1  Upload the DICOM exam</span><span class="step">2  Confirm pre and post volumes</span>'
        '<span class="step">3  Read the class: 1 = pCR, 0 = non-pCR</span></div></div>' % (PROJECT_TITLE, PROTOTYPE_SUBTITLE),
        unsafe_allow_html=True)

    try:
        bundles = get_bundles()
        first = list(SUBTYPE_FILES)[0]
        if first not in bundles:
            raise FileNotFoundError("final_models_126.joblib not found next to the app")
        bundle = bundles[first]
    except Exception as e:
        st.error("Could not load the model files (they must be in the same folder as app_mri_126.py): %s" % e)
        return

    tab_predict, tab_req, tab_about = st.tabs(["Predict", "Requirements", "About the model"])
    with tab_req:
        st.markdown(REQUIREMENTS_MD)
        st.write("**Range of the training scans**")
        st.dataframe(pd.DataFrame([{"Property": k, "Training range": "%g to %g" % v} for k, v in TRAINING_RANGES.items()]),
                     hide_index=True)
    with tab_about:
        st.markdown(ABOUT_MD)
        for label, b in bundles.items():
            st.write("**%s** (trained on %d patients)" % (label, b.get("n_patients", 0)))
            st.dataframe(pd.DataFrame({"Model": list(b["cv_auc"].keys()),
                                       "Cross-validated AUC": ["%.2f" % v for v in b["cv_auc"].values()]}), hide_index=True)

    with tab_predict:
        st.markdown('<div class="card"><h4>Required input</h4><p>The <b>baseline (pre-treatment) DCE breast MRI</b> as '
                    '<b>DICOM</b>: T1 fat-suppressed, 3D axial, with a pre-contrast and at least one post-contrast phase. '
                    'Intended for <b>TNBC or HER2+</b> tumors before neoadjuvant chemotherapy. You may upload one patient or '
                    'several patients (for example one ZIP with five patients). Details: tab <i>Requirements</i>.</p></div>',
                    unsafe_allow_html=True)

        params = load_params()
        choices = list(bundles)
        subtype = choices[0]
        if len(choices) > 1:
            subtype = st.radio("Tumor subtype (known from the biopsy)", choices, horizontal=True,
                               help="Subtype-specific models were trained on fewer patients (about 60 each) and are exploratory.")
        bundle = bundles[subtype]
        best = max(bundle["cv_auc"].values())
        st.caption("Models used: %s, trained on %d patients, best cross-validated AUC %.2f." % (subtype, bundle.get("n_patients", 0), best)
                   + (" At this size the AUC is very uncertain (95% interval roughly 0.5 to 0.75), close to chance." if subtype == "TNBC" else "")
                   + (" Exploratory: tumor volume alone reaches about the same AUC." if subtype == "HER2+" else ""))
        mode = st.radio("Input format", ["DICOM exam (recommended)", "NRRD or NIfTI files (advanced)"], horizontal=True)

        jobs = []                      # list of (patient name, pre volume dict or None, post volume dict or None)
        pre_up = post_up = None
        ready = False

        if mode.startswith("DICOM"):
            files = st.file_uploader("Select all DICOM files of the exam (any file name), or upload ZIP files of the exam folders",
                                     accept_multiple_files=True)
            if files:
                with st.spinner("Reading the DICOM exam..."):
                    volumes, notes = scan_uploaded_exam(files)
                for n in notes:
                    st.markdown('<div class="notice">%s</div>' % n, unsafe_allow_html=True)
                patients = dl.patients_in(volumes)
                if not patients:
                    st.error("No 3D MR volumes were found in the upload. Check that you selected the DICOM files (or ZIP) of the exam.")
                else:
                    scope = "One patient"
                    if len(patients) > 1:
                        st.success("%d patients found in the upload." % len(patients))
                        scope = st.radio("What do you want to analyse?", ["One patient", "All patients (the app picks the pre and "
                                         "first post-contrast volume of each)"], horizontal=True)
                    if scope.startswith("One"):
                        pid = patients[0] if len(patients) == 1 else st.selectbox("Patient", patients)
                        sub = dl.volumes_of(volumes, pid)
                        if len(sub) < 2:
                            st.error("Fewer than two 3D MR volumes were found for this patient. A DCE exam needs a pre-contrast and "
                                     "a post-contrast volume.")
                        else:
                            labels = [v["label"] for v in sub]
                            sug = dl.suggest_pre_post(sub)
                            st.write("The app suggests the two earliest dynamic volumes; please confirm them.")
                            c1, c2 = st.columns(2)
                            pre_i = c1.selectbox("Pre-contrast volume", range(len(sub)), format_func=lambda i: labels[i],
                                                 index=sug[0] if sug else 0)
                            post_i = c2.selectbox("First post-contrast volume", range(len(sub)), format_func=lambda i: labels[i],
                                                  index=sug[1] if sug else 1)
                            t1, t2 = st.columns(2)
                            t1.image(dl.preview_slice(sub[pre_i]), caption="Pre-contrast (middle slice)", clamp=True)
                            t2.image(dl.preview_slice(sub[post_i]), caption="Post-contrast (middle slice)", clamp=True)
                            if pre_i == post_i:
                                st.error("Pre and post must be two different volumes.")
                            else:
                                if sub[pre_i]["geometry"] != sub[post_i]["geometry"]:
                                    st.markdown('<div class="notice">The two volumes do not have the same size or position. The '
                                                'pre-contrast volume will be resampled onto the post-contrast volume.</div>',
                                                unsafe_allow_html=True)
                                jobs = [(pid, sub[pre_i], sub[post_i])]
                    else:
                        skipped = []
                        for pid in patients:
                            sub = dl.volumes_of(volumes, pid)
                            sug = dl.suggest_pre_post(sub) if len(sub) >= 2 else None
                            if sug:
                                jobs.append((pid, sub[sug[0]], sub[sug[1]]))
                            else:
                                skipped.append(pid)
                        if skipped:
                            st.markdown('<div class="notice">No pre/post pair could be found for: %s. They are skipped.</div>'
                                        % ", ".join(skipped), unsafe_allow_html=True)
                        st.markdown('<div class="notice">Each patient takes about 1 minute. Check the tumor outline of every patient '
                                    'in the results.</div>', unsafe_allow_html=True)
                    ready = bool(jobs)
        else:
            c1, c2 = st.columns(2)
            pre_up = c1.file_uploader("1. Pre-contrast volume (.nrrd, .nii, .nii.gz)", type=["nrrd", "nii", "gz"])
            post_up = c2.file_uploader("2. First post-contrast volume", type=["nrrd", "nii", "gz"])
            ready = bool(pre_up and post_up)

        with st.expander("Advanced options: tumor mask upload, breast side, thresholds", expanded=False):
            side_choice = st.selectbox("Breast side (only for a whole bilateral scan)", ["Whole image", "Right breast", "Left breast"],
                                       help="The training scans were single-breast volumes. For a bilateral scan choose the side with the tumor.")
            side = {"Whole image": None, "Right breast": "right", "Left breast": "left"}[side_choice]
            bg = st.number_input("Background level (fraction of the 99.5th percentile of the pre-contrast image)", min_value=0.0,
                                 max_value=0.5, value=float(params.get("bg_fraction", autoseg.DEFAULTS["bg_fraction"])), step=0.01)
            pe = st.number_input("Enhancement threshold (%)", min_value=10.0, max_value=300.0,
                                 value=float(params.get("pe_threshold", autoseg.DEFAULTS["pe_threshold"])), step=5.0)
            own_mask_file = st.file_uploader("Recommended for a reliable result: your own tumor mask (.nrrd or .nii) of the post-contrast image, single patient only. Leave empty to use the approximate automatic segmentation.",
                                             type=["nrrd", "nii", "gz"])
            params = dict(params, bg_fraction=bg, pe_threshold=pe)

        if st.button("Predict", type="primary", disabled=not ready):
            tmp = tempfile.mkdtemp(prefix="run_")
            items, errors = [], []
            bar = st.progress(0.0)
            status = st.empty()
            try:
                if mode.startswith("DICOM"):
                    todo = jobs
                else:
                    todo = [("uploaded files", None, None)]
                for k, (pid, pre_v, post_v) in enumerate(todo):
                    prefix = "" if len(todo) == 1 else "Patient %d of %d (%s): " % (k + 1, len(todo), pid)
                    try:
                        status.info(prefix + "Reading the images ...")
                        if mode.startswith("DICOM"):
                            pre_img, post_img = dl.load_volume(pre_v), dl.load_volume(post_v)
                        else:
                            pre_img = read_image_file(save_upload(pre_up, tmp))
                            post_img = read_image_file(save_upload(post_up, tmp))
                        mask_img = read_image_file(save_upload(own_mask_file, tmp)) if (own_mask_file and len(todo) == 1) else None
                        out = analyse(pre_img, post_img, mask_img, bundle, params, side, report=lambda t, p=prefix: status.info(p + t))
                        out["patient"] = pid
                        items.append(out)
                    except ModuleNotFoundError:
                        status.empty()
                        st.error("PyRadiomics is not installed in this Python. Run:  python -m pip install pyradiomics")
                        return
                    except Exception as e:
                        errors.append((pid, str(e)))
                    bar.progress((k + 1) / len(todo))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            status.empty()
            bar.empty()
            st.session_state["results"] = {"items": items, "errors": errors, "subtype": subtype} if (items or errors) else None
            if not items:
                for pid, msg in errors:
                    st.error("%s: %s" % (pid, msg))

        res = st.session_state.get("results")
        if res and res["items"]:
            show_results(res, bundles.get(res.get("subtype"), bundle))
            if res.get("subtype") and res.get("subtype") != first:
                st.caption("These scores come from the %s-specific models." % res["subtype"])

    st.markdown('<div class="small" style="text-align:center;margin:22px 0 6px 0">Research prototype. Not a medical device.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
