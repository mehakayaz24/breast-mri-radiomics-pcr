"""
Read an uploaded breast DCE-MRI exam in DICOM format and find its 3D volumes (pre-contrast, post-contrast ...).

Accepted: the .dcm files of the exam (also files without extension), or a ZIP of the exam folder.
Hospitals export dynamic series in different ways, all of these are handled:
  - one series per phase (pre, post1, post2 ...)
  - one series that contains all phases (split by TemporalPositionIdentifier, AcquisitionNumber or instance blocks)
Not handled: enhanced multi-frame DICOM (one file with many frames) and 4D files from some vendors. The app says so.

The volumes are only SUGGESTED (earliest two volumes with the same geometry); the user confirms them in the app.
"""
import os
import zipfile
from collections import defaultdict

import numpy as np
import pydicom
import SimpleITK as sitk

MIN_SLICES = 16


def save_uploads(uploaded_files, folder):
    """uploaded_files: Streamlit UploadedFile objects. ZIP files are unpacked, other files are stored as they are."""
    os.makedirs(folder, exist_ok=True)
    for i, up in enumerate(uploaded_files):
        name = os.path.basename(up.name) or ("file%d" % i)
        if name.lower().endswith(".zip"):
            zpath = os.path.join(folder, "_upload%d.zip" % i)
            with open(zpath, "wb") as f:
                f.write(up.getbuffer())
            with zipfile.ZipFile(zpath) as z:
                z.extractall(os.path.join(folder, "zip%d" % i))
            os.remove(zpath)
        else:
            with open(os.path.join(folder, "%06d_%s" % (i, name)), "wb") as f:
                f.write(up.getbuffer())


def _round_pos(ipp):
    return tuple(round(float(v), 2) for v in ipp)


def _read_headers(folder):
    heads, n_files, n_multiframe, n_other = [], 0, 0, 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            n_files += 1
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                continue
            if getattr(ds, "Modality", "") != "MR":
                n_other += 1
                continue
            if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
                n_multiframe += 1
                continue
            if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
                n_other += 1
                continue
            heads.append({
                "file": path,
                "uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "patient_id": str(getattr(ds, "PatientID", "") or "unknown"),
                "pos": _round_pos(ds.ImagePositionPatient),
                "ipp": np.array(ds.ImagePositionPatient, dtype=float),
                "iop": [float(v) for v in ds.ImageOrientationPatient],
                "inst": int(getattr(ds, "InstanceNumber", 0) or 0),
                "temporal": getattr(ds, "TemporalPositionIdentifier", None),
                "acq_num": getattr(ds, "AcquisitionNumber", None),
                "acq_time": str(getattr(ds, "AcquisitionTime", "") or getattr(ds, "ContentTime", "") or ""),
                "series_desc": str(getattr(ds, "SeriesDescription", "") or ""),
                "series_num": int(getattr(ds, "SeriesNumber", 0) or 0),
                "rows": int(ds.Rows), "cols": int(ds.Columns),
                "pix": [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])],
            })
    return heads, {"files": n_files, "multiframe": n_multiframe, "other": n_other}


def _split_series(items):
    """Split the slices of ONE series into volumes (a series may hold several phases). Returns list of lists or None."""
    positions = {it["pos"] for it in items}
    n_pos = len(positions)
    if len(items) == n_pos:
        return [items]
    if len(items) % n_pos:
        return None
    n_vol = len(items) // n_pos

    def unique(group):
        return len({it["pos"] for it in group}) == len(group)

    for key in ("temporal", "acq_num"):
        vals = [it[key] for it in items]
        if all(v is not None for v in vals):
            d = defaultdict(list)
            for it, v in zip(items, vals):
                d[str(v)].append(it)
            if len(d) == n_vol and all(len(g) == n_pos and unique(g) for g in d.values()):
                return sorted(d.values(), key=lambda g: min(it["inst"] for it in g))
    ordered = sorted(items, key=lambda it: it["inst"])
    blocks = [ordered[i * n_pos:(i + 1) * n_pos] for i in range(n_vol)]
    if all(unique(b) for b in blocks):
        return blocks
    return None


def scan_exam(folder):
    """Returns (volumes, notes). volumes: list of dicts with files (sorted by slice position) and descriptive fields."""
    heads, counts = _read_headers(folder)
    notes = []
    if counts["multiframe"]:
        notes.append("%d enhanced multi-frame DICOM files were ignored (not supported). Export the exam as classic DICOM "
                     "(one file per slice)." % counts["multiframe"])
    by_series = defaultdict(list)
    for h in heads:
        by_series[h["uid"]].append(h)

    volumes = []
    for uid, items in by_series.items():
        groups = _split_series(items)
        if groups is None:
            notes.append("Series '%s' could not be split into 3D volumes and was skipped." % items[0]["series_desc"])
            continue
        for k, g in enumerate(groups):
            if len(g) < MIN_SLICES:
                continue
            iop = g[0]["iop"]
            normal = np.cross(iop[:3], iop[3:])
            g = sorted(g, key=lambda it: float(np.dot(it["ipp"], normal)))
            depth = np.array([float(np.dot(it["ipp"], normal)) for it in g])
            gaps = np.diff(depth)
            if gaps.min() <= 0 or (gaps.max() - gaps.min()) > 0.05 * gaps.mean():
                notes.append("Series '%s' has irregular slice spacing and was skipped." % g[0]["series_desc"])
                continue
            first = g[0]
            sx, sy = first["pix"][1], first["pix"][0]
            sz = float(gaps.mean())
            geometry = (first["rows"], first["cols"], len(g), round(sx, 2), round(sy, 2), round(sz, 2),
                        tuple(round(v, 0) for v in first["ipp"]))
            volumes.append({
                "patient_id": first["patient_id"],
                "files": [it["file"] for it in g], "n_slices": len(g), "size": (first["cols"], first["rows"], len(g)),
                "spacing": (sx, sy, sz), "series_desc": first["series_desc"], "series_num": first["series_num"],
                "phase_in_series": k + 1, "n_phases_in_series": len(groups), "acq_time": first["acq_time"],
                "geometry": geometry, "order": (first["series_num"], k, first["acq_time"]),
            })
    volumes.sort(key=lambda v: (v["patient_id"], v["order"]))
    return volumes, notes


def patients_in(volumes):
    return sorted({v["patient_id"] for v in volumes})


def volumes_of(volumes, patient_id):
    """The volumes of one patient, numbered from 1 (so the labels can be shown in a drop-down)."""
    sub = [dict(v) for v in volumes if v["patient_id"] == patient_id]
    sub.sort(key=lambda v: v["order"])
    for i, v in enumerate(sub):
        suffix = (" - phase %d of %d" % (v["phase_in_series"], v["n_phases_in_series"])) if v["n_phases_in_series"] > 1 else ""
        v["label"] = "#%d  Series %s: %s%s  (%d slices, %.2f x %.2f x %.2f mm)" % (
            i + 1, v["series_num"], v["series_desc"] or "no description", suffix, v["n_slices"], *v["spacing"])
    return sub


def suggest_pre_post(volumes):
    """Indexes (pre, post) of the two earliest volumes that share the geometry of the biggest dynamic group."""
    groups = defaultdict(list)
    for i, v in enumerate(volumes):
        groups[v["geometry"]].append(i)
    candidates = [idx for idx in groups.values() if len(idx) >= 2]
    if not candidates:
        return None
    best = max(candidates, key=lambda idx: (len(idx), -idx[0]))
    return best[0], best[1]


def load_volume(volume):
    """SimpleITK image of one volume. Geometry from SimpleITK, voxel values read with pydicom (rescale applied)."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(volume["files"])
    geom = reader.Execute()
    try:
        slices = []
        for f in volume["files"]:
            d = pydicom.dcmread(f)
            a = d.pixel_array
            slope = float(getattr(d, "RescaleSlope", 1.0))
            icpt = float(getattr(d, "RescaleIntercept", 0.0))
            if slope != 1.0 or icpt != 0.0:
                a = a.astype(np.float32) * slope + icpt
            slices.append(a)
        vol = np.stack(slices)
        if vol.shape == (geom.GetSize()[2], geom.GetSize()[1], geom.GetSize()[0]):
            img = sitk.GetImageFromArray(vol)
            img.CopyInformation(geom)
            return img
    except Exception:
        pass
    return geom                                    # e.g. compressed DICOM that pydicom cannot decode: use SimpleITK's values


def preview_slice(volume):
    """Middle slice as a 0..1 float array for a thumbnail."""
    f = volume["files"][len(volume["files"]) // 2]
    d = pydicom.dcmread(f)
    a = d.pixel_array.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
