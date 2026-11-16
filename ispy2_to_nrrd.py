"""
I-SPY2 / ACRIN-6698: baseline (T0) DCE MRI + FTV "Analysis Mask" (DICOM SEG)  ->  NRRD

What it does, per patient folder  <patient>/MR  and  <patient>/SEG :
  1. reads the MR volume (original DCE, cropped) and saves  <patient>_image.nrrd
  2. reads the bit-encoded Analysis Mask and builds
        TUMOR (FTV) mask  (voxels with mask value 0 = included in FTV; saved with value 255,
                           same as the ISPY1 masks and label=255 in extract_all.py) -> <patient>_mask.nrrd
        VOI mask          (rectangular volume of interest; bit 32 clear), optional   -> <patient>_mask_voi.nrrd
  3. checks everything (shapes, geometry, "is the VOI really a box?") and writes a log CSV.
     A patient that fails a check is NOT written, it is only reported in the log.

Mask bits (TCIA "Analysis mask files description"; masks are INVERSE: 0 = included):
  1 PE threshold | 2 MNC filter | 8 Background | 32 Manual VOI | 64 OMIT regions

NOTE: the cropped "original DCE" series stores ALL DCE phases (pre + post-contrast) in ONE series,
so the script first splits it into phases and saves only the phase you choose (PHASE_INDEX).
The mask is the same for every phase.

Use:
  1) MODE = "inspect"  -> looks at ONE patient only, prints details, writes nothing.
  2) MODE = "all"      -> converts every patient found under RAW_DIR (needs PHASE_INDEX set).

Needs: pip install pydicom   (numpy and SimpleITK are already in radiomics_env)
"""
import os
import csv
import glob
import traceback
import numpy as np
import pydicom
import SimpleITK as sitk

# ============================ EDIT THESE ============================
RAW_DIR = r"C:\PATH\TO\Dataset ISPY2\raw_dicom"   # folder that contains the extracted batch folders
OUT_DIR = r"C:\PATH\TO\Dataset ISPY2\nrrd"        # NRRD files will be written here
MODE = "all"                                      # "all" = convert every patient; "inspect" = only look at two patients
INSPECT_PATIENTS = ["ISPY2-129837", "ACRIN-6698-104268"]   # patients to look at in inspect mode (empty list = first patient found)
PHASE_INDEX = 1                                # which DCE phase to save: 0 = first (pre-contrast), 1 = first post-contrast, ...
                                                  # must be set to a number before MODE = "all"; use the same phase you used for ISPY1
SAVE_PRE = True                                   # also save the pre-contrast phase as <patient>_pre.nrrd (needed for automatic segmentation)
PRE_PHASE_INDEX = 0                               # pre-contrast = first phase
SAVE_VOI_MASK = False                             # also save the rectangular VOI as <patient>_mask_voi.nrrd
IMAGE_SUFFIX = "_image.nrrd"                      # change to match your ISPY1 naming if needed
MASK_SUFFIX = "_mask.nrrd"                        # tumor (FTV) mask, value 255
VOI_SUFFIX = "_mask_voi.nrrd"
PRE_SUFFIX = "_pre.nrrd"
TUMOR_VALUE = 255                                 # must match  extractor.settings['label']  in extract_all.py
# ====================================================================

VOI_BIT = 32
BOX_TOLERANCE = 0.999   # VOI voxels / bounding-box voxels must be at least this


def find_patients(raw_dir):
    """Every folder that has both an MR and a SEG subfolder is a patient."""
    found = []
    for root, dirs, files in os.walk(raw_dir):
        if os.path.basename(root) == "MR":
            patient_dir = os.path.dirname(root)
            seg_dir = os.path.join(patient_dir, "SEG")
            if os.path.isdir(seg_dir):
                found.append((os.path.basename(patient_dir), root, seg_dir))
    found.sort()
    return found


def read_seg(seg_dir):
    """Return (values uint8 [frames,y,x], frame positions [frames,3], seg orientation or None)."""
    files = [f for f in glob.glob(os.path.join(seg_dir, "*")) if os.path.isfile(f)]
    if len(files) != 1:
        raise RuntimeError("SEG folder has %d files, expected 1" % len(files))
    ds = pydicom.dcmread(files[0])
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.max() > 255:
        raise RuntimeError("SEG values exceed 255 (max %s)" % arr.max())
    arr = arr.astype(np.uint8)

    frames = ds.PerFrameFunctionalGroupsSequence
    if len(frames) != arr.shape[0]:
        raise RuntimeError("frame count mismatch: %d vs %d" % (len(frames), arr.shape[0]))
    ipps = np.array([fg.PlanePositionSequence[0].ImagePositionPatient for fg in frames], dtype=float)

    iop = None
    try:
        iop = [float(v) for v in ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient]
    except Exception:
        try:
            iop = [float(v) for v in frames[0].PlaneOrientationSequence[0].ImageOrientationPatient]
        except Exception:
            iop = None
    return arr, ipps, iop


def _round_pos(ipp):
    return tuple(round(float(v), 2) for v in ipp)


def find_phases(mr_dir, n_frames):
    """
    The cropped 'original DCE' series holds every DCE phase. Split it into phases.
    Returns (phases, info): phases = list of file-name lists (each sorted by slice position),
    ordered in acquisition order; info = text lines describing how the split was done.
    """
    files = sorted(f for f in glob.glob(os.path.join(mr_dir, "*")) if os.path.isfile(f))
    if not files:
        raise RuntimeError("MR folder is empty")
    heads = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    total = len(files)
    if total % n_frames:
        raise RuntimeError("MR has %d slices, not a multiple of the %d SEG frames" % (total, n_frames))
    n_phases = total // n_frames

    pos = [_round_pos(h.ImagePositionPatient) for h in heads]
    iop = [float(v) for v in heads[0].ImageOrientationPatient]
    normal = np.cross(iop[:3], iop[3:])
    depth = [float(np.dot(np.array(h.ImagePositionPatient, dtype=float), normal)) for h in heads]

    def unique_positions(idx):
        return len({pos[i] for i in idx}) == len(idx)

    groups, how = None, None
    if n_phases == 1:
        groups, how = [list(range(total))], "single phase"
    else:
        for name in ["TemporalPositionIdentifier", "AcquisitionNumber", "AcquisitionTime", "ContentTime", "TriggerTime"]:
            vals = [getattr(h, name, None) for h in heads]
            if any(v is None for v in vals):
                continue
            d = {}
            for i, v in enumerate(vals):
                d.setdefault(str(v), []).append(i)
            if len(d) == n_phases and all(len(g) == n_frames and unique_positions(g) for g in d.values()):
                groups, how = list(d.values()), "grouped by " + name
                break
        if groups is None:
            inst = [getattr(h, "InstanceNumber", None) for h in heads]
            if all(v is not None for v in inst):
                order = sorted(range(total), key=lambda i: int(inst[i]))
                blocks = [order[b * n_frames:(b + 1) * n_frames] for b in range(n_phases)]
                if all(unique_positions(b) for b in blocks):
                    groups, how = blocks, "consecutive InstanceNumber blocks"
    if groups is None:
        raise RuntimeError("could not split the MR series into %d phases of %d slices" % (n_phases, n_frames))

    def order_key(g):
        inst = [getattr(heads[i], "InstanceNumber", None) for i in g]
        if all(v is not None for v in inst):
            return min(int(v) for v in inst)
        return min(g)
    groups.sort(key=order_key)

    phases, info = [], ["MR slices: %d  ->  %d phases of %d slices (%s)" % (total, n_phases, n_frames, how)]
    for n, g in enumerate(groups):
        g = sorted(g, key=lambda i: depth[i])
        gaps = np.diff([depth[i] for i in g])
        if gaps.min() <= 0 or (gaps.max() - gaps.min()) > 0.05 * gaps.mean():
            raise RuntimeError("phase %d has non-uniform slice spacing" % n)
        phases.append([files[i] for i in g])
        h = heads[g[0]]
        info.append("  phase %d: first InstanceNumber=%s AcquisitionNumber=%s AcquisitionTime=%s ContentTime=%s" % (
            n, getattr(h, "InstanceNumber", "?"), getattr(h, "AcquisitionNumber", "?"),
            getattr(h, "AcquisitionTime", "?"), getattr(h, "ContentTime", "?")))
    return phases, info


def read_phase(file_names):
    """Geometry from SimpleITK, voxel values read directly with pydicom (so bit-depth is never guessed)."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    geom = reader.Execute()

    slices = []
    for f in file_names:
        d = pydicom.dcmread(f)
        a = d.pixel_array
        slope = float(getattr(d, "RescaleSlope", 1.0))
        icpt = float(getattr(d, "RescaleIntercept", 0.0))
        if slope != 1.0 or icpt != 0.0:
            a = a.astype(np.float32) * slope + icpt
        slices.append(a)
    vol = np.stack(slices)
    if vol.shape != (geom.GetSize()[2], geom.GetSize()[1], geom.GetSize()[0]):
        raise RuntimeError("pydicom volume shape %s differs from SimpleITK %s" % (vol.shape, geom.GetSize()))
    img = sitk.GetImageFromArray(vol)
    img.CopyInformation(geom)
    return img


def header_summary(file_name):
    d = pydicom.dcmread(file_name, stop_before_pixels=True)
    return "BitsAllocated=%s BitsStored=%s PixelRepresentation=%s RescaleSlope=%s RescaleIntercept=%s ImageType=%s" % (
        getattr(d, "BitsAllocated", "?"), getattr(d, "BitsStored", "?"), getattr(d, "PixelRepresentation", "?"),
        getattr(d, "RescaleSlope", "-"), getattr(d, "RescaleIntercept", "-"),
        "\\".join(getattr(d, "ImageType", ["?"])))


def align_seg(arr, ipps, seg_iop, mr):
    """Put the SEG frames on the MR slice grid. Returns (values uint8 [z,y,x], number of MR slices without a frame)."""
    size_x, size_y, size_z = mr.GetSize()
    if arr.shape[1:] != (size_y, size_x):
        raise RuntimeError("SEG slice size %s differs from MR %s" % (arr.shape[1:], (size_y, size_x)))
    if seg_iop is not None:
        d = mr.GetDirection()
        mr_iop = [d[0], d[3], d[6], d[1], d[4], d[7]]
        if max(abs(a - b) for a, b in zip(seg_iop, mr_iop)) > 1e-3:
            raise RuntimeError("SEG and MR have different orientation")

    z_spacing = mr.GetSpacing()[2]
    origins = np.array([mr.TransformIndexToPhysicalPoint((0, 0, int(k))) for k in range(size_z)])

    # slices that have no SEG frame are treated as fully masked (255 = excluded)
    values = np.full((size_z, size_y, size_x), 255, dtype=np.uint8)
    used = set()
    for i in range(arr.shape[0]):
        dist = np.linalg.norm(origins - ipps[i], axis=1)
        k = int(dist.argmin())
        if dist[k] > 0.25 * z_spacing:
            raise RuntimeError("SEG frame %d does not line up with any MR slice (%.3f mm off)" % (i, dist[k]))
        if k in used:
            raise RuntimeError("two SEG frames map to MR slice %d" % k)
        used.add(k)
        values[k] = arr[i]
    return values, size_z - len(used)


def analyse(values):
    voi = (values & VOI_BIT) == 0
    ftv = values == 0
    info = {"voi_voxels": int(voi.sum()), "ftv_voxels": int(ftv.sum())}
    if info["voi_voxels"] > 0:
        idx = np.argwhere(voi)
        lo, hi = idx.min(0), idx.max(0)
        box = int(np.prod(hi - lo + 1))
        info["boxness"] = info["voi_voxels"] / box
        info["bbox_zyx"] = (tuple(int(v) for v in lo), tuple(int(v) for v in hi))
    else:
        info["boxness"] = 0.0
        info["bbox_zyx"] = None
    info["ftv_inside_voi"] = bool((ftv & ~voi).sum() == 0)
    uniq, cnt = np.unique(values, return_counts=True)
    info["unique_values"] = {int(u): int(c) for u, c in zip(uniq, cnt)}
    return voi, ftv, info


def to_image(boolean_array, reference):
    img = sitk.GetImageFromArray(boolean_array.astype(np.uint8) * TUMOR_VALUE)
    img.CopyInformation(reference)
    return img


def process(pid, mr_dir, seg_dir, write):
    row = {"patient": pid, "status": "", "message": ""}
    try:
        arr, ipps, seg_iop = read_seg(seg_dir)
        phases, phase_info = find_phases(mr_dir, arr.shape[0])
        row["n_phases"] = len(phases)

        if write:
            if PHASE_INDEX is None:
                raise RuntimeError("PHASE_INDEX is not set (edit it at the top of the script)")
            if not (0 <= PHASE_INDEX < len(phases)):
                raise RuntimeError("PHASE_INDEX %s but this patient has %d phases" % (PHASE_INDEX, len(phases)))
            chosen = [PHASE_INDEX]
        else:
            chosen = list(range(len(phases)))

        problems = []
        results = {}
        for ph in chosen:
            mr = read_phase(phases[ph])
            try:
                values, missing = align_seg(arr, ipps, seg_iop, mr)
                results[ph] = (mr, values, missing)
            except Exception as e:
                problems.append("phase %d: %s" % (ph, e))
        if not results:
            raise RuntimeError("; ".join(problems))

        ph0 = chosen[0] if write else min(results)
        mr, values, missing = results[ph0]
        voi, ftv, info = analyse(values)

        row.update({
            "phase_saved": PHASE_INDEX if write else "",
            "mr_size_xyz": str(mr.GetSize()),
            "spacing_xyz": str(tuple(round(s, 4) for s in mr.GetSpacing())),
            "pixel_type": mr.GetPixelIDTypeAsString(),
            "missing_seg_slices": missing,
            "voi_voxels": info["voi_voxels"],
            "ftv_voxels": info["ftv_voxels"],
            "boxness": round(info["boxness"], 5),
            "ftv_inside_voi": info["ftv_inside_voi"],
        })

        if missing:
            problems.append("%d MR slices have no SEG frame" % missing)
        if info["voi_voxels"] == 0:
            problems.append("VOI is empty")
        elif info["boxness"] < BOX_TOLERANCE:
            problems.append("VOI is not a clean box (boxness %.4f)" % info["boxness"])
        if not info["ftv_inside_voi"]:
            problems.append("FTV voxels fall outside the VOI")
        if info["ftv_voxels"] == 0:
            problems.append("FTV is empty")

        if write:
            if problems:
                row["status"] = "SKIPPED"
                row["message"] = "; ".join(problems)
            else:
                sitk.WriteImage(mr, os.path.join(OUT_DIR, pid + IMAGE_SUFFIX), True)
                sitk.WriteImage(to_image(ftv, mr), os.path.join(OUT_DIR, pid + MASK_SUFFIX), True)
                if SAVE_VOI_MASK:
                    sitk.WriteImage(to_image(voi, mr), os.path.join(OUT_DIR, pid + VOI_SUFFIX), True)
                if SAVE_PRE:
                    pre = read_phase(phases[PRE_PHASE_INDEX])
                    if pre.GetSize() != mr.GetSize():
                        raise RuntimeError("pre-contrast phase has a different size than the saved phase")
                    sitk.WriteImage(pre, os.path.join(OUT_DIR, pid + PRE_SUFFIX), True)
                row["status"] = "OK"
        else:
            row["status"] = "INSPECT"
            row["message"] = "; ".join(problems) if problems else "all checks passed"
            print("\n=== %s ===" % pid)
            for line in phase_info:
                print(line)
            rest = voi & ~ftv
            m0 = results[min(results)][0]
            print("image size (x,y,z)=%s spacing=%s type=%s, missing SEG slices: %d" % (
                m0.GetSize(), tuple(round(v, 4) for v in m0.GetSpacing()), m0.GetPixelIDTypeAsString(), results[min(results)][2]))
            print("per phase:  tumor mean | rest-of-VOI mean | tumor/rest-of-VOI | whole-image mean | min..max | dtype")
            crops = []
            zc = int(np.argmax(ftv.sum(axis=(1, 2))))
            lo, hi = info["bbox_zyx"]
            y0, x0 = max(lo[1] - 10, 0), max(lo[2] - 10, 0)
            y1, x1 = hi[1] + 11, hi[2] + 11
            for ph in sorted(results):
                marr = sitk.GetArrayFromImage(results[ph][0]).astype(float)
                tmean = float(marr[ftv].mean()) if ftv.any() else float("nan")
                rmean = float(marr[rest].mean()) if rest.any() else float("nan")
                print("  phase %d:  %8.1f | %8.1f | %6.2f | %8.1f | %g..%g | %s" % (
                    ph, tmean, rmean, tmean / rmean if rmean else float("nan"), marr.mean(), marr.min(), marr.max(),
                    sitk.GetArrayFromImage(results[ph][0]).dtype))
                print("           header:", header_summary(phases[ph][0]))
                crops.append(marr[zc, y0:y1, x0:x1])
            # every tile is scaled on its own, so a dim phase is still visible
            tiles = []
            for c in crops:
                lo_v, hi_v = float(np.percentile(c, 1)), float(np.percentile(c, 99.5))
                tiles.append(np.clip((c - lo_v) / ((hi_v - lo_v) or 1.0) * 255.0, 0, 255))
            tiles.append(ftv[zc, y0:y1, x0:x1].astype(float) * 255.0)   # last tile = tumor mask
            sep = np.full((tiles[0].shape[0], 3), 128.0)
            strip = np.hstack([t if i == len(tiles) - 1 else np.hstack([t, sep]) for i, t in enumerate(tiles)])
            strip = np.repeat(np.repeat(strip, 3, axis=0), 3, axis=1).astype(np.uint8)
            png = os.path.join(OUT_DIR, "phase_preview_%s.png" % pid)
            sitk.WriteImage(sitk.GetImageFromArray(strip), png)
            print("Preview picture (phases 0..%d left to right, each scaled on its own, last tile = tumor mask): %s" % (max(results), png))
            print("Unique mask values and voxel counts:", info["unique_values"])
            print("VOI voxels:", info["voi_voxels"], " VOI box (z,y,x lo/hi):", info["bbox_zyx"])
            print("VOI boxness (1.0 = perfect box):", round(info["boxness"], 5))
            print("FTV voxels (value 0):", info["ftv_voxels"], " FTV inside VOI:", info["ftv_inside_voi"])
            print("Checks:", row["message"])
    except Exception as e:
        row["status"] = "ERROR"
        row["message"] = "%s: %s" % (type(e).__name__, e)
        if not write:
            traceback.print_exc()
    return row


def main():
    patients = find_patients(RAW_DIR)
    print("Patients found:", len(patients))
    if not patients:
        print("Nothing found. Check RAW_DIR; each patient folder needs an MR and a SEG subfolder.")
        return
    names = [p[0] for p in patients]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        print("WARNING: these patients appear more than once:", dup)

    if MODE == "inspect":
        chosen = [p for p in patients if p[0] in INSPECT_PATIENTS] if INSPECT_PATIENTS else patients[:1]
        if not chosen:
            print("None of INSPECT_PATIENTS found:", INSPECT_PATIENTS)
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        for c in chosen:
            process(*c, write=False)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for n, (pid, mr_dir, seg_dir) in enumerate(patients, 1):
        row = process(pid, mr_dir, seg_dir, write=True)
        rows.append(row)
        print("[%d/%d] %s -> %s %s" % (n, len(patients), pid, row["status"], row["message"]))

    fields = ["patient", "status", "message", "n_phases", "phase_saved", "mr_size_xyz", "spacing_xyz", "pixel_type",
              "missing_seg_slices", "voi_voxels", "ftv_voxels", "boxness", "ftv_inside_voi"]
    log_path = os.path.join(OUT_DIR, "conversion_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "OK" for r in rows)
    print("\nDone: %d OK, %d not OK, out of %d. Log: %s" % (ok, len(rows) - ok, len(rows), log_path))


if __name__ == "__main__":
    main()
