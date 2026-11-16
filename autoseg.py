"""
Automatic tumor segmentation from a baseline DCE-MRI: pre-contrast + first post-contrast volume.

It follows the idea of the I-SPY functional tumor volume (FTV): a voxel belongs to the tumor if its signal
rises by at least PE_THRESHOLD percent after contrast (PE = (post - pre) / pre * 100) and the pre-contrast
signal is above a background level. Then the main enhancing lesion is selected:

  1. enhancing voxels  = PE >= pe_threshold  AND  pre > background level
  2. opening (open_mm) removes thin enhancing structures; find connected clusters
  3. keep the cluster with the largest total enhancement (the main lesion)
  4. keep the enhancing voxels inside that cluster, grown by grow_mm (so the speckled edge is not lost)

Not a deep-learning segmentation: a transparent rule-based method. It needs BOTH images, on the same patient grid.
Parameters (bg_fraction) are calibrated against the reference masks by autoseg_evaluate.py.
"""
import numpy as np
import SimpleITK as sitk

DEFAULTS = {"bg_fraction": 0.35, "pe_threshold": 70.0, "close_mm": 3.0, "grow_mm": 3.0, "min_volume_ml": 0.5, "open_mm": 2.0}


def _same_grid(a, b):
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4))


def _radius_vox(mm, spacing):
    return [max(1, int(round(mm / s))) for s in spacing]


def _physical_x(img):
    """Patient left-right coordinate (LPS: larger x = patient's left) of every voxel, array shape (z, y, x)."""
    nx, ny, nz = img.GetSize()
    sx, sy, sz = img.GetSpacing()
    d = img.GetDirection()
    ox = img.GetOrigin()[0]
    i = np.arange(nx, dtype=np.float32)[None, None, :] * (sx * d[0])
    j = np.arange(ny, dtype=np.float32)[None, :, None] * (sy * d[1])
    k = np.arange(nz, dtype=np.float32)[:, None, None] * (sz * d[2])
    return ox + i + j + k


def auto_segment(pre, post, bg_fraction=None, pe_threshold=None, close_mm=None, grow_mm=None, min_volume_ml=None,
                 side=None, open_mm=None):
    """pre, post: SimpleITK 3D images. side: None (whole image), "left" or "right" breast of a bilateral scan
    (split at the body midline, patient left = larger x in LPS). Returns (mask uint8 0/1 on the post grid, info dict)."""
    p = dict(DEFAULTS)
    for k, v in (("bg_fraction", bg_fraction), ("pe_threshold", pe_threshold), ("close_mm", close_mm),
                 ("grow_mm", grow_mm), ("min_volume_ml", min_volume_ml), ("open_mm", open_mm)):
        if v is not None:
            p[k] = v

    post = sitk.Cast(post, sitk.sitkFloat32)
    pre = sitk.Cast(pre, sitk.sitkFloat32)
    if not _same_grid(pre, post):
        pre = sitk.Resample(pre, post, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    a_pre = sitk.GetArrayFromImage(pre).astype(np.float64)
    a_post = sitk.GetArrayFromImage(post).astype(np.float64)
    positive = a_pre[a_pre > 0]
    if positive.size == 0:
        raise ValueError("The pre-contrast image has no signal. Are pre and post the right way round?")
    background = p["bg_fraction"] * np.percentile(positive, 99.5)

    valid = a_pre > background
    pe = np.zeros_like(a_pre)
    pe[valid] = (a_post[valid] - a_pre[valid]) / a_pre[valid] * 100.0
    cand = pe >= p["pe_threshold"]
    if side in ("left", "right"):
        x = _physical_x(post)
        midline = float(np.median(x[valid]))
        cand &= (x > midline) if side == "left" else (x < midline)
    if not cand.any():
        raise ValueError("No voxel enhances by %.0f%% or more. Check that the images are pre- and post-contrast of the same exam." % p["pe_threshold"])

    spacing = post.GetSpacing()
    vox_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    cand_img = sitk.GetImageFromArray(cand.astype(np.uint8))
    cand_img.CopyInformation(post)

    # Version 2 (tuned on the 50 patients, see autoseg_tune.py): an opening removes thin enhancing structures (vessels,
    # glandular speckle) that the closing of version 1 glued into one breast-sized cluster; the component with the
    # largest total enhancement is taken as the tumor. If the opening removes everything, version 1 is used.
    weights = np.clip(pe, 0, 300)
    opened = sitk.BinaryMorphologicalOpening(cand_img, _radius_vox(p["open_mm"], spacing))
    lab = sitk.GetArrayFromImage(sitk.ConnectedComponent(opened))
    if lab.max() > 0:
        n_clusters = int(lab.max())
        score = np.bincount(lab.ravel(), weights=weights.ravel(), minlength=n_clusters + 1)
        score[0] = 0
        main = int(score.argmax())
        share = float(score[main] / max(score.sum(), 1e-9))
    else:
        closed = sitk.BinaryMorphologicalClosing(cand_img, _radius_vox(p["close_mm"], spacing))
        lab = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(closed)))
        n_clusters = int(lab.max())
        counts = np.bincount(lab[cand & (lab > 0)], minlength=n_clusters + 1)
        counts[0] = 0
        main = int(counts.argmax())
        share = float(counts[main] / max(counts.sum(), 1))
    region = sitk.GetImageFromArray((lab == main).astype(np.uint8))
    region.CopyInformation(post)
    dist = sitk.SignedMaurerDistanceMap(region, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    near = sitk.GetArrayFromImage(dist) <= p["grow_mm"]
    mask = (cand & near).astype(np.uint8)

    volume = float(mask.sum() * vox_ml)
    info = {"tumor_voxels": int(mask.sum()), "tumor_volume_ml": volume, "n_enhancing_clusters": n_clusters,
            "main_cluster_share": share, "background_level": float(background),
            "params": p, "side": side, "warnings": []}
    if volume < p["min_volume_ml"]:
        info["warnings"].append("Very small segmented volume (%.2f mL): the automatic segmentation may have failed." % volume)
    if info["main_cluster_share"] < 0.6 and n_clusters > 1:
        info["warnings"].append("Several separate enhancing clusters; only the largest was used (share %.0f%%)." % (100 * info["main_cluster_share"]))
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(post)
    return out, info


def dice(a, b):
    a = sitk.GetArrayFromImage(a) > 0
    b = sitk.GetArrayFromImage(b) > 0
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")
