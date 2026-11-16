# =====================================================================================================
# Download baseline (T0) DICOM + analysis masks of 68 NEW randomly chosen I-SPY2 / ACRIN-6698 patients
#   56 "dev"  -> added to training/development (70 + 56 = 126)
#   12 "test" -> final untouched test (6 pCR + 6 non-pCR), used ONCE at the very end
# Run in Google Colab (whole file in ONE cell). Upload ispy2_new_batch_68.csv when asked.
# Output: ZIP files of 10 patients each (dev_batch1..., test_batch1...) downloaded directly to your laptop
# (about 6.3 GB in total; no Google Drive needed). Free space needed on the laptop: ~13 GB (zips + extracted).
# =====================================================================================================
import os, shutil
import pandas as pd
from google.colab import files

os.system("pip install idc-index -q")
from idc_index import IDCClient

files.upload()                                              # choose ispy2_new_batch_68.csv
pick = pd.read_csv("ispy2_new_batch_68.csv")
pick["Case_ID"] = pick["Case_ID"].astype(str)
print("Patients selected:", len(pick))
print(pd.crosstab([pick["role"], pick["subtype"]], pick["pCR"]))

client = IDCClient()
df = client.index
names = ["ISPY2-" + c for c in pick["Case_ID"]] + ["ACRIN-6698-" + c for c in pick["Case_ID"]]
base = df[df["collection_id"].isin(["ispy2", "acrin_6698"]) & df["PatientID"].isin(names)
          & df["StudyDescription"].str.endswith("MRI_T0")]
d = base[(base["Modality"] == "MR") & base["SeriesDescription"].str.contains("original DCE")]
print("Baseline DCE series found:", len(d), "| size (GB): %.2f" % (d["series_size_MB"].sum() / 1024))
seg = base[(base["Modality"] == "SEG") & base["SeriesDescription"].str.contains("mask", case=False, na=False)]
seg = seg[seg["PatientID"].isin(d["PatientID"])].drop_duplicates("PatientID")
print("Analysis masks found:", len(seg), "| patients:", seg["PatientID"].nunique())
print(seg["SeriesDescription"].value_counts())

out = "/content/new68_dicom"
shutil.rmtree(out, ignore_errors=True)
for part in (d, seg):
    client.download_from_selection(downloadDir=out, seriesInstanceUID=list(part["SeriesInstanceUID"]),
                                   dirTemplate="%PatientID/%Modality", use_s5cmd_sync=True)

# labels (keep them, do not look at the test labels until the final test)
d = d.assign(Case_ID=d["PatientID"].str.split("-").str[-1])
lab = pick.merge(d[["Case_ID", "PatientID"]].drop_duplicates(), on="Case_ID")[["PatientID", "subtype", "pCR", "role"]]
lab.to_csv("new68_labels.csv", index=False)

# ---- ZIPs of 10 patients each, downloaded straight to your laptop (no Google Drive needed) ----
import time
os.makedirs("/content/zips68", exist_ok=True)
for role in ("dev", "test"):
    ids = sorted(lab[lab.role == role]["PatientID"])
    ids = [p for p in ids if os.path.isdir(os.path.join(out, p))]
    lab[lab.role == role].to_csv("%s_labels.csv" % role, index=False)
    files.download("%s_labels.csv" % role)
    for i in range(0, len(ids), 10):
        batch = ids[i:i + 10]
        tmp = "/content/batch_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        for p in batch:
            shutil.copytree(os.path.join(out, p), os.path.join(tmp, p))
        name = "%s_batch%d" % (role, i // 10 + 1)
        z = shutil.make_archive("/content/zips68/" + name, "zip", tmp)
        print(name, len(batch), "patients:", z)
        files.download(z)                    # if Chrome asks, click "Allow" for multiple downloads
        time.sleep(60)                       # give the browser time to finish each ~1 GB file
print("Done. Check that you have dev_batch1..6 and test_batch1..2 (and the two label CSVs).")
