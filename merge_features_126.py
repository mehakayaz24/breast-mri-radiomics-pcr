"""Join the features of the 70 patients (retrain_70) and the 56 new development patients -> 126 patients. I-SPY1 and the 12 test patients are NOT included."""
import pandas as pd
OLD = r"C:\Users\ntech\Desktop\retrain_70\radiomics_features_70.csv"
NEW = r"C:\Users\ntech\Desktop\retrain_126\features_dev56.csv"
OUT = r"C:\Users\ntech\Desktop\retrain_126\radiomics_features_126.csv"
a = pd.read_csv(OLD)
b = pd.read_csv(NEW)
print("old:", len(a), "| new:", len(b))
both = set(a["Patient_ID"]) & set(b["Patient_ID"])
assert not both, "same patient in both files: %s" % both
cols = [c for c in a.columns if c in b.columns]
df = pd.concat([a[cols], b[cols]], ignore_index=True)
df.to_csv(OUT, index=False)
print("Saved %d patients to %s" % (len(df), OUT))
print(df.groupby("Dataset").size().to_string())
