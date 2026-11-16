# =====================================================================================================
# CNN experiments for pCR prediction (run on Google Colab, GPU: Runtime > Change runtime type > T4 GPU)
#
# Input : cnn_patches.npz (made by make_patches.py on the laptop). Upload it when the first cell asks.
# Output: AUC of three CNN based approaches, all validated at PATIENT level (a patient is never in train and
#         test at the same time) with repeated stratified 5-fold cross-validation:
#   A) ResNet18 (ImageNet pre-trained), last block fine-tuned, WITH data augmentation (training data only)
#   B) the same network WITHOUT augmentation  (to show what the augmentation does)
#   C) frozen ResNet18 "deep features" + PCA + Logistic Regression
#
# How to use: copy the whole file into ONE Colab cell and run it (or split at the "# ---- cell" lines).
# =====================================================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---- cell 1: load the patches
from google.colab import files
uploaded = files.upload()                       # choose cnn_patches.npz
d = np.load("cnn_patches.npz", allow_pickle=True)
P = d["patches"].astype(np.float32)             # (patients, 5 positions, 3 slices, 64, 64), values 0..1
y = d["labels"].astype(int)
N, S = P.shape[0], P.shape[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("patients:", N, "| pCR=1:", int(y.sum()), "| samples per patient:", S, "| device:", device)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
EPOCHS, BATCH, LR = 15, 16, 3e-4
N_REPEATS = 3


def prep(x):
    """x: (B,3,64,64) in 0..1  ->  ImageNet-normalised 112 x 112 input."""
    x = F.interpolate(x, size=112, mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def augment(x):
    """Random flip, rotation (+-15 deg), shift (+-3 px), brightness and contrast. Used on TRAINING batches only."""
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) * 2 - 1) * (15 * np.pi / 180)
    tx = (torch.rand(b, device=x.device) * 2 - 1) * 0.1
    ty = (torch.rand(b, device=x.device) * 2 - 1) * 0.1
    flip = (torch.rand(b, device=x.device) < 0.5).float() * 2 - 1
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = torch.cos(ang) * flip
    theta[:, 0, 1] = -torch.sin(ang)
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = torch.sin(ang) * flip
    theta[:, 1, 1] = torch.cos(ang)
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * 0.2
    bright = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    return (x * contrast + bright).clamp(0, 1)


def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():             # only the last block and the new classifier are trained
        p.requires_grad = True
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 1))
    return m.to(device)


def freeze_batchnorm(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def train_cnn(train_idx, use_aug, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = make_model()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)
    X = torch.tensor(P[train_idx].reshape(-1, 3, 64, 64))
    Y = torch.tensor(np.repeat(y[train_idx], S), dtype=torch.float32)
    n = len(X)
    for _ in range(EPOCHS):
        m.train()
        freeze_batchnorm(m)
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            if use_aug:
                xb = augment(xb)
            loss = F.binary_cross_entropy_with_logits(m(prep(xb)).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict_patients(m, idx):
    m.eval()
    X = torch.tensor(P[idx].reshape(-1, 3, 64, 64)).to(device)
    prob = torch.sigmoid(m(prep(X)).squeeze(1)).cpu().numpy().reshape(len(idx), S)
    return prob.mean(axis=1)                    # one score per patient = mean over its 5 positions


def run_cv(fit_predict, name):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=42)
    oof = np.zeros((N_REPEATS, N))
    for k, (tr, te) in enumerate(rskf.split(np.zeros(N), y)):
        oof[k // 5, te] = fit_predict(tr, te, k)
    aucs = np.array([roc_auc_score(y, oof[r]) for r in range(N_REPEATS)])
    pooled = oof.mean(axis=0)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        b = rng.integers(0, N, N)
        if len(set(y[b])) == 2:
            boot.append(roc_auc_score(y[b], pooled[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("%-52s AUC %.3f +/- %.3f   (95%% CI %.2f to %.2f)" % (name, aucs.mean(), aucs.std(), lo, hi))
    return aucs.mean()


# ---- cell 2: A) fine-tuned CNN with augmentation, B) without augmentation
print("\nRepeated stratified 5-fold CV at patient level, %d repeats (takes a few minutes on a GPU)\n" % N_REPEATS)
res = {}
res["A"] = run_cv(lambda tr, te, k: predict_patients(train_cnn(tr, True, k), te), "A) ResNet18 fine-tuned, with augmentation")
res["B"] = run_cv(lambda tr, te, k: predict_patients(train_cnn(tr, False, k), te), "B) ResNet18 fine-tuned, no augmentation")

# ---- cell 3: C) frozen ResNet18 deep features + classical classifier
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()
backbone = backbone.to(device).eval()
with torch.no_grad():
    allx = torch.tensor(P.reshape(-1, 3, 64, 64)).to(device)
    feats = torch.cat([backbone(prep(allx[i:i + 64])) for i in range(0, len(allx), 64)]).cpu().numpy()
deep = feats.reshape(N, S, -1).mean(axis=1)     # (patients, 512)


def deep_fit_predict(tr, te, k):
    clf = make_pipeline(StandardScaler(), PCA(n_components=10, random_state=0),
                        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    clf.fit(deep[tr], y[tr])
    return clf.predict_proba(deep[te])[:, 1]


res["C"] = run_cv(deep_fit_predict, "C) frozen ResNet18 features + PCA + Logistic Regression")

print("\nCompare with the radiomics models on the same 126 patients (Random Forest about 0.62).")
print("With only 126 patients a CNN is expected to be weaker or similar. Report whatever comes out.")
