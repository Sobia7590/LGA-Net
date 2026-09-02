"""
=====================================================================
dual_branch_no_lgag — pooled OOF predictions + bootstrap 95% CI
=====================================================================
Table 7's CI column for the other six ablation configs is a bootstrap
over each config's POOLED out-of-fold predictions (all 3 seeds'
validation labels/preds concatenated into one array, then 2000
resamples, matching Phase 7D / bootstrap_qwk_ci in the notebook) --
NOT a bootstrap over the 3 seed-mean QWK scalars. To add this config
to Table 7 on equal footing, we need that same pooled-OOF bootstrap.

This script does NOT retrain. It reloads the three already-saved best
checkpoints:
    dual_branch_no_lgag_best.pth          (seed 42, legacy no-suffix name)
    dual_branch_no_lgag_seed123_best.pth
    dual_branch_no_lgag_seed2024_best.pth
rebuilds each seed's identical validation split (same train_test_split
call, same random_state=seed as the training script), runs one
inference pass to recover per-sample labels/preds/probs, pools all
three seeds, and bootstraps QWK exactly like Phase 7D.

Run:
    python dual_branch_no_lgag_bootstrap_ci.py

Outputs (saved to WORK_DIR):
    dual_branch_no_lgag_multiseed_oof.npz   (pooled labels/preds/probs)
    dual_branch_no_lgag_bootstrap_ci.csv    (mean_qwk, ci_lo, ci_hi)
=====================================================================
"""
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# =============================================================
# ── CONFIGURATION — path resolution (identical to the training script) ──
# =============================================================
CANDIDATE_BASE_DIRS = [
    "/ssd4/ycheol/LGA_NET",
    os.path.expanduser("~/LGA_NET"),
    r"C:\Users\Sobia Khan\Downloads\LGA_NET",
]
BASE_DIR = None
for candidate in CANDIDATE_BASE_DIRS:
    if os.path.isdir(os.path.join(candidate, "aptos2019-blindness-detection")):
        BASE_DIR = candidate
        break
if BASE_DIR is None:
    searched = "\n  - ".join(CANDIDATE_BASE_DIRS)
    raise FileNotFoundError(f"Could not find 'aptos2019-blindness-detection' under any of:\n  - {searched}")
print(f"Using BASE_DIR: {BASE_DIR}")

APTOS_IMG = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")
APTOS_CSV = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train.csv")
WORK_DIR  = os.path.join(BASE_DIR, "outputs_v2")

IMG_SIZE    = 380
BATCH_SIZE  = 6
NUM_WORKERS = min(8, (os.cpu_count() or 4))
NUM_CLASSES = 5
TRIAL_BASE  = "dual_branch_no_lgag"

# seed -> checkpoint path. seed 42 uses the legacy no-suffix name (that run
# predates this multiseed script); 123/2024 use the _seed{N}_ pattern the
# multiseed training script actually saved.
CHECKPOINTS = {
    42:   os.path.join(WORK_DIR, f"{TRIAL_BASE}_best.pth"),
    123:  os.path.join(WORK_DIR, f"{TRIAL_BASE}_seed123_best.pth"),
    2024: os.path.join(WORK_DIR, f"{TRIAL_BASE}_seed2024_best.pth"),
}
for seed, ckpt in CHECKPOINTS.items():
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint for seed={seed}: {ckpt}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================
# ── PREPROCESSING / DATASET (identical to the training script) ───────────
# =============================================================
def crop_black_border(img_rgb, tol=10):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img_rgb
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_rgb[y0:y1, x0:x1]

def circular_retina_mask(img_rgb, radius_scale=0.48):
    h, w, _ = img_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(radius_scale * min(h, w)), 255, -1)
    return cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

def enhance_green_channel(img_rgb):
    img = img_rgb.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    return img

def preprocess_retinal(path, size=380):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = crop_black_border(img, tol=10)
    img = cv2.resize(img, (size, size))
    img = circular_retina_mask(img, radius_scale=0.48)
    img = enhance_green_channel(img)
    return img.astype(np.float32) / 255.0

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df, self.img_dir, self.transform = df.reset_index(drop=True), img_dir, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f'{row["id_code"]}.png')
        img = preprocess_retinal(img_path, size=IMG_SIZE)
        img = Image.fromarray((img * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, int(row['diagnosis'])

# =============================================================
# ── MODEL (identical to the training script) ─────────────────
# =============================================================
class EfficientNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('efficientnet_b4', pretrained=False)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((12, 12))
        self.proj = nn.Conv2d(1792, 512, kernel_size=1)
    def forward(self, x):
        return self.proj(self.pool(self.features(x)))

class SwinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, features_only=True)
        self.proj = nn.Conv2d(768, 512, kernel_size=1)
    def forward(self, x):
        x_swin = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.swin(x_swin)
        x = feats[-1]
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(12, 12), mode='bilinear', align_corners=False)
        return self.proj(x)

class DualBranchNoLGAG(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.efficientnet = EfficientNetBranch()
        self.swin = SwinBranch()
        self.fusion = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        fused = torch.cat([self.efficientnet(x), self.swin(x)], dim=1)
        return self.classifier(self.fusion(fused))

@torch.no_grad()
def infer_val_split(seed, ckpt_path):
    df = pd.read_csv(APTOS_CSV)
    _, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=seed)
    val_ds = APTOSDataset(val_df, APTOS_IMG, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DualBranchNoLGAG(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    probs_arr = np.concatenate(all_probs)
    qwk = cohen_kappa_score(labels_arr, preds_arr, weights='quadratic')
    print(f"  seed={seed}: reconstructed Val QWK={qwk:.4f} (n={len(labels_arr)}) — cross-check against the training log's own best-epoch QWK.")

    del model
    torch.cuda.empty_cache()
    return labels_arr, preds_arr, probs_arr

# =============================================================
# ── RUN INFERENCE FOR EACH SEED, POOL, SAVE .npz ──────────────
# =============================================================
print("\nReconstructing per-seed validation predictions from saved checkpoints (no retraining) ...")
runs = []
for seed, ckpt in CHECKPOINTS.items():
    labels_arr, preds_arr, probs_arr = infer_val_split(seed, ckpt)
    runs.append((labels_arr, preds_arr, probs_arr))

labels_all = np.concatenate([r[0] for r in runs])
preds_all = np.concatenate([r[1] for r in runs])
probs_all = np.concatenate([r[2] for r in runs])

oof_path = os.path.join(WORK_DIR, f"{TRIAL_BASE}_multiseed_oof.npz")
np.savez(oof_path, labels=labels_all, preds=preds_all, probs=probs_all)
print(f"\nSaved: {oof_path}")

# =============================================================
# ── BOOTSTRAP 95% CI (identical method/params to Phase 7D) ───
# =============================================================
def bootstrap_qwk_ci(labels, preds, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_qwks = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_qwks[b] = cohen_kappa_score(labels[idx], preds[idx], weights='quadratic')
    lo, hi = np.percentile(boot_qwks, [2.5, 97.5])
    return boot_qwks.mean(), lo, hi

mean_qwk, lo, hi = bootstrap_qwk_ci(labels_all, preds_all)
print(f"\nBootstrap 95% CI on pooled (3-seed) QWK, n_boot=2000, seed=42:")
print(f"  dual_branch_no_lgag: {mean_qwk:.4f} [{lo:.4f}, {hi:.4f}]")

ci_df = pd.DataFrame([{"config_key": TRIAL_BASE, "qwk_mean": mean_qwk, "ci_lo": lo, "ci_hi": hi}])
ci_csv = os.path.join(WORK_DIR, f"{TRIAL_BASE}_bootstrap_ci.csv")
ci_df.to_csv(ci_csv, index=False)
print(f"Saved: {ci_csv}")

# Reference check: does this CI overlap Full LGA-Net's reference CI, [0.8955, 0.9142]
# (already reported in Table 7)?
ref_lo, ref_hi = 0.8955, 0.9142
overlaps = not (hi < ref_lo or lo > ref_hi)
print(f"\nOverlaps Full LGA-Net reference CI [{ref_lo}, {ref_hi}]? {'Yes' if overlaps else 'No'}")
