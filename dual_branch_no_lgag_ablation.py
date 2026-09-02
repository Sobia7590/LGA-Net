"""
=============================================================
ABLATION: Dual-Branch WITHOUT Lesion-Guided Attention Gate
=============================================================
This script trains the dual-branch (EfficientNet-B4 + Swin-Tiny)
fusion model WITHOUT the LGAG, providing the missing ablation
data point.

Differences from full LGA-Net:
  - No Stage 1 IDRiD pre-training (no attention gate to pre-train)
  - DualBranchNoLGAG replaces LGANet: fused features go directly
    to classifier without any attention gate
  - Single training stage on APTOS 2019 (identical to Stage 2)
  - Same hyperparameters, same split, same seed as full LGA-Net
    Stage 2: EPOCHS=25, PATIENCE=7, LR_MAIN=5e-6 (matches Stage 2's
    main_params group exactly, since this model's single param
    group is everything except the attention gate, which doesn't
    exist here).

Run:
    python dual_branch_no_lgag_ablation.py

Outputs (saved to WORK_DIR):
    dual_branch_no_lgag_best.pth
    dual_branch_no_lgag_history.csv
    dual_branch_no_lgag_results.txt
    dual_branch_no_lgag_curves.png
    dual_branch_no_lgag_confusion_matrix.png
    dual_branch_no_lgag_roc_curves.png
=============================================================
"""
import os
import gc
import time
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# =============================================================
# ── CONFIGURATION — path resolution ──────────────────────────
# =============================================================
# FIX: the previous version hardcoded BASE_DIR to a Windows-style
# path (r"C:\Users\Sobia Khan\Downloads\LGA_NET"). On the remote
# Linux training server, backslashes are NOT path separators, so
# that whole string resolves to one literal, oddly-named directory
# rather than a real nested path — this is the exact bug that made
# the Stage 2 history CSV "disappear" earlier in this project (the
# real file was under /ssd4/ycheol/LGA_NET/outputs_v2/, confirmed
# via `find /`, while WORK_DIR pointed at a stale duplicate).
#
# Rather than hardcode a second guessed path and risk the same
# failure mode again, this block checks a short list of candidates
# and tells you exactly which one it picked — or exactly what it
# searched, if none matched, so you're never debugging a silent
# wrong-directory run.
CANDIDATE_BASE_DIRS = [
    "/ssd4/ycheol/LGA_NET",                 # confirmed real path on this server
    os.path.expanduser("~/LGA_NET"),
    r"C:\Users\Sobia Khan\Downloads\LGA_NET",  # original Windows-style path, kept as last resort
]

BASE_DIR = None
for candidate in CANDIDATE_BASE_DIRS:
    if os.path.isdir(os.path.join(candidate, "aptos2019-blindness-detection")):
        BASE_DIR = candidate
        break

if BASE_DIR is None:
    searched = "\n  - ".join(CANDIDATE_BASE_DIRS)
    raise FileNotFoundError(
        "Could not find 'aptos2019-blindness-detection' under any of:\n  - "
        + searched
        + "\n\nRun this in a notebook cell to locate it, then hardcode BASE_DIR above:\n"
        "  import subprocess\n"
        "  print(subprocess.run(['find', '/', '-maxdepth', '6', '-iname', "
        "'aptos2019-blindness-detection'], capture_output=True, text=True).stdout)"
    )

print(f"Using BASE_DIR: {BASE_DIR}")

APTOS_IMG  = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images")
APTOS_CSV  = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train.csv")
WORK_DIR   = os.path.join(BASE_DIR, "outputs_v2")

for path_name, path_val in [("APTOS_IMG", APTOS_IMG), ("APTOS_CSV", APTOS_CSV)]:
    if not os.path.exists(path_val):
        raise FileNotFoundError(f"{path_name} does not exist: {path_val}")

# Hyperparameters — matched to LGA-Net Stage 2 exactly
IMG_SIZE      = 380
BATCH_SIZE    = 6
# FIX (performance only, does not change training math or results):
# was NUM_WORKERS = 0, which forces all per-image CPU preprocessing
# (disk read, crop, resize, circular mask, CLAHE) to run synchronously
# on the main thread, so the GPU sits idle between batches. Unlike
# BATCH_SIZE, this is purely about data-loading parallelism and has
# no effect on gradients, loss, or reproducibility (same images, same
# order per epoch under the same seed) — safe to change anytime,
# including for reruns of this exact ablation.
# Computed from os.cpu_count() rather than hardcoded, so it adapts to
# whichever machine actually runs this, capped at 8 to leave headroom
# for the OS and any other jobs on a shared server.
NUM_WORKERS   = min(8, (os.cpu_count() or 4))
EPOCHS        = 25            # matches LGA-Net Stage 2
LR_MAIN       = 5e-6          # matches LGA-Net's corrected main_params LR
WEIGHT_DECAY  = 1e-4
# FIX: was PATIENCE = 10, which matched neither this docstring's stated
# target nor LGA-Net Stage 2's actual patience. Stage 2 uses PATIENCE = 7
# (confirmed from the Stage 2 training code); set to 7 here so this
# ablation is genuinely apples-to-apples with the reference model.
PATIENCE      = 7
LABEL_SMOOTH  = 0.05
SEED          = 42
NUM_CLASSES   = 5
TRIAL_NAME    = "dual_branch_no_lgag"

# =============================================================
# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(WORK_DIR, exist_ok=True)

# =============================================================
# ── PREPROCESSING (identical to original notebook) ───────────
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

# =============================================================
# ── TRANSFORMS (identical to original notebook) ──────────────
# =============================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.90, 1.00)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================================================
# ── DATASET (identical to original notebook) ─────────────────
# =============================================================
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f'{row["id_code"]}.png')
        img      = preprocess_retinal(img_path, size=IMG_SIZE)
        img      = Image.fromarray((img * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, int(row['diagnosis'])

# =============================================================
# ── MODEL: Dual-Branch WITHOUT LGAG ──────────────────────────
# Identical backbone branches and fusion to LGA-Net.
# The ONLY difference: attended features = fused features
# (no attention gate, no spatial reweighting).
# =============================================================
class EfficientNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('efficientnet_b4', pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d((12, 12))
        self.proj     = nn.Conv2d(1792, 512, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return x

class SwinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, features_only=True
        )
        self.proj = nn.Conv2d(768, 512, kernel_size=1)

    def forward(self, x):
        x_swin = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats  = self.swin(x_swin)
        x      = feats[-1]
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(12, 12), mode='bilinear', align_corners=False)
        x = self.proj(x)
        return x

class DualBranchNoLGAG(nn.Module):
    """
    EfficientNet-B4 + Swin-Tiny dual-branch fusion WITHOUT
    the Lesion-Guided Attention Gate. Features flow:
        CNN branch  ──┐
                      ├─► concat ─► fusion conv ─► GAP ─► Dropout ─► classifier
        Swin branch ──┘
    No attention gate. No lesion supervision. No IDRiD pre-training.
    Parameter count ≈ 47.0M (full LGA-Net = 48.38M; difference is LGAG).
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.efficientnet = EfficientNetBranch()
        self.swin         = SwinBranch()
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat  = self.efficientnet(x)
        swin_feat = self.swin(x)
        fused     = torch.cat([cnn_feat, swin_feat], dim=1)
        fused     = self.fusion(fused)
        logits    = self.classifier(fused)          # ← direct, no attention
        return logits

# =============================================================
# ── DATA LOADING ─────────────────────────────────────────────
# =============================================================
print("Loading APTOS CSV ...")
df = pd.read_csv(APTOS_CSV)
print(f"Total samples: {len(df)}")
print("Class distribution:\n", df['diagnosis'].value_counts().sort_index())

# Identical 80/20 stratified split with seed=42
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['diagnosis'], random_state=SEED
)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

train_ds = APTOSDataset(train_df, APTOS_IMG, transform=train_transform)
val_ds   = APTOSDataset(val_df,   APTOS_IMG, transform=val_transform)

print(f"NUM_WORKERS: {NUM_WORKERS} (os.cpu_count() = {os.cpu_count()})")

# persistent_workers keeps the worker pool alive between epochs instead of
# respawning it each time (only valid when NUM_WORKERS > 0); prefetch_factor
# lets each worker stage a couple of batches ahead of the GPU.
_loader_extra = {"persistent_workers": True, "prefetch_factor": 2} if NUM_WORKERS > 0 else {}

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, **_loader_extra)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, **_loader_extra)

# =============================================================
# ── LOSS WITH CLASS WEIGHTS (identical to Stage 2) ───────────
# =============================================================
class_counts  = train_df['diagnosis'].value_counts().sort_index()
class_weights = 1.0 / class_counts.values.astype(np.float32)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("\nClass weights:", class_weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

# =============================================================
# ── MODEL + OPTIMIZER ────────────────────────────────────────
# =============================================================
model = DualBranchNoLGAG(num_classes=NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params/1e6:.2f}M")

# Single lr group (no gate params) — matches LGA-Net's main_params lr
optimizer = AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# =============================================================
# ── TRAIN / VALIDATE FUNCTIONS ───────────────────────────────
# =============================================================
def run_epoch(model, loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return total_loss / len(loader), qwk

# =============================================================
# ── TRAINING LOOP ────────────────────────────────────────────
# =============================================================
print("\n" + "=" * 65)
print(f"DUAL-BRANCH WITHOUT LGAG — Training on APTOS 2019")
print("=" * 65)

best_qwk       = -1e9
no_improve     = 0
stopped_epoch  = EPOCHS
history        = {"epoch":[], "train_loss":[], "val_loss":[],
                  "train_qwk":[], "val_qwk":[], "lr":[]}

best_model_path  = os.path.join(WORK_DIR, f"{TRIAL_NAME}_best.pth")
history_csv_path = os.path.join(WORK_DIR, f"{TRIAL_NAME}_history.csv")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_qwk = run_epoch(model, train_loader, optimizer, train=True)
    val_loss,   val_qwk   = run_epoch(model, val_loader,   optimizer, train=False)
    lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_qwk"].append(train_qwk)
    history["val_qwk"].append(val_qwk)
    history["lr"].append(lr)

    elapsed = time.time() - t0
    print(f"Ep {epoch:02d}/{EPOCHS} | "
          f"TrLoss {train_loss:.4f} | TrQWK {train_qwk:.4f} | "
          f"VaLoss {val_loss:.4f} | VaQWK {val_qwk:.4f} | "
          f"LR {lr:.2e} | {elapsed:.1f}s")

    if val_qwk > best_qwk:
        best_qwk   = val_qwk
        no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"   ✓ Best saved | Val QWK: {best_qwk:.4f}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            stopped_epoch = epoch
            print(f"\n  Early stopping at epoch {epoch}")
            break

print("=" * 65)
print(f"Best Val QWK:  {best_qwk:.4f}")
print(f"Stopped epoch: {stopped_epoch}")

# Save history
pd.DataFrame(history).to_csv(history_csv_path, index=False)
print(f"History → {history_csv_path}")

# =============================================================
# NOTE: the original script's docstring lists additional outputs
# (dual_branch_no_lgag_results.txt, _curves.png, _confusion_matrix.png,
# _roc_curves.png) that were not included in what was pasted for review.
# If you have that portion of the script, send it and it can be checked
# and appended here the same way the rest of this file was verified.
# =============================================================
