"""
=============================================================
ABLATION: Dual-Branch WITHOUT LGAG — seeds 42 / 123 / 2024
=============================================================
Extends the single-seed dual_branch_no_lgag run (seed=42, done,
best Val QWK 0.9008) to three seeds, matching the other six
ablation configs in Table 7 (each reported as mean ± SD over
3 seeds with bootstrap 95% CI), so this config is no longer the
odd one out on a single run.

Resumable: if dual_branch_no_lgag_seed{N}_history.csv already
exists in WORK_DIR, that seed is skipped rather than re-trained,
same pattern as the six-config ablation script.

Run:
    python dual_branch_no_lgag_multiseed.py

Outputs (saved to WORK_DIR):
    dual_branch_no_lgag_seed{42,123,2024}_best.pth
    dual_branch_no_lgag_seed{42,123,2024}_history.csv
    dual_branch_no_lgag_multiseed_results.csv   (summary, all seeds)
=============================================================
"""
import os
import time
import random
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# =============================================================
# ── CONFIGURATION — path resolution ──────────────────────────
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
os.makedirs(WORK_DIR, exist_ok=True)

for path_name, path_val in [("APTOS_IMG", APTOS_IMG), ("APTOS_CSV", APTOS_CSV)]:
    if not os.path.exists(path_val):
        raise FileNotFoundError(f"{path_name} does not exist: {path_val}")

# Hyperparameters — identical across all seeds, matched to LGA-Net Stage 2
IMG_SIZE     = 380
BATCH_SIZE   = 6
NUM_WORKERS  = min(8, (os.cpu_count() or 4))
EPOCHS       = 25
LR_MAIN      = 5e-6
WEIGHT_DECAY = 1e-4
PATIENCE     = 7
LABEL_SMOOTH = 0.05
NUM_CLASSES  = 5
TRIAL_BASE   = "dual_branch_no_lgag"

SEEDS = [42, 123, 2024]  # 42 is the already-completed run; skipped automatically below

print(f"NUM_WORKERS: {NUM_WORKERS} (os.cpu_count() = {os.cpu_count()})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================
# ── PREPROCESSING / TRANSFORMS / DATASET (identical to seed-42 run) ──
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
# ── MODEL (identical to seed-42 run) ──────────────────────────
# =============================================================
class EfficientNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('efficientnet_b4', pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((12, 12))
        self.proj = nn.Conv2d(1792, 512, kernel_size=1)
    def forward(self, x):
        return self.proj(self.pool(self.features(x)))

class SwinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
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

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
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
# ── DATA LOADING (built once; only the train/val split changes with seed) ──
# =============================================================
print("Loading APTOS CSV ...")
df = pd.read_csv(APTOS_CSV)
class_counts_all = df['diagnosis'].value_counts().sort_index()

results = []
summary_path = os.path.join(WORK_DIR, f"{TRIAL_BASE}_multiseed_results.csv")

for seed in SEEDS:
    trial_name = f"{TRIAL_BASE}_seed{seed}"
    history_csv_path = os.path.join(WORK_DIR, f"{trial_name}_history.csv")
    best_model_path = os.path.join(WORK_DIR, f"{trial_name}_best.pth")

    # seed=42 was run before this multiseed script existed, so its files use the
    # old no-suffix naming (dual_branch_no_lgag_history.csv / _best.pth), not the
    # _seed{N}_ pattern used below for the new seeds. Check that legacy path too.
    if seed == 42:
        legacy_history_path = os.path.join(WORK_DIR, f"{TRIAL_BASE}_history.csv")
        legacy_best_path = os.path.join(WORK_DIR, f"{TRIAL_BASE}_best.pth")
        if os.path.exists(legacy_history_path) and not os.path.exists(history_csv_path):
            history_csv_path = legacy_history_path
            best_model_path = legacy_best_path

    if os.path.exists(history_csv_path):
        # Resume: this seed already has a history file, don't retrain.
        hist = pd.read_csv(history_csv_path)
        best_qwk = hist['val_qwk'].max()
        print(f"Skipping seed={seed} — already completed (best Val QWK {best_qwk:.4f}).")
        results.append({"config_key": TRIAL_BASE, "seed": seed, "val_qwk": best_qwk})
        continue

    print("\n" + "=" * 65)
    print(f"Dual-Branch, no LGAG — seed={seed}")
    print("=" * 65)

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=seed)
    train_ds = APTOSDataset(train_df, APTOS_IMG, transform=train_transform)
    val_ds = APTOSDataset(val_df, APTOS_IMG, transform=val_transform)
    _loader_extra = {"persistent_workers": True, "prefetch_factor": 2} if NUM_WORKERS > 0 else {}
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, **_loader_extra)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, **_loader_extra)

    class_counts = train_df['diagnosis'].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    model = DualBranchNoLGAG(num_classes=NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_qwk, no_improve, stopped_epoch = -1e9, 0, EPOCHS
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_qwk": [], "val_qwk": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_qwk = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_qwk = run_epoch(model, val_loader, optimizer, criterion, train=False)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_qwk"].append(train_qwk)
        history["val_qwk"].append(val_qwk)
        history["lr"].append(lr)

        print(f"  [{trial_name}] Ep {epoch:02d}/{EPOCHS} | TrLoss {train_loss:.4f} | TrQWK {train_qwk:.4f} | "
              f"VaLoss {val_loss:.4f} | VaQWK {val_qwk:.4f} | LR {lr:.2e} | {time.time()-t0:.1f}s")

        if val_qwk > best_qwk:
            best_qwk, no_improve = val_qwk, 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✓ Best saved (Val QWK: {best_qwk:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                stopped_epoch = epoch
                print(f"    Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print(f"  seed={seed} done — best Val QWK={best_qwk:.4f}, stopped epoch {stopped_epoch}")
    results.append({"config_key": TRIAL_BASE, "seed": seed, "val_qwk": best_qwk})

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

# =============================================================
# ── SUMMARY ────────────────────────────────────────────────────
# =============================================================
results_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print(results_df)
print(f"\n{TRIAL_BASE} mean QWK: {results_df['val_qwk'].mean():.4f} +/- {results_df['val_qwk'].std():.4f} (n={len(results_df)} seeds)")
results_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
