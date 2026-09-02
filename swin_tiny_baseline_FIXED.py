# ── BASELINE 4 — Swin-Tiny on APTOS (same split, same logging style) ──
# NOTE: unlike the other three baselines, Swin-Tiny needs its own dedicated
# 224x224 loaders (its own train/val DataLoader objects) rather than reusing
# the shared aptos_train_loader/aptos_val_loader built at IMG_SIZE (380) --
# swin_tiny_patch4_window7_224 expects 224x224 input for its window
# partitioning to line up correctly. Batch size also drops to 4 to match
# what was used for its 5-fold CV, since the smaller batch was needed for
# memory headroom at this resolution/model combination.
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
from torchvision import transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
# =========================================================
# CONFIG
# =========================================================
BASELINE_NAME = "swin_tiny_aptos_baseline"  # matches Phase 11's expected filename exactly
NUM_CLASSES = 5
SWIN_IMG_SIZE = 224
SWIN_BATCH = 4
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
# =========================================================
# DEDICATED 224x224 LOADERS
# =========================================================
swin_train_tf = T.Compose([
    T.Resize((SWIN_IMG_SIZE, SWIN_IMG_SIZE)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
swin_val_tf = T.Compose([
    T.Resize((SWIN_IMG_SIZE, SWIN_IMG_SIZE)),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
swin_train_loader = DataLoader(
    APTOSDataset(train_aptos, APTOS_IMG, swin_train_tf), batch_size=SWIN_BATCH,
    shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
swin_val_loader = DataLoader(
    APTOSDataset(val_aptos, APTOS_IMG, swin_val_tf), batch_size=SWIN_BATCH,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
# =========================================================
# MODEL
# =========================================================
swin_model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=NUM_CLASSES
).to(device)
total_params = sum(p.numel() for p in swin_model.parameters())
trainable_params = sum(p.numel() for p in swin_model.parameters() if p.requires_grad)
print(f"{BASELINE_NAME} total params: {total_params:,}")
print(f"{BASELINE_NAME} trainable params: {trainable_params:,}")
# =========================================================
# CLASS WEIGHTS
# =========================================================
class_counts = train_aptos['diagnosis'].value_counts().sort_index()
print("\nAPTOS train class counts:")
print(class_counts)
class_weights = 1.0 / class_counts.values.astype(np.float32)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("\nClass weights:")
print(class_weights)
# =========================================================
# LOSS / OPTIMIZER / SCHEDULER
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    swin_model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)
EPOCHS = 25
PATIENCE = 7      # standardized — matches LGA-Net Stage 2 / CV / all other baselines
                   # (NOT 5 — that was the original, uncorrected value)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,   # tied to EPOCHS so it always matches the actual budget
    eta_min=1e-6
)
# =========================================================
# TRAIN / VALIDATE FUNCTIONS
# =========================================================
def train_one_epoch_baseline(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / len(loader)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, qwk
def validate_one_epoch_baseline(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / len(loader)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, qwk
# =========================================================
# TRAINING LOOP
# =========================================================
print("\n" + "=" * 65)
print(f"TRAINING BASELINE — {BASELINE_NAME}")
print("=" * 65)
best_val_qwk = -1e9
best_val_loss = 1e9   # logged only now, does not drive checkpointing
no_improve = 0
stopped_epoch = EPOCHS
history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'train_qwk': [], 'val_qwk': [], 'lr': []
}
best_model_path = os.path.join(WORK_DIR, f"{BASELINE_NAME}_best.pth")
history_csv_path = os.path.join(WORK_DIR, f"{BASELINE_NAME}_history.csv")
for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss, train_qwk = train_one_epoch_baseline(
        swin_model, swin_train_loader, optimizer, criterion, device
    )
    val_loss, val_qwk = validate_one_epoch_baseline(
        swin_model, swin_val_loader, criterion, device
    )
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_qwk'].append(train_qwk)
    history['val_qwk'].append(val_qwk)
    history['lr'].append(current_lr)
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f} | "
        f"LR: {current_lr:.6f} | Time: {time.time() - start:.1f}s"
    )
    if val_qwk > best_val_qwk:
        best_val_qwk = val_qwk
        best_val_loss = val_loss   # logged only, not a save trigger
        no_improve = 0
        torch.save(swin_model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (Val QWK: {best_val_qwk:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            stopped_epoch = epoch
            print(f"\n  Early stopping at epoch {epoch}")
            break
print("=" * 65)
history_df = pd.DataFrame(history)
history_df.to_csv(history_csv_path, index=False)
print("History saved to:", history_csv_path)
print(f"\n{BASELINE_NAME} training complete!")
print(f"Stopped epoch: {stopped_epoch}")
print(f"Best Val QWK:  {best_val_qwk:.4f}")
print(f"Model saved → {best_model_path}")
