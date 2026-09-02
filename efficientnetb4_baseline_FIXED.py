# ── BASELINE 3 — EfficientNet-B4 on APTOS (same split, same logging style) ──
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
from sklearn.metrics import cohen_kappa_score
# =========================================================
# CONFIG
# =========================================================
BASELINE_NAME = "efficientnetb4_aptos_baseline"
NUM_CLASSES = 5
# reproducibility — was missing
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
# =========================================================
# MODEL
# =========================================================
efficientnet_model = timm.create_model(
    'efficientnet_b4',
    pretrained=True,
    num_classes=NUM_CLASSES
).to(device)
total_params = sum(p.numel() for p in efficientnet_model.parameters())
trainable_params = sum(p.numel() for p in efficientnet_model.parameters() if p.requires_grad)
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
    efficientnet_model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)
EPOCHS = 25
PATIENCE = 7      # standardized — matches LGA-Net Stage 2 / CV / all other baselines
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
        efficientnet_model, aptos_train_loader, optimizer, criterion, device
    )
    val_loss, val_qwk = validate_one_epoch_baseline(
        efficientnet_model, aptos_val_loader, criterion, device
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
    # single criterion (val_qwk only) — no OR-logic with val_loss
    if val_qwk > best_val_qwk:
        best_val_qwk = val_qwk
        best_val_loss = val_loss
        no_improve = 0
        torch.save(efficientnet_model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved | Best Val QWK: {best_val_qwk:.4f} | Val Loss at this epoch: {val_loss:.4f}")
    else:
        no_improve += 1
        print(f"  No improvement {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            stopped_epoch = epoch
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break
print("=" * 65)
history_df = pd.DataFrame(history)
history_df.to_csv(history_csv_path, index=False)
print("History saved to:", history_csv_path)
print(f"\n{BASELINE_NAME} complete!")
print(f"Early stopping at epoch: {stopped_epoch}")
print(f"Best Val QWK:            {best_val_qwk:.4f}")
print(f"Best Val Loss (logged, not selection criterion): {best_val_loss:.4f}")
print(f"Model saved → {best_model_path}")
