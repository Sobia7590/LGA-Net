#messidor-2
import os, gc, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve)
from sklearn.preprocessing import label_binarize
GRAPH_DIR    = os.path.join(BASE_DIR, "outputs_600dpi")
os.makedirs(GRAPH_DIR, exist_ok=True)
CLASS_NAMES  = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
colors       = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
# ── CORRECT PATHS ─────────────────────────────────────────────
MESSIDOR_DIR = r"C:\Users\Sobia Khan\Downloads\LGA_NET\Messidor-2"
MESSIDOR_IMG = os.path.join(MESSIDOR_DIR, "images", "IMAGES")
MESSIDOR_CSV = os.path.join(MESSIDOR_DIR, "messidor_data.csv")
# ── LOAD CSV ──────────────────────────────────────────────────
mess_df = pd.read_csv(MESSIDOR_CSV)
print("Columns:", list(mess_df.columns))
print(mess_df.head(3))
mess_df = mess_df[['id_code', 'diagnosis']].dropna()
mess_df['diagnosis'] = mess_df['diagnosis'].astype(int)
mess_df = mess_df[mess_df['diagnosis'].isin([0,1,2,3,4])].reset_index(drop=True)
print(f"\nMessidor-2 samples: {len(mess_df)}")
print(mess_df['diagnosis'].value_counts().sort_index())
# ── DATALOADER ────────────────────────────────────────────────
mess_ds     = Messidor2Dataset(mess_df, MESSIDOR_IMG, transform=val_transform)
mess_loader = DataLoader(mess_ds, batch_size=APTOS_BATCH,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print(f"✅ mess_loader ready — {len(mess_loader)} batches")
# ── LOAD MODEL ────────────────────────────────────────────────
ext_model = LGANet(num_classes=5).to(device)
ext_model.load_state_dict(torch.load(
    os.path.join(WORK_DIR,
        'lganet_stage2_best_stage2_aptos_full_finetune.pth'),
    map_location=device))
ext_model.eval()
print("✅ Model loaded.")
# ── TTA ───────────────────────────────────────────────────────
# NOTE: TTA is applied here for external validation only -- it was NOT used
# for APTOS internal CV / baselines, so this isn't apples-to-apples with
# those numbers. State this explicitly in the Methods section.
def tta_predict(model, images):
    augs = [images,
            torch.flip(images, dims=[3]),
            torch.flip(images, dims=[2]),
            torch.rot90(images, 1, [2, 3]),
            torch.rot90(images, 3, [2, 3])]
    preds = []
    for aug in augs:
        logits, _ = model(aug)
        preds.append(torch.softmax(logits, dim=1))
    return torch.stack(preds).mean(dim=0)
# ── INFERENCE ─────────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(mess_loader):
        imgs  = imgs.to(device)
        probs = tta_predict(ext_model, imgs)
        preds = torch.argmax(probs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(mess_loader)} done...")
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
print("✅ Inference complete.")
# ── METRICS ───────────────────────────────────────────────────
qwk_mess  = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
acc_mess  = (all_labels == all_preds).mean()
auc_mess  = roc_auc_score(all_labels, all_probs, multi_class='ovr')
cm_mess   = confusion_matrix(all_labels, all_preds)
cm_norm   = cm_mess.astype(float) / cm_mess.sum(axis=1, keepdims=True)
y_bin     = label_binarize(all_labels, classes=list(range(5)))
# FIX: guard against classes with zero samples in Messidor-2 (common for
# Severe/Proliferative in external DR datasets) -- roc_auc_score errors
# ("Only one class present in y_true") if a column has no positives.
per_auc = {}
classes_present = []
for i in range(5):
    if y_bin[:, i].sum() == 0:
        per_auc[CLASS_NAMES[i]] = float('nan')
        print(f"  Note: '{CLASS_NAMES[i]}' has 0 samples in Messidor-2 -- AUC not computable, reporting N/A")
        continue
    per_auc[CLASS_NAMES[i]] = roc_auc_score(y_bin[:, i], all_probs[:, i])
    classes_present.append(i)
print(f"\nQWK      : {qwk_mess:.4f}")
print(f"Accuracy : {acc_mess:.4f}")
print(f"AUC (OvR): {auc_mess:.4f}")
print("\nPer-class AUC:")
for cls, val in per_auc.items():
    print(f"  {cls:15s}: {val:.4f}" if not np.isnan(val) else f"  {cls:15s}: N/A (no samples)")
print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)}")
# ── FIGURE 1: Normalised CM ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_yticklabels(CLASS_NAMES)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                ha='center', va='center', fontsize=10)
ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
plt.tight_layout()
path = os.path.join(GRAPH_DIR, "lganet_messidor2_cm_norm.png")
plt.savefig(path, dpi=600, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {path}")
# ── FIGURE 2: Raw CM ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
im2 = ax.imshow(cm_mess, cmap='Blues')
plt.colorbar(im2, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_yticklabels(CLASS_NAMES)
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(cm_mess[i, j]),
                ha='center', va='center', fontsize=10)
ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
plt.tight_layout()
path = os.path.join(GRAPH_DIR, "lganet_messidor2_cm_raw.png")
plt.savefig(path, dpi=600, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {path}")
# ── FIGURE 3: ROC Curves ──────────────────────────────────────
# FIX: only plot classes that actually have positive samples -- roc_curve
# errors the same way roc_auc_score does for an absent class.
fig, ax = plt.subplots(figsize=(7, 6))
for i in classes_present:
    cls = CLASS_NAMES[i]
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    ax.plot(fpr, tpr, color=colors[i],
            label=f"{cls} (AUC={per_auc[cls]:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(GRAPH_DIR, "lganet_messidor2_roc.png")
plt.savefig(path, dpi=600, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {path}")
# ── SAVE UPDATED SUMMARY ──────────────────────────────────────
summary = {
    'dataset': 'Messidor-2', 'n_samples': len(all_labels),
    'qwk': round(qwk_mess, 4), 'accuracy': round(acc_mess, 4),
    'auc_ovr': round(auc_mess, 4),
    **{f'auc_{k.replace(" ","_")}': (round(v, 4) if not np.isnan(v) else None)
       for k, v in per_auc.items()}
}
pd.DataFrame([summary]).to_csv(
    os.path.join(WORK_DIR, 'lganet_messidor2_summary.csv'), index=False)
print("\n✅ All three figures and summary saved.")
del ext_model
gc.collect()
torch.cuda.empty_cache()
