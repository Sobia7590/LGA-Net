# ── PHASE 5B — Plot Stage 2 curves (loss and QWK) ──────────────────────────
import os
import pandas as pd
import matplotlib.pyplot as plt

TRIAL_NAME = "stage2_aptos_full_finetune"

# Hardcoded to the confirmed real path (found via `find /`), not built from
# WORK_DIR: WORK_DIR in this kernel currently resolves to a stale, literally-
# named "C:\Users\...\outputs_v2" directory (a duplicate from an earlier
# session) that does NOT contain the actual Stage 2 CSV. The real file is
# under /ssd4/ycheol/LGA_NET/outputs_v2/. If you rerun this after fixing
# WORK_DIR in the notebook, feel free to switch back to the os.path.join
# form below.
history_csv_path = "/ssd4/ycheol/LGA_NET/outputs_v2/lganet_stage2_history_stage2_aptos_full_finetune.csv"
# history_csv_path = os.path.join(WORK_DIR, f'lganet_stage2_history_{TRIAL_NAME}.csv')

if not os.path.exists(history_csv_path):
    raise FileNotFoundError(
        f"Not found: {history_csv_path}\n"
        f"Check that Stage 2 training actually saved a history CSV under this exact TRIAL_NAME."
    )
hist2 = pd.read_csv(history_csv_path)

best_idx = hist2['val_qwk'].idxmax()
best_epoch = int(hist2.loc[best_idx, 'epoch'])
best_qwk_val = hist2.loc[best_idx, 'val_qwk']

# ── Loss curve ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(hist2['epoch'], hist2['train_loss'], label='Train Loss', marker='o', markersize=3)
plt.plot(hist2['epoch'], hist2['val_loss'], label='Val Loss', marker='o', markersize=3)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.6,
            label=f'Best checkpoint (epoch {best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# No in-image title: BSPC's Guide for Authors requires the title to live in
# the figure caption text in the manuscript, not on the artwork itself
# ("A caption should consist of a brief title (not displayed on the figure
# itself) and a description of the image"). Add the caption in Word/LaTeX,
# not here.
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# Saving next to the CSV (real, confirmed-existing path) rather than
# GRAPH_DIR, since GRAPH_DIR is another notebook variable that may be
# similarly stale right now. Move the PNG afterward if you want it in
# GRAPH_DIR once that variable is confirmed correct.
loss_curve_path = os.path.join(os.path.dirname(history_csv_path), f'lganet_stage2_loss_curve_{TRIAL_NAME}.png')
plt.savefig(loss_curve_path, dpi=300)
plt.show()
print("Saved:", loss_curve_path)

# ── QWK curve ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(hist2['epoch'], hist2['train_qwk'], label='Train QWK', marker='o', markersize=3)
plt.plot(hist2['epoch'], hist2['val_qwk'], label='Val QWK', marker='o', markersize=3)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.6,
            label=f'Best checkpoint (epoch {best_epoch}, QWK={best_qwk_val:.4f})')
plt.xlabel('Epoch')
plt.ylabel('QWK')
# No in-image title — see note above.
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
qwk_curve_path = os.path.join(os.path.dirname(history_csv_path), f'lganet_stage2_qwk_curve_{TRIAL_NAME}.png')
plt.savefig(qwk_curve_path, dpi=300)
plt.show()
print("Saved:", qwk_curve_path)
