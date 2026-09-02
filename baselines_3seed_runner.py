# ── 3-SEED reruns: ResNet50, MobileNetV2, EfficientNet-B4 ──
# Professor's review (§2) requires 5-fold CV OR at minimum 3 seeds for every
# baseline. Swin-Tiny already has 5-fold CV. This covers the other three at
# the minimum bar (3 seeds) instead of full CV, to meet the requirement
# without the multi-day time cost. Same recipe as each baseline's fixed
# single-run script: EPOCHS=25, PATIENCE=7, single-val_qwk checkpoint
# criterion, LR=5e-5, class-weighted label-smoothed CE.
import os, time, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.metrics import cohen_kappa_score, roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEEDS = [42, 123, 2024]
BASELINE_SPECS = {
    "resnet50":        dict(timm_name="resnet50",        img_size=IMG_SIZE, lr=5e-5),
    "mobilenetv2":     dict(timm_name="mobilenetv2_100",  img_size=IMG_SIZE, lr=5e-5),
    "efficientnet_b4": dict(timm_name="efficientnet_b4",  img_size=IMG_SIZE, lr=5e-5),
}
EPOCHS = 25
PATIENCE = 7

def run_one(name, spec, seed):
    set_seed(seed)
    model = timm.create_model(spec["timm_name"], pretrained=True, num_classes=5).to(device)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    class_counts = train_aptos['diagnosis'].value_counts().sort_index()
    cw = 1.0 / class_counts.values.astype(np.float32)
    cw = cw / cw.sum() * len(cw)
    cw = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_qwk, no_improve, best_eval = -1e9, 0, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        for imgs, labels in aptos_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds, labs, probs = [], [], []
        with torch.no_grad():
            for imgs, labels in aptos_val_loader:
                logits = model(imgs.to(device))
                probs.extend(torch.softmax(logits, 1).cpu().numpy())
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labs.extend(labels.numpy())
        val_qwk = cohen_kappa_score(labs, preds, weights='quadratic')
        scheduler.step()
        print(f"    [{name} seed={seed}] Epoch {epoch:02d}/{EPOCHS} | Val QWK: {val_qwk:.4f} | Time: {time.time()-t0:.1f}s")
        if val_qwk > best_qwk:
            best_qwk, no_improve = val_qwk, 0
            best_eval = (np.array(labs), np.array(preds), np.array(probs))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    [{name} seed={seed}] Early stopping at epoch {epoch}")
                break
    y_true, y_pred, y_prob = best_eval
    acc = (y_true == y_pred).mean()
    auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr')
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"config_key": name, "config": name, "seed": seed,
            "val_qwk": best_qwk, "val_acc": acc, "val_auc": auc_ovr, "params_M": params_M}

results = []
for name, spec in BASELINE_SPECS.items():
    print("=" * 65); print(f"{name} — 3-seed reruns"); print("=" * 65)
    for seed in SEEDS:
        results.append(run_one(name, spec, seed))

results_df = pd.DataFrame(results)
out_csv = os.path.join(WORK_DIR, "baselines_3seed_results.csv")
results_df.to_csv(out_csv, index=False)
print("\n" + results_df.round(4).to_string(index=False))
summary = results_df.groupby("config_key")["val_qwk"].agg(["mean", "std"])
print("\nSummary (mean +/- std across 3 seeds):")
print(summary.round(4).to_string())
print(f"\nSaved: {out_csv}")
