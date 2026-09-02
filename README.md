# LGA-Net: Lesion-Guided Attention Network for Diabetic Retinopathy Grading

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-green)](https://www.kaggle.com/c/aptos2019-blindness-detection)
[![QWK](https://img.shields.io/badge/QWK-0.9049%C2%B10.0069-brightgreen)](#results-on-aptos-2019)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Official implementation of **LGA-Net**, a dual-branch CNN-Transformer architecture with a Lesion-Guided Attention Gate (LGAG), cross-dataset mask-supervised via IDRiD before fine-tuning on APTOS 2019.

---

## Overview

LGA-Net fuses:

- **EfficientNet-B4** (local lesion texture features)
- **Swin-Tiny Transformer** (hierarchical global context)

through a **Lesion-Guided Attention Gate**, warm-started with pixel-level lesion supervision from IDRiD, then fine-tuned end-to-end on APTOS 2019.

**Headline result:** QWK = **0.9049 ± 0.0069** under 5-fold stratified cross-validation on APTOS 2019 (out-of-fold pooled predictions), statistically indistinguishable from a Swin-Tiny-only backbone (paired Wilcoxon, p = 0.3125) and significantly ahead of ResNet-50, EfficientNet-B4, and MobileNetV2 (all p < 0.05).

A six-configuration, three-seed ablation shows no single design choice, including mask supervision itself, drives that QWK on its own — every ablated configuration's 95% CI overlaps the full model's. The paper's contribution is therefore framed around what the ablation *does* support: quantitative interpretability evidence (attention-lesion overlap, deletion/insertion testing) rather than a raw-accuracy claim. See the paper for the full statistical treatment.

---

## Results on APTOS 2019

Single-run comparison across all five architectures, identical data pipeline and training protocol:

| Model | Params (M) | QWK | Accuracy | AUC (OvR) |
|---|---|---|---|---|
| **LGA-Net (Ours)** | 48.4 | **0.9008** | **0.8349** | 0.8738 |
| Swin-Tiny | 27.5 | 0.8944 | 0.8213 | 0.8411 |
| ResNet-50 | 23.5 | 0.8773 | 0.7926 | **0.9049** |
| EfficientNet-B4 | 17.6 | 0.8698 | 0.7858 | 0.8779 |
| MobileNetV2 | 2.2 | 0.8395 | 0.7681 | 0.8789 |

Primary evaluation, 5-fold stratified cross-validation (out-of-fold pooled): **QWK 0.9049 ± 0.0069, Accuracy 0.8263 ± 0.0120, AUC 0.8908 ± 0.0254** (n = 5 folds).

Parameter counts reflect each architecture's actual 5-class classification head used in this study (measured directly via `sum(p.numel() for p in model.parameters())`), not the standard 1000-class ImageNet head commonly cited elsewhere.

### Ablation Study

Three seeds per configuration, bootstrap 95% CI (pooled per-sample predictions, 2000 resamples). Full model's own CI: [0.8955, 0.9142].

| Configuration | QWK (mean ± SD) | 95% CI | Overlaps full model? |
|---|---|---|---|
| **Full LGA-Net (reference)** | **0.9049 ± 0.0069** | [0.8955, 0.9142] | — |
| No LGAG (dual-branch fusion, no attention gate) | 0.9129 ± 0.0136 | [0.9017, 0.9231] | Yes |
| No mask supervision (λ = 0) | 0.9006 ± 0.0015 | [0.8878, 0.9127] | Yes |
| No Stage 1 warm-start | 0.9018 ± 0.0020 | [0.8885, 0.9145] | Yes |
| Swin-Tiny + LGAG only (no CNN branch) | 0.9026 ± 0.0033 | [0.8896, 0.9146] | Yes |
| λ = 0.15 | 0.9007 ± 0.0043 | [0.8870, 0.9136] | Yes |
| λ = 0.5 | 0.9042 ± 0.0074 | [0.8911, 0.9162] | Yes |
| λ = 1.0 | 0.9011 ± 0.0068 | [0.8880, 0.9132] | Yes |

No ablated configuration is statistically distinguishable from the full model on raw QWK. LGA-Net's demonstrated contribution is on interpretability (attention-lesion overlap, deletion/insertion testing), not raw grading accuracy — see the paper's Discussion.

---

## Installation

```bash
git clone https://github.com/Sobia7590/LGA-Net.git
cd LGA-Net
pip install -r requirements.txt
```

---

## Dataset Setup

### APTOS 2019

Download from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) and place as:

```
LGA_NET/
└── aptos2019-blindness-detection/
    ├── train_images/
    └── train.csv
```

### IDRiD (for Stage 1 attention pre-training)

Download from [IDRiD Grand Challenge](https://idrid.grand-challenge.org/) and place as:

```
LGA_NET/
└── IDRID_DATASET/
    └── A. Segmentation/
        └── A. Segmentation/
            ├── 1. Original Images/
            └── 2. All Segmentation Groundtruths/
```

### Messidor-2 (external validation only)

Used only for zero-shot external evaluation; never enters training. Grade labels follow Krause et al., *Ophthalmology* 2018.

Update the `BASE_DIR` path in the notebook/scripts to match your machine.

---

## Usage

### Full LGA-Net training (Stage 1 + Stage 2, 5-fold CV)

Open and run `LGA_NET_V3.ipynb` cell by cell.

### Ablation configurations (6 configs, 3 seeds each)

```bash
python dual_branch_no_lgag_multiseed.py   # No LGAG (dual-branch fusion only)
```

Other ablation configurations (no mask supervision, no warm-start, Swin-only, λ sweep) follow the same multiseed pattern; see `dual_branch_no_lgag_ablation.py` for the shared training loop.

### Bootstrap confidence intervals

```bash
python dual_branch_no_lgag_bootstrap_ci.py
```

Reconstructs per-sample predictions from saved checkpoints and computes the pooled-OOF bootstrap 95% CI used throughout Table 7, matching the methodology used for every other ablation row.

---

## Model Architecture

```
Input (380×380)
    │
    ├── EfficientNet-B4 ──► Pool → Conv 1792→512 ──┐
    │                                               ├──► Concat (B×1024×12×12)
    └── Swin-Tiny ────────► Pool → Conv  768→512 ──┘
                                                    │
                                              Fusion Conv
                                            (1024→512, BN, ReLU)
                                                    │
                                    Lesion-Guided Attention Gate
                                    (supervised by IDRiD masks)
                                                    │
                                    GAP → Dropout(0.5) → FC(512→5)
                                                    │
                                            5-class output
```

### Training Protocol

- **Stage 1:** Frozen backbones, LGAG warm-started on IDRiD lesion masks (BCE supervision), attention-loss-based checkpoint selection (QWK is unreliable on 11 validation images).
- **Stage 2:** Full fine-tuning on APTOS 2019, 5-fold stratified CV, discriminative learning rates (5×10⁻⁶ backbone, 5×10⁻⁵ LGAG/classifier), checkpoint selection on validation QWK.

---

## External Validation (Messidor-2)

Zero-shot transfer, unified 5-way test-time augmentation protocol (identity, horizontal/vertical flip, 90°/270° rotation) applied identically to all models on both datasets:

| Model | Dataset | QWK | Accuracy | AUC (OvR) |
|---|---|---|---|---|
| LGA-Net | APTOS (internal) | 0.8927 | 0.8349 | 0.8835 |
| LGA-Net | Messidor-2 (external) | 0.5423 | 0.5722 | 0.6621 |
| Swin-Tiny | APTOS (internal) | 0.8961 | 0.8240 | 0.8532 |
| Swin-Tiny | Messidor-2 (external) | 0.5771 | 0.5837 | 0.6516 |
| ResNet-50 | APTOS (internal) | 0.8868 | 0.7995 | 0.9169 |
| ResNet-50 | Messidor-2 (external) | 0.4724 | 0.6210 | 0.7901 |
| EfficientNet-B4 | APTOS (internal) | 0.8712 | 0.7790 | 0.9013 |
| EfficientNet-B4 | Messidor-2 (external) | 0.4465 | 0.6135 | 0.7472 |
| MobileNetV2 | APTOS (internal) | 0.8403 | 0.7858 | 0.8877 |
| MobileNetV2 | Messidor-2 (external) | 0.3899 | 0.5981 | 0.7164 |

Every model's QWK drops substantially from APTOS to Messidor-2. LGA-Net beats all three CNN baselines externally by a clear margin, but Swin-Tiny is numerically ahead of LGA-Net on Messidor-2 (0.5771 vs. 0.5423); LGA-Net's mask-guided attention does not close the domain-shift gap. These external comparisons are descriptive, not significance-tested (see the paper's Discussion for the caveat and a plausible explanation).

---

## Pre-trained Weights

Model weights are too large for GitHub (each checkpoint is ~100–195MB, above GitHub's 100MB hard limit). Host them externally, e.g. Google Drive, and link below:

| Checkpoint | Link | Size |
|---|---|---|
| Stage 1 best (IDRiD warm-start) | [Add link] | ~120MB |
| Stage 2 best (APTOS, canonical run) | [Add link] | ~195MB |
| Ablation checkpoints (6 configs × 3 seeds × 2 stages) | [Add link] | ~195MB each |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lganet2026,
  title   = {LGA-Net: Cross-Dataset Mask-Supervised Attention over Fused
             CNN--Transformer Features for Diabetic Retinopathy Grading},
  author  = {Arshad, Sobia},
 }
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [APTOS 2019 Kaggle Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [IDRiD Challenge](https://idrid.grand-challenge.org/)
- [timm library](https://github.com/rwightman/pytorch-image-models) by Ross Wightman
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
