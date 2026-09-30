# LGA-Net: Cross-Dataset Mask-Supervised Attention for Diabetic Retinopathy Grading

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-green)](https://www.kaggle.com/c/aptos2019-blindness-detection)
[![QWK](https://img.shields.io/badge/QWK-0.9049%C2%B10.0069-brightgreen)](#results-on-aptos-2019)
[![Accepted-MICAD 2026](https://img.shields.io/badge/Accepted-MICAD%202026-blueviolet)](#overview)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Official implementation of **LGA-Net** (~48.4M parameters), a dual-branch CNN-Transformer architecture with a Lesion-Guided Attention Gate (LGAG), cross-dataset mask-supervised via IDRiD before fine-tuning on APTOS 2019.
>
> **Accepted at the 7th International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2026)**, 22-24 October 2026, Edinburgh, UK - Paper ID 761.

---

## Overview

LGA-Net fuses:

- **EfficientNet-B4** (local, texture-like lesion features)
- **Swin-Tiny Transformer** (hierarchical global context)

through a **Lesion-Guided Attention Gate (LGAG)** - a single-stream attention mechanism supervised directly by pixel-level IDRiD lesion masks in Stage 1, then transferred, unmodified, to grade APTOS 2019 in Stage 2.

The paper's contributions are: (1) the dual-branch, cross-dataset mask-supervised LGAG itself; (2) a two-stage training regime evaluated under 5-fold cross-validation against four backbone baselines with paired and independent-samples significance tests; (3) a seven-configuration, three-seed ablation testing whether any single design choice, including mask supervision itself, moves raw QWK beyond run-to-run noise; and (4) a quantitative interpretability analysis - attention-lesion Dice/IoU at native resolution, a pooled attention-density correlation replicated on a fully held-out test set, and a deletion/insertion test - paired with a fully quantified, per-grade zero-shot external validation on Messidor-2.

**Headline result:** QWK = **0.9049 ± 0.0069** under 5-fold stratified cross-validation on APTOS 2019 (pooled out-of-fold predictions), statistically indistinguishable from a Swin-Tiny-only backbone (paired Wilcoxon, p = 0.3125) and significantly ahead of ResNet-50, EfficientNet-B4, and MobileNetV2 (bootstrap 95% CIs on the QWK difference exclude zero in all three cases).

The ablation shows no single design choice drives that QWK on its own - every ablated configuration's 95% CI overlaps the full model's. As the paper's own conclusion puts it, the contribution is narrower than the project's original framing: *cross-dataset mask supervision is not shown to drive raw grading accuracy, but is shown, with quantitative rather than qualitative evidence, to produce attention consistent with a spatial regularization role.*

---

## Results on APTOS 2019

Pooled out-of-fold predictions, 5-fold cross-validation: **QWK 0.9049 ± 0.0069, Accuracy 0.8263, AUC 0.8908** (macro-averaged one-vs-rest).

LGA-Net vs. each baseline, matched 5-fold cross-validation, paired significance tests:

| Comparison | Mean diff. (QWK) | p (Wilcoxon) | Bootstrap 95% CI |
|---|---|---|---|
| vs. Swin-Tiny | -0.0064 | 0.3125 | [-0.0140, +0.0004] |
| vs. ResNet-50 | +0.0088 | 0.0625 | [+0.0010, +0.0167] |
| vs. EfficientNet-B4 | +0.0192 | 0.0625 | [+0.0098, +0.0290] |
| vs. MobileNetV2 | +0.0353 | 0.0625 | [+0.0250, +0.0457] |

LGA-Net and Swin-Tiny show no statistically significant difference (a null result, not evidence of equivalence), while LGA-Net is significantly ahead of all three CNN baselines under this matched protocol. With only 5 paired folds, the Wilcoxon test's minimum possible p-value is 0.0625 - a floor imposed by sample size, which is why the bootstrap CI on pooled predictions is the more informative test here.

### Ablation Study

A seven-configuration, three-seed ablation: removing the LGAG (dual-branch fusion only), removing mask supervision (λ = 0), removing the Stage 1 warm-start, using the Swin-Tiny + LGAG branch alone (no CNN branch), and sweeping the mask-loss weight λ ∈ {0.15, 0.5, 1.0} against the paper's default of λ = 0.30.

No ablated configuration shows a bootstrap-CI/Mann-Whitney-detectable difference from the full model. The largest numerical gap is **No LGAG: +0.0080 mean QWK**, with a 95% CI of **[-0.0045, +0.0221]** - comfortably including zero. With only 3 seeds against 5 folds, this indicates the data cannot resolve a difference at this sample size, not that the ablated configurations are proven equal.

LGA-Net's demonstrated contribution is therefore on interpretability (attention-lesion overlap, deletion/insertion testing), not raw grading accuracy - see the paper's Discussion.

---

## Interpretability

Because no single architectural choice is shown to drive raw QWK, LGA-Net's evidence for the LGAG rests on what its attention actually does.

- At its native 12×12 resolution, LGAG attention overlaps IDRiD lesion annotations with **Dice 0.6227 (IoU 0.4641)** on the internal 11-image validation split, and **Dice 0.6286 (IoU 0.4708)** on IDRiD's official, fully held-out 27-image Testing Set, which enters this pipeline at no other point.
- In both samples, attention overlaps a trivial circular field-of-view mask more strongly than it overlaps lesion masks (Dice 0.7612 and 0.7672, respectively), so it behaves more like a general disease-vs-no-disease signal than a precise lesion localizer.
- Pooled across patches, attention magnitude correlates with local lesion density at **r = 0.199** (internal split, n = 1,584 patches / 11 images) and **r = 0.221** (held-out test set, n = 3,888 patches / 27 images, p = 3.64 × 10⁻⁴⁴); cluster bootstraps that resample whole images (5,000 resamples) give 95% CIs of [0.086, 0.317] and [0.175, 0.267], both excluding zero.
- Treating each image as one independent observation instead, the image-level correlation is **not significant** in either sample (r = -0.308, p = 0.358, n = 11 internal; r = 0.170, p = 0.396, n = 27 held-out) - what replicates across both samples is the local, patch-to-patch pattern, not a between-image claim.
- A deletion/insertion test gives **deletion AUC 0.4525, insertion AUC 0.4736** (n = 30) - a gap of only 0.021, consistent with a stable spatial prior rather than a per-image explanation.

---

## Installation

```bash
git clone https://github.com/Sobia7590/LGA-Net.git
cd LGA-Net
pip install -r Requirements.txt
```

---

## Dataset Setup

### APTOS 2019

3,662 images; primary training and in-domain evaluation (5-fold stratified CV). Download from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) and place as:

```
LGA_NET/
└── aptos2019-blindness-detection/
    ├── train_images/
    └── train.csv
```

### IDRiD (for Stage 1 attention pre-training)

54 images with pixel-level lesion masks; used only for the Stage 1 warm-start with a frozen backbone. Re-split 43 train / 11 validation; IDRiD's official, separate 27-image Testing Set is reserved entirely as a held-out interpretability check and never enters training or model selection. Download from [IDRiD Grand Challenge](https://idrid.grand-challenge.org/) and place as:

```
LGA_NET/
└── IDRID_DATASET/
    └── A. Segmentation/
        └── A. Segmentation/
            ├── 1. Original Images/
            └── 2. All Segmentation Groundtruths/
```

### Messidor-2 (external validation only)

1,744 images, held out entirely for zero-shot external validation; never enters training. Grade labels follow Krause et al., *Ophthalmology* 2018.

Update the `BASE_DIR` path in the notebook/scripts to match your machine.

---

## Usage

### Full LGA-Net training (Stage 1 + Stage 2, 5-fold CV)

Open and run `LGA_NET.ipynb` cell by cell.

### Ablation configurations (7 configs, 3 seeds each)

```bash
python dual_branch_no_lgag_multiseed.py   # No LGAG (dual-branch fusion only)
```

Other ablation configurations (no mask supervision, no warm-start, Swin-only, λ sweep) follow the same multiseed pattern; see `dual_branch_no_lgag_ablation.py` for the shared training loop.

### Bootstrap confidence intervals

```bash
python dual_branch_no_lgag_bootstrap_ci.py
```

Reconstructs per-sample predictions from saved checkpoints and computes the pooled-OOF bootstrap 95% CI used throughout the paper's statistical comparisons.

---

## Model Architecture (~48.4M parameters)

```
Input (380x380)
    |
    +-- EfficientNet-B4 --> 1792-ch feature map (12x12) --> 1x1 Conv -> 512 --+
    |                                                                        +--> Concat (1024x12x12)
    +-- Swin-Tiny (input resized 224x224) --> 768-ch feature map --> --+      |
                                        bilinear upsample to 12x12 ----+--> 1x1 Conv -> 512 --+
                                                                        |
                                                Fusion: 1x1 Conv (1024->512), BatchNorm, ReLU  -->  F
                                                                        |
                                            Lesion-Guided Attention Gate (LGAG)
                            Two 3x3 convs (512->256->128, each BatchNorm + ReLU) + 1x1 Conv + Sigmoid --> A (12x12)
                                                                        |
                                                        F' = F (x) A  (gated features)
                                                                        |
                                        Global Average Pool -> Dropout(p=0.5) -> FC(512->5)
                                                                        |
                                                            5-class DR grade output
```

### Training Protocol

- **Stage 1:** Frozen backbones; only the projection, fusion, and LGAG layers are optimized, using IDRiD's 43 training images and their pixel-level lesion masks. Loss combines a label-smoothed cross-entropy term (ε = 0.05) on the grade labels with a binary cross-entropy term between the predicted attention map and the ground-truth lesion mask, weighted by **λ = 0.30**. Checkpoints are selected on validation attention (mask) loss rather than QWK, since QWK on an 11-image validation set is dominated by sampling noise. Best validation attention loss: **0.3674 at epoch 16**.
- **Stage 2:** The entire network is unfrozen and fine-tuned end-to-end on APTOS 2019 with discriminative learning rates (5×10⁻⁶ backbone, 5×10⁻⁵ LGAG/classifier), a class-weighted, label-smoothed cross-entropy loss, AdamW, and a cosine schedule for up to 25 epochs with early stopping (patience 7), selecting checkpoints on validation QWK. Canonical run: validation QWK **0.9008 at epoch 23**.

---

## External Validation (Messidor-2)

Under a unified five-way test-time-augmentation protocol (identity, horizontal/vertical flip, 90°/270° rotation) applied identically to all models on both datasets:

- On APTOS (in-domain), LGA-Net and Swin-Tiny remain close: QWK **0.8927** vs. **0.8961**.
- Zero-shot on Messidor-2 (never used in training or model selection), every model's QWK drops substantially - a real domain-shift gap, reported in full. LGA-Net falls to QWK **0.5423**; Swin-Tiny falls to **0.5771**, numerically ahead of LGA-Net externally.
- LGA-Net still beats all three CNN baselines (ResNet-50, EfficientNet-B4, MobileNetV2) externally, by **+0.070 to +0.152 QWK** - descriptive margins; the significance tests in [Results](#results-on-aptos-2019) cover only the in-domain APTOS comparisons.
- A per-grade breakdown shows the domain-shift gap concentrated in the minority grades: **Severe DR recall is only 0.080** (6/75 correctly identified, despite perfect precision), while the majority **No DR grade is comparatively well identified (recall 0.810)**.

LGA-Net's mask-guided attention does not close the domain-shift gap; see the paper's Discussion for the full treatment and a plausible explanation.

---

## Pre-trained Weights

Model weights are too large for GitHub (each checkpoint is ~100-195MB, above GitHub's 100MB hard limit). Host them externally, e.g. Google Drive, and link below:

| Checkpoint | Link | Size |
|---|---|---|
| Stage 1 best (IDRiD warm-start) | [Add link] | ~120MB |
| Stage 2 best (APTOS, canonical run) | [Add link] | ~195MB |
| Ablation checkpoints (7 configs × 3 seeds × 2 stages) | [Add link] | ~195MB each |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{arshad2026lganet,
  title     = {LGA-Net: Cross-Dataset Mask-Supervised Attention for
               Diabetic Retinopathy Grading},
  author    = {Arshad, Sobia and Kim, Yongcheol},
  booktitle = {Proceedings of the 7th International Conference on Medical
               Imaging and Computer-Aided Diagnosis (MICAD 2026)},
  year      = {2026},
  address   = {Edinburgh, UK},
  note      = {Paper ID 761}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

This research was supported by the ANCHOR program through the ANCHOR Center, Gyeongsangnam-do, funded by the Ministry of Education (MOE) and the Gyeongsangnam-do Provincial Government, Republic of Korea (Grant No. 2026-ANCHOR-16-008).

- [APTOS 2019 Kaggle Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [IDRiD Challenge](https://idrid.grand-challenge.org/)
- [timm library](https://github.com/rwightman/pytorch-image-models) by Ross Wightman
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
