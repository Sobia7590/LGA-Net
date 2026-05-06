# LGA-Net: Lesion-Guided Attention Network for Diabetic Retinopathy Grading

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-green)](https://www.kaggle.com/c/aptos2019-blindness-detection)
[![QWK](https://img.shields.io/badge/QWK-0.9074%20%C2%B10.0065-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Official implementation of **LGA-Net**, a dual-branch CNN-Transformer architecture with a Lesion-Guided Attention Gate (LGAG) for automated five-class diabetic retinopathy grading.

---

## Overview

LGA-Net fuses:
- **EfficientNet-B4** (local lesion texture features)
- **Swin-Tiny Transformer** (hierarchical global context)

through a **Lesion-Guided Attention Gate** pre-trained with pixel-level lesion supervision from the IDRiD segmentation dataset, before end-to-end fine-tuning on APTOS 2019.

**Key result:** QWK = 0.9074 ± 0.0065 under 5-fold stratified cross-validation — the most statistically robust evaluation in this comparison class.

---

## Results on APTOS 2019

| Model | QWK | Accuracy | AUC (OvR) | Eval Protocol |
|---|---|---|---|---|
| MobileNetV2 | 0.8264 | 0.7449 | 0.8624 | Single run |
| EfficientNet-B4 | 0.8509 | 0.7831 | 0.8747 | Single run |
| ResNet50 | 0.8697 | 0.7844 | 0.8848 | Single run |
| Swin-Tiny | 0.8933 | 0.8131 | 0.9046 | Single run |
| **LGA-Net (Ours)** | **0.9074 ± 0.0065** | **0.8296** | **0.8961** | **5-fold CV** |

### Ablation Study

| Variant | QWK | Accuracy | AUC | Params |
|---|---|---|---|---|
| EfficientNet-B4 only | 0.8509 | 0.7831 | 0.8747 | 17.56M |
| Swin-Tiny only | 0.8933 | 0.8131 | 0.9046 | 27.52M |
| Dual-branch (no LGAG) | 0.8859 | 0.7749 | 0.8947 | 46.91M |
| **LGA-Net Full** | **0.9074 ± 0.0065** | **0.8296** | **0.8961** | 48.38M |

---

## Installation

```bash
git clone https://github.com/YourUsername/LGA-Net.git
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

Update the `BASE_DIR` path in the notebook to match your machine.

---

## Usage

### Full LGA-Net training (Stage 1 + Stage 2 with 5-fold CV)
Open and run `LGA_NET_V3.ipynb` cell by cell.

### Ablation: Dual-branch without LGAG
```bash
python dual_branch_no_lgag_ablation.py
```

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
- **Stage 1:** Frozen backbone, LGAG trained on IDRiD lesion masks (BCE supervision, λ=0.30), 11 epochs
- **Stage 2:** Full fine-tuning on APTOS 2019, 5-fold stratified CV, early stopping (patience=5), 15 max epochs per fold

---

## Pre-trained Weights

Model weights are too large for GitHub. Download from Google Drive:

| Checkpoint | Link | Size |
|---|---|---|
| Stage 1 best (IDRiD) | [Download](#) | ~190MB |
| Stage 2 best (APTOS fold 2) | [Download](#) | ~190MB |

> Replace `#` with your Google Drive link after uploading.

---

## External Validation (Messidor-2)

Zero-shot transfer with domain normalisation and TTA:

| Class | AUC | Recall |
|---|---|---|
| No DR | 0.751 | 86% |
| Mild | 0.557 | 14% |
| Moderate | 0.746 | 31% |
| Severe | 0.610 | 16% |
| Proliferative | 0.899 | 49% |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lganet2025,
  title   = {LGA-Net: A Lesion-Guided Attention Network Combining CNN and
             Transformer Features for Automated Diabetic Retinopathy Grading},
  author  = {Author, First and Author, Second and Author, Third},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  note    = {Under review}
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
