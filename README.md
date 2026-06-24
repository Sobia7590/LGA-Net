# LGA-Net: Lesion-Guided Attention Network for Diabetic Retinopathy Grading




## Overview

LGA-Net fuses EfficientNet-B4 (local lesion features) with Swin-Tiny (global context) through a Lesion-Guided Attention Gate (LGAG), a single-stream spatial attention module supervised with pixel-level IDRiD lesion masks during a dedicated warm-start stage before grade-level fine-tuning.
---

## Results

### 5-Fold Stratified Cross-Validation (APTOS 2019)

| Fold | QWK |
|------|-----|
| Fold 1 | 0.9132 |
| Fold 2 | 0.9091 |
| Fold 3 | 0.9159 |
| Fold 4 | 0.8984 |
| Fold 5 | 0.9037 |
| **Mean** | **0.9080 ± 0.0071** |

| Metric | Value |
|--------|-------|
| Mean QWK | **0.9080 ± 0.0071** |
| OOF QWK | 0.9081 |
| Accuracy | 0.8250 ± 0.0188 |
| AUC (macro OvR) | 0.8975 ± 0.0222 |
| Single-run best QWK | 0.8981 (epoch 17) |

### External Validation (Messidor-2, zero-shot)

| Metric | Value |
|--------|-------|
| QWK | 0.5608 |
| Accuracy | 0.6032 |
| Macro AUC | 0.7126 |
| No DR Recall | 92.8% |

### Baseline Comparison

| Method | QWK | Accuracy | Params (M) |
|--------|-----|----------|------------|
| MobileNetV2 | 0.8264 | 0.7449 | 2.23 |
| EfficientNet-B4 | 0.8509 | 0.7831 | 17.56 |
| ResNet50 | 0.8697 | 0.7844 | 23.52 |
| Swin-Tiny | 0.8933 | 0.8131 | 27.52 |
| Wang et al. (2022)† | 0.912 | 0.821 | ~36 |
| Sun et al. (2023)† | 0.916 | 0.834 | ~42 |
| **LGA-Net (Ours)** | **0.9080±0.0071*** | **0.8250*** | **48.38** |

\* 5-fold cross-validation mean. † Single-run published result.

### Ablation Study

| Variant | QWK | Params (M) |
|---------|-----|------------|
| EfficientNet-B4 only | 0.8509 | 17.56 |
| Swin-Tiny only | 0.8933 | 27.52 |
| Dual-branch, no LGAG | 0.8762 | 46.91 |
| **LGA-Net full** | **0.8981** | **48.38** |

---

## Architecture

```
Input Fundus Image (380×380)
          │
    ┌─────┴─────┐
    │           │
EfficientNet-B4  Swin-Tiny
 B×512×12×12    B×512×12×12
    │           │
    └─────┬─────┘
          │ Concat → B×1024×12×12
    Feature Fusion (Conv1×1·BN·ReLU)
          │ B×512×12×12
    LGAG Attention Gate
    A ∈ [0,1]^{12×12}  F′ = F ⊙ A
          │
    GAP → Dropout(0.5) → Linear(512→5)
          │
    5-Class DR Grade
```

### Two-Stage Training

| | Stage 1 | Stage 2 |
|--|---------|---------|
| Dataset | IDRiD (54 images) | APTOS 2019 (3,662) |
| Backbone | Frozen | Unfrozen |
| Loss | ℒCE + 0.30·ℒBCE | ℒCE |
| LR backbone | 1×10⁻⁴ | 5×10⁻⁵ |
| LR gate | 1×10⁻⁴ | 5×10⁻⁶ |
| Best epoch | 11 | 17 |

---

## Web Application

Real-time DR grading via FastAPI + React.

```bash
# Backend (port 8000)
conda activate adni_clean
uvicorn backend:app --reload --port 8000

# Frontend (port 5173)
cd lganet-frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a fundus image to get instant DR grade, confidence score, per-class probabilities, and clinical recommendation.

---

## Datasets

| Dataset | Role | N | Link |
|---------|------|---|------|
| APTOS 2019 | Training + 5-fold CV | 3,662 | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| IDRiD | LGAG lesion pre-training | 54 | [Grand Challenge](https://idrid.grand-challenge.org) |
| Messidor-2 | Zero-shot external validation | 1,744 | [ADCIS](https://www.adcis.net/en/third-party/messidor2) |

---

## Pre-trained Models

Download from Google Drive: **[Model Weights](YOUR_DRIVE_LINK_HERE)**

| File | Description |
|------|-------------|
| `lganet_stage2_best_stage2_aptos_full_finetune.pth` | Main model (QWK 0.8981) |
| `lganet_stage1_best_frozen.pth` | Stage 1 checkpoint |
| `lganet_fold1_best.pth` – `fold5_best.pth` | 5-fold checkpoints |

---

## Installation

```bash
pip install torch torchvision timm fastapi uvicorn python-multipart \
            pillow opencv-python scikit-learn numpy pandas matplotlib
```

---

## Repository Structure

```
LGA-Net/
├── LGA_NET.ipynb          # Complete training + evaluation notebook
├── backend.py                # FastAPI inference server
├── requirements.txt          # Python dependencies
├── lganet-frontend/          # React web application
│   ├── src/                  # App source code
│   └── package.json
├── outputs_v2/               # Training CSVs, evaluation results
└── outputs_600dpi/           # All 41 figures at 600 DPI
```

---

## Citation

```bibtex
@article{sobia2025lganet,
  title={LGA-Net: A Lesion-Guided Attention Network Combining Convolutional
         and Transformer Encoders for Automated Diabetic Retinopathy Grading},
  author={Sobia, Arshad and Kim, Yongcheol},
  journal={Artificial Intelligence in Medicine},
  year={2025},
  publisher={Elsevier}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
