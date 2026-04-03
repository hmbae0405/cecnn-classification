# cecnn-classification
# AI-Assisted Classification of Colorectal Lesions in Colonoscopy Using Attention-Enhanced CNN

> **Manuscript:** *AI-Assisted Classification of Colorectal Lesions in Colonoscopy Using Attention Enhanced CNN*  
> **Authors:** Hyunmin Pae, Junghun Han, Hong Jun Park, Hyunil Kim, Hyun-Soo Kim, Jae Seung Soh, Seong-Jung Kim, Ji Eun Kim, Sejung Yang, Su Young Kim  
> **Status:** Under revision

---

## Overview

Colonoscopy remains the gold standard for colorectal cancer prevention, yet lesion miss rates persist in clinical practice. This repository implements a lightweight, attention-enhanced deep learning pipeline for binary classification of colonoscopic images into **normal** and **abnormal** categories.

We investigate whether integrating the **Convolutional Block Attention Module (CBAM)** into a ConvNeXt-Tiny backbone can improve classification performance while preserving computational efficiency. CBAM sequentially applies channel attention and spatial attention to highlight lesion-relevant features and suppress background noise.

**Models included:**

| Model | Description |
|---|---|
| `ConvNeXt-Tiny` | Baseline — ImageNet-pretrained, frozen backbone, 2-class head |
| `ConvNeXt-Tiny + CBAM` | CBAM inserted after the final feature stage, before global pooling |

---

## Repository Structure

```
.
├── convnext_tiny.py          # Baseline model
├── convnext_tiny_cbam.py     # Attention-enhanced model
├── README.md
└── results/                  # Auto-generated per run
    └── run_YYYYMMDD_HHMMSS/
        ├── best_model.pth
        ├── training_plot.png
        ├── roc_curve.png
        ├── confusion_matrix.png
        └── classification_metrics.txt
```

---

## Dataset Format

Place images directly under each split folder. Labels are inferred from filenames:

- `*nor*` → class `0` (normal)  
- `*ab*` → class `1` (abnormal)

```
path_data/
├── train/
│   ├── case01_nor.jpg
│   ├── case02_ab.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

> **Note:** Patient image data are not included in this repository. The dataset contains retrospective multi-institutional clinical data and is not publicly shareable.

---

## Preprocessing & Augmentation

Preprocessing steps applied to all images (as described in the manuscript):

1. Black-background removal
2. Lesion-centered cropping
3. Resize to `224 × 224`
4. ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

Training augmentations:

- Random horizontal flip
- Random rotation (±15°)
- Random brightness adjustment
- Random contrast adjustment

---

## Training Details

Both models share the same training pipeline for a fair comparison:

| Setting | Value |
|---|---|
| Backbone weights | ImageNet pretrained (`IMAGENET1K_V1`) |
| Backbone | Frozen |
| Loss | Class-weighted `CrossEntropyLoss` |
| Optimizer | Adam (`lr = 0.001`) |
| Scheduler | StepLR |
| Early stopping | Based on validation accuracy |
| Best model saved as | `best_model.pth` |
| GPU | NVIDIA RTX 2080Ti |

---

## Usage

1. Edit the path variables at the top of each script:

```python
path_data  = "YOUR_DATASET_PATH"   # root folder containing train/val/test
path_data2 = "YOUR_OUTPUT_PATH"    # where results/ will be written
```

2. Run either script:

```bash
python convnext_tiny.py
```

```bash
python convnext_tiny_cbam.py
```

---

## Evaluation Outputs

After training completes, the following are saved to `results/run_YYYYMMDD_HHMMSS/`:

- `best_model.pth` — best checkpoint by validation accuracy
- `training_plot.png` — loss and accuracy curves
- `roc_curve.png` — ROC curve with AUROC
- `confusion_matrix.png` — confusion matrix on the test set
- `classification_metrics.txt` — accuracy, precision, recall, F1, AUROC

---

## Key Findings

CBAM integration consistently improved classification performance across CNN architectures. **ConvNeXt-Tiny + CBAM** achieved the strongest overall results in our study, and MobileNetV4 showed a notable performance gain after CBAM integration. These results support CBAM as a lightweight and practical strategy for AI-assisted colonoscopy.

---

