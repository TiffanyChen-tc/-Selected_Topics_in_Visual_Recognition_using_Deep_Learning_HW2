# HW2 — Digit Detection with Deformable DETR

## Introduction

This project tackles digit detection using a **Deformable DETR** framework built on a ResNet-50 backbone. The model is trained end-to-end on a COCO-format dataset containing digits 0–9, using Hungarian matching for set-based loss computation and multi-scale deformable attention for efficient feature aggregation across C2–C5 feature levels.

Key design choices:
- **Backbone**: ResNet-50 (ImageNet V2 pretrained), all BatchNorm layers frozen
- **Neck**: Per-level 1×1 Conv + GroupNorm projections to d=256
- **Encoder/Decoder**: 6-layer deformable attention encoder and decoder with iterative box refinement
- **Loss**: CE + L1 + GIoU with auxiliary decoder losses (aux_weight=0.5)
- **Augmentation**: Letterbox resize → GaussNoise/Blur/MotionBlur + ColorJitter + Affine scale jitter
- **Inference**: Per-class NMS + confidence threshold sweep + max-dets cap

---

## Environment Setup

**Requirements**: Python 3.10+, CUDA-capable GPU (tested on RTX 4090)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations scipy pycocotools matplotlib tqdm
```

**Dataset layout**:
```
nycu-hw2-data/
    train/          # training images
    valid/          # validation images
    test/           # test images (no annotations)
    train.json      # COCO-format annotations
    valid.json      # COCO-format annotations
```

---

## Usage

### Training

```bash
python train.py --data-dir nycu-hw2-data
```

Outputs are saved to `output/`:
- `best.pth` — checkpoint with highest val mAP
- `last.pth` — latest checkpoint
- `loss_curves.png` — train/val loss curves (updated every eval interval)
- `confusion_matrix.png` — generated at end of training using best.pth

### Inference

Sweep to find the best confidence threshold on val, then run test inference:

```bash
python infer.py --data-dir nycu-hw2-data --checkpoint output/best.pth --sweep
```

Run with a fixed threshold:

```bash
python infer.py --data-dir nycu-hw2-data --checkpoint output/best.pth --threshold 0.02
```

Predictions are saved to `pred.json` in COCO result format.

## Performance Snapshot

| Method | Val mAP | Test mAP |
|---|---|---|
| Baseline (DETR) | 0.408 | 0.30 |
| + Deformable DETR | 0.443 | 0.35 |
| + Data Augmentation | 0.452 | 0.37 |
| + Threshold Tuning (thr=0.02) | **0.465** | **0.38** |

Val mAP breakdown at best checkpoint (epoch 18):

| Metric | Value |
|---|---|
| AP@[0.50:0.95] | 0.465 |
| AP@0.50 | 0.927 |
| AP@0.75 | 0.390 |
| AR@100 | 0.552 |