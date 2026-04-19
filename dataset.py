"""Dataset classes for NYCU HW2 digit detection.

Data layout:
    <data_dir>/
        train/   valid/   test/        <- image files
        train.json   valid.json        <- COCO-format annotations

Augmentation pipeline (training only):
  1. Letterbox resize to img_size × img_size (grey padding).
  2. Albumentations:
       - OneOf([GaussNoise, GaussianBlur, MotionBlur], p=0.4)
       - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05, p=0.5)
     No flip, no rotation, no crop.
     Scale variation is handled by multi-scale training in train.py.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert

try:
    import albumentations as _albu  # only checks the package exists
    _ALBU_OK = True
except ImportError:
    _ALBU_OK = False


def _build_albu_pipeline():
    """Augmentation pipeline for digit detection.

    Three groups:
      - OneOf([GaussNoise, GaussianBlur, MotionBlur], p=0.4)
            simulates camera quality variation (blur, compression, sensor noise)
      - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05, p=0.5)
            replaces RandomBrightnessContrast + HueSaturationValue in one transform
      - Affine(scale=0.85–1.15, p=0.5)
            light scale jitter; no rotation, no translate, no shear.
            Digits stay within frame; boxes tracked by albumentations bbox pipeline.

    No flip, no rotation, no crop.
    """
    if not _ALBU_OK:
        raise ImportError("albumentations is required. Run: pip install albumentations")

    from albumentations import Compose, OneOf, GaussNoise, GaussianBlur, MotionBlur, ColorJitter, Affine

    return Compose([
        OneOf([
            GaussNoise(p=1.0),
            GaussianBlur(blur_limit=(3, 5), p=1.0),
            MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.4),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05, p=0.5),
        Affine(scale=(0.85, 1.15), rotate=0, shear=0, p=0.5),
    ], bbox_params={
        "format": "coco",
        "label_fields": ["category_ids"],
        "min_visibility": 0.3,
    })


def _load_coco_json(json_path: str) -> tuple:
    with open(json_path) as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations[ann["image_id"]].append(ann)
    return images, annotations


def _letterbox(img: Image.Image, size: int) -> tuple:
    """Resize to (size, size) with grey padding, preserving aspect ratio."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    padded.paste(img, (pad_x, pad_y))
    return padded, scale, pad_x, pad_y


def _apply_letterbox_to_boxes(boxes, scale, pad_x, pad_y) -> list:
    return [[x1 * scale + pad_x, y1 * scale + pad_y,
             x2 * scale + pad_x, y2 * scale + pad_y]
            for x1, y1, x2, y2 in boxes]


def _clip_boxes(boxes, size) -> tuple:
    out_boxes, out_mask = [], []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(size), x2), min(float(size), y2)
        out_boxes.append([x1, y1, x2, y2])
        out_mask.append((x2 - x1 > 2) and (y2 - y1 > 2))
    return out_boxes, out_mask


class DigitDetectionDataset(Dataset):
    """COCO-format digit detection dataset.

    Args:
        img_dir: Directory containing image files.
        json_path: COCO-format annotation JSON.
        img_size: Square letterbox target size.
        augment: Enable training augmentation.
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, img_dir: str, json_path: str, img_size: int = 448, augment: bool = True):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self._albu = _build_albu_pipeline() if augment else None
        self._images, self._annotations = _load_coco_json(json_path)
        self.img_ids = list(self._images.keys())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> tuple:
        raw_img, raw_boxes, labels = self._load(idx)
        img, scale, pad_x, pad_y = _letterbox(raw_img, self.img_size)
        boxes = _apply_letterbox_to_boxes(raw_boxes, scale, pad_x, pad_y)

        if self.augment:
            img, boxes, labels = self._apply_albu(img, boxes, labels)

        boxes, keep = _clip_boxes(boxes, self.img_size)
        boxes = [b for b, k in zip(boxes, keep) if k]
        labels = [lb for lb, k in zip(labels, keep) if k]

        img_t = TF.normalize(TF.to_tensor(img), self._MEAN, self._STD)
        img_id = self.img_ids[idx]
        info = self._images[img_id]

        if boxes:
            boxes_t = box_convert(torch.tensor(boxes, dtype=torch.float32), "xyxy", "cxcywh")
            boxes_t = (boxes_t / self.img_size).clamp(0.0, 1.0)
            valid = (boxes_t[:, 2] > 1e-4) & (boxes_t[:, 3] > 1e-4)
            boxes_t = boxes_t[valid]
            labels_t = (torch.tensor(labels, dtype=torch.long) - 1)[valid]
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        return img_t, {
            "image_id": torch.tensor(img_id),
            "orig_size": torch.tensor([info["height"], info["width"]]),
            "boxes": boxes_t,
            "labels": labels_t,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }

    def _load(self, idx: int) -> tuple:
        img_id = self.img_ids[idx]
        info = self._images[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        boxes, labels = [], []
        for ann in self._annotations[img_id]:
            x, y, w, h = ann["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
        return img, boxes, labels

    def _apply_albu(self, img, boxes, labels) -> tuple:
        """Apply albumentations pipeline (noise/blur/colour/affine scale-jitter).

        Boxes are clipped to image bounds before passing to albumentations 2.0,
        which strictly validates that all coordinates are in [0, image_size].
        """
        if self._albu is None:
            return img, boxes, labels
        img_np = np.array(img)
        W, H = img.size
        # Clip to image bounds and filter degenerate boxes before albumentations
        # 2.0 strict validator (normalises coords internally, rejects < 0).
        valid_coco, valid_labels = [], []
        for (x1, y1, x2, y2), lbl in zip(boxes, labels):
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 - x1 > 1 and y2 - y1 > 1:
                valid_coco.append([x1, y1, x2 - x1, y2 - y1])
                valid_labels.append(lbl)
        t = self._albu(image=img_np, bboxes=valid_coco, category_ids=valid_labels)
        img = Image.fromarray(t["image"])
        boxes = [[x, y, x + w, y + h] for x, y, w, h in t["bboxes"]]
        labels = list(t["category_ids"])
        return img, boxes, labels


class TestDataset(Dataset):
    """Image-only dataset for test split (no annotations).

    Image IDs are parsed from numeric file stems (e.g. "00123.png" → 123).
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, img_dir: str, img_size: int = 448):
        self.img_size = img_size
        self.img_files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in Path(img_dir).glob(ext))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple:
        path = self.img_files[idx]
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        padded, scale, pad_x, pad_y = _letterbox(img, self.img_size)
        img_t = TF.normalize(TF.to_tensor(padded), self._MEAN, self._STD)
        return img_t, {"image_id": int(path.stem), "orig_size": (orig_h, orig_w),
                       "scale": scale, "pad_x": pad_x, "pad_y": pad_y}


def collate_fn(batch) -> tuple:
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def test_collate_fn(batch) -> tuple:
    imgs, metas = zip(*batch)
    return torch.stack(imgs), list(metas)