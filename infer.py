"""Inference script for trained DETR digit detection model.

Features:
  - Per-class NMS: each digit class is NMS'd independently.
  - Max-dets cap: limits predictions per image (default 30).
  - Threshold sweep: finds the best score threshold on val before test inference.

Usage:
    python infer.py --data-dir nycu-hw2-data --checkpoint output/best.pth --sweep
    python infer.py --data-dir nycu-hw2-data --checkpoint output/best.pth --threshold 0.02
"""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from torchvision.ops import nms as tv_nms

from dataset import DigitDetectionDataset, TestDataset, collate_fn, test_collate_fn
from model import DETR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="nycu-hw2-data")
    p.add_argument("--checkpoint", type=str, default="output/best.pth")
    p.add_argument("--output", type=str, default="pred.json")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--max-dets", type=int, default=30)
    p.add_argument("--no-test", action="store_true")
    return p.parse_args()


def load_model(checkpoint_path: str, num_queries: int, device) -> DETR:
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("args", {})
    model = DETR(num_classes=10, d_model=saved.get("d_model", 256),
                 nhead=saved.get("nhead", 8),
                 num_encoder_layers=saved.get("num_encoder_layers", 6),
                 num_decoder_layers=saved.get("num_decoder_layers", 6),
                 dim_feedforward=saved.get("dim_feedforward", 1024),
                 dropout=0.0, num_queries=num_queries, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _per_class_nms(boxes: torch.Tensor, scores: torch.Tensor,
                   labels: torch.Tensor, iou_threshold: float) -> tuple:
    """NMS applied independently per class to avoid suppressing different digits."""
    if len(scores) == 0:
        return boxes, scores, labels
    keep_idx = []
    for cls_id in labels.unique():
        mask = labels == cls_id
        orig_idx = mask.nonzero(as_tuple=True)[0]
        kept = tv_nms(boxes[mask].float(), scores[mask].float(), iou_threshold)
        keep_idx.append(orig_idx[kept])
    keep = torch.cat(keep_idx)
    keep = keep[scores[keep].argsort(descending=True)]
    return boxes[keep], scores[keep], labels[keep]


def _decode_single(pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                   img_size: int, pad_x: float, pad_y: float,
                   letterbox_scale: float, orig_w: float, orig_h: float,
                   threshold: float) -> tuple:
    """Decode one image's predictions to (boxes_xyxy, scores, labels) in original coords."""
    probs = pred_logits.float().softmax(-1)
    scores, labels = probs[:, :-1].max(-1)
    keep = scores > threshold
    if not keep.any():
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    sc, cid = scores[keep], labels[keep]
    bx = box_convert(pred_boxes[keep].float() * img_size, "cxcywh", "xyxy")
    bx[:, [0, 2]] = ((bx[:, [0, 2]] - pad_x) / letterbox_scale).clamp(0.0, orig_w)
    bx[:, [1, 3]] = ((bx[:, [1, 3]] - pad_y) / letterbox_scale).clamp(0.0, orig_h)
    return bx, sc, cid


def _infer_batch(model, imgs: torch.Tensor, metas: list, device, img_size: int,
                 threshold: float, nms_iou: float, max_dets: int, is_val: bool) -> list:
    """Run inference for one batch, apply per-class NMS and max-dets cap."""
    with torch.no_grad():
        out = model(imgs.to(device))
    results = []
    for i, m in enumerate(metas):
        oh, ow = m["orig_size"].tolist() if is_val else m["orig_size"]
        bx, sc, cid = _decode_single(
            out["pred_logits"][i], out["pred_boxes"][i],
            img_size, m["pad_x"], m["pad_y"], m["scale"], ow, oh, threshold)
        if len(sc):
            bx, sc, cid = _per_class_nms(bx, sc, cid, nms_iou)
        if len(sc) > max_dets:
            topk = sc.topk(max_dets)
            bx, cid, sc = bx[topk.indices], cid[topk.indices], topk.values
        img_id = m["image_id"].item() if is_val else m["image_id"]
        preds = [{"category_id": int(c) + 1, "bbox": box, "score": float(s)}
                 for s, c, box in zip(sc.tolist(), cid.tolist(),
                                      box_convert(bx, "xyxy", "xywh").tolist())]
        results.append((img_id, preds))
    return results


@torch.no_grad()
def run_val(model, loader, device, img_size, threshold, nms_iou, max_dets) -> list:
    model.eval()
    preds = []
    for imgs, targets in loader:
        for img_id, ps in _infer_batch(model, imgs, targets, device, img_size,
                                       threshold, nms_iou, max_dets, is_val=True):
            preds.extend({"image_id": img_id, **p} for p in ps)
    return preds


@torch.no_grad()
def run_test(model, loader, device, img_size, threshold, nms_iou, max_dets) -> list:
    model.eval()
    preds = []
    for imgs, metas in loader:
        for img_id, ps in _infer_batch(model, imgs, metas, device, img_size,
                                       threshold, nms_iou, max_dets, is_val=False):
            preds.extend({"image_id": img_id, **p} for p in ps)
    return preds


def _compute_map(preds: list, gt_json: str) -> float:
    if not preds:
        return 0.0
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return float(len(preds)) / 1e6
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preds, f)
        coco_dt = coco_gt.loadRes(f.name)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


def sweep_threshold(model, val_loader, val_json, device, img_size, nms_iou, max_dets) -> float:
    print(f"Sweeping thresholds | max_dets={max_dets}")
    best_thresh, best_map = 0.5, -1.0
    thresholds = [0.01, 0.02, 0.03] + np.arange(0.05, 0.90, 0.05).tolist()
    for thr in thresholds:
        preds = run_val(model, val_loader, device, img_size, thr, nms_iou, max_dets)
        with contextlib.redirect_stdout(io.StringIO()):
            score = _compute_map(preds, val_json)
        tag = " ←" if score > best_map else ""
        print(f"  thr={thr:.2f}  mAP={score:.4f}  n={len(preds)}{tag}")
        if score > best_map:
            best_map, best_thresh = score, thr
    print(f"Best threshold: {best_thresh:.2f}  (mAP={best_map:.4f})")
    return best_thresh


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | max_dets: {args.max_dets}")

    data_dir = Path(args.data_dir)
    model = load_model(args.checkpoint, args.num_queries, device)
    print(f"Loaded: {args.checkpoint}")

    val_ds = DigitDetectionDataset(data_dir / "valid", data_dir / "valid.json",
                                   img_size=args.img_size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    val_json = str(data_dir / "valid.json")

    threshold = args.threshold
    if args.sweep or threshold is None:
        threshold = sweep_threshold(model, val_loader, val_json, device,
                                    args.img_size, args.nms_iou, args.max_dets)
    else:
        map_val = _compute_map(
            run_val(model, val_loader, device, args.img_size,
                    threshold, args.nms_iou, args.max_dets), val_json)
        print(f"Val mAP = {map_val:.4f}  (thr={threshold:.3f})")
    print(f"\nUsing threshold = {threshold:.3f}")

    if args.no_test:
        return

    test_ds = TestDataset(data_dir / "test", img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=test_collate_fn)
    print(f"Inference on {len(test_ds)} test images …")
    preds = run_test(model, test_loader, device, args.img_size,
                     threshold, args.nms_iou, args.max_dets)
    print(f"Total predictions: {len(preds)}")
    with open(args.output, "w") as f:
        json.dump(preds, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()