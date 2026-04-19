"""Train DETR for NYCU HW2 digit detection.
LR schedule: linear warmup → cosine decay.
Usage:
    python train.py --data-dir nycu-hw2-data
    python train.py --data-dir nycu-hw2-data --resume output/last.pth --lr 3e-5
"""

import argparse
import contextlib
import io
import json
import math
import random
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for server/training
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from torchvision.ops import nms as tv_nms
from tqdm import tqdm

from dataset import DigitDetectionDataset, collate_fn
from model import DETR, DETRLoss, HungarianMatcher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DETR for digit detection")
    p.add_argument("--data-dir", type=str, default="nycu-hw2-data")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4, help="Peak LR for transformer/heads")
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="Peak LR for backbone")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-ce", type=float, default=1.0)
    p.add_argument("--loss-bbox", type=float, default=5.0)
    p.add_argument("--loss-giou", type=float, default=2.0)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--eval-interval", type=int, default=3)
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--grad-clip", type=float, default=0.1)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def _amp_settings(device, amp_mode: str) -> tuple:
    """Returns (amp_enabled, amp_dtype, use_scaler)."""
    if device.type != "cuda" or amp_mode == "none":
        return False, None, False
    if amp_mode == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True  # fp16 needs GradScaler


def _cast_fp32(outputs: dict) -> dict:
    """Recursively cast all float tensors in outputs to fp32."""
    result = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            result[k] = v.float()
        elif isinstance(v, list):
            result[k] = [{kk: vv.float() if isinstance(vv, torch.Tensor) and vv.is_floating_point()
                          else vv for kk, vv in item.items()}
                         if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def _build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min((epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, criterion, optimizer, loader, device, epoch,
                    scaler, amp_enabled, amp_dtype, grad_clip) -> float:
    model.train()
    total_loss, n_valid, n_bad = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False, dynamic_ncols=True)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(imgs)
        outputs_fp32 = _cast_fp32(outputs)
        if not (outputs_fp32["pred_logits"].isfinite().all() and
                outputs_fp32["pred_boxes"].isfinite().all()):
            n_bad += 1
            continue
        loss, loss_dict = criterion(outputs_fp32, targets)
        if not loss.isfinite():
            n_bad += 1
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        n_valid += 1
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / n_valid:.4f}",
                         ce=f"{loss_dict['loss_ce'].item():.3f}",
                         bbox=f"{loss_dict['loss_bbox'].item():.3f}",
                         giou=f"{loss_dict['loss_giou'].item():.3f}")
    pbar.close()
    if n_bad:
        print(f"  [warn] skipped {n_bad}/{len(loader)} batches (NaN)")
    return total_loss / max(n_valid, 1)


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device, amp_enabled, amp_dtype) -> tuple:
    """Returns (total, ce, bbox, giou) averaged over the val set."""
    model.eval()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    n = 0
    for imgs, targets in tqdm(loader, desc="            [val]  ", leave=False, dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(imgs)
        loss, loss_dict = criterion(_cast_fp32(outputs), targets)
        if loss.isfinite():
            totals["loss"]      += loss.item()
            totals["loss_ce"]   += loss_dict["loss_ce"].item()
            totals["loss_bbox"] += loss_dict["loss_bbox"].item()
            totals["loss_giou"] += loss_dict["loss_giou"].item()
            n += 1
    n = max(n, 1)
    return tuple(totals[k] / n for k in ("loss", "loss_ce", "loss_bbox", "loss_giou"))


def _per_class_nms(boxes: torch.Tensor, scores: torch.Tensor,
                   labels: torch.Tensor, iou_threshold: float) -> tuple:
    """NMS per digit class — mirrors infer.py to keep val metric consistent."""
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


@torch.no_grad()
def evaluate_map(model, loader, device, val_json: str, img_size: int,
                 amp_enabled: bool, amp_dtype,
                 score_thr: float = 0.05, nms_iou: float = 0.5,
                 max_dets: int = 30) -> float:
    """Compute COCO mAP@[0.50:0.95] using the same post-processing as infer.py.

    Uses per-class NMS and max_dets cap so the checkpoint selected as 'best'
    is evaluated with the same pipeline as final test inference.
    TTA is skipped here (too slow during training) but single-scale with
    consistent NMS is already a much better proxy than global NMS was.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return 0.0

    model.eval()
    preds = []
    for imgs, targets in tqdm(loader, desc="            [mAP]  ", leave=False, dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(imgs)
        for i, target in enumerate(targets):
            probs = outputs["pred_logits"][i].float().softmax(-1)
            scores, cls_ids = probs[:, :-1].max(-1)
            keep = scores > score_thr
            if not keep.any():
                continue
            sc, cid = scores[keep], cls_ids[keep]
            bx = box_convert(outputs["pred_boxes"][i][keep].float() * img_size, "cxcywh", "xyxy")
            pad_x, pad_y = target["pad_x"], target["pad_y"]
            scale = target["scale"]
            orig_h, orig_w = target["orig_size"].tolist()
            bx[:, [0, 2]] = ((bx[:, [0, 2]] - pad_x) / scale).clamp(0, orig_w)
            bx[:, [1, 3]] = ((bx[:, [1, 3]] - pad_y) / scale).clamp(0, orig_h)

            # Per-class NMS — matches infer.py
            bx, sc, cid = _per_class_nms(bx, sc, cid, nms_iou)

            # Max-dets cap — matches infer.py
            if len(sc) > max_dets:
                topk = sc.topk(max_dets)
                bx, cid, sc = bx[topk.indices], cid[topk.indices], topk.values

            boxes_xywh = box_convert(bx, "xyxy", "xywh")
            img_id = target["image_id"].item()
            preds.extend({"image_id": img_id, "category_id": int(c) + 1,
                          "bbox": box, "score": float(s)}
                         for s, c, box in zip(sc.tolist(), cid.tolist(), boxes_xywh.tolist()))

    if not preds:
        print("  [mAP] n_preds=0")
        return 0.0
    print(f"  [mAP] n_preds={len(preds)}")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(val_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(preds, f)
        coco_dt = coco_gt.loadRes(f.name)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def plot_curves(train_losses: list, val_losses: list, output_dir: Path) -> None:
    """Save training and validation loss curves to output_dir/loss_curves.png."""
    epochs = range(len(train_losses))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val Loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve → {path}")


@torch.no_grad()
def compute_confusion_matrix(model, loader, device, img_size: int,
                              amp_enabled: bool, amp_dtype,
                              output_dir: Path,
                              score_thr: float = 0.05,
                              iou_thr: float = 0.5) -> None:
    """Match predictions to GT via IoU, build and save a 10×10 confusion matrix.

    Rows = GT class (0-9), Columns = predicted class (0-9).
    Unmatched GTs are counted in a 'background' column (col 10).
    """
    num_cls = 10
    # shape (num_cls, num_cls + 1): last col = missed (predicted as background)
    cm = np.zeros((num_cls, num_cls + 1), dtype=np.int64)

    model.eval()
    for imgs, targets in tqdm(loader, desc="  Confusion matrix", leave=False, dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(imgs)

        for i, target in enumerate(targets):
            probs = outputs["pred_logits"][i].float().softmax(-1)
            scores, pred_cls = probs[:, :-1].max(-1)
            keep = scores > score_thr

            gt_boxes  = target["boxes"]   # normalised cxcywh
            gt_labels = target["labels"]  # 0-indexed
            if len(gt_boxes) == 0:
                continue

            # Convert GT to xyxy pixel space
            gt_xyxy = box_convert(gt_boxes.float() * img_size, "cxcywh", "xyxy")

            if not keep.any():
                # All GT unmatched
                for lbl in gt_labels.tolist():
                    cm[lbl, num_cls] += 1
                continue

            # Convert predictions to xyxy pixel space
            pred_xyxy = box_convert(
                outputs["pred_boxes"][i][keep].float() * img_size, "cxcywh", "xyxy")
            pred_lbl = pred_cls[keep]

            # IoU matrix: (num_gt, num_pred)
            iou = box_iou(gt_xyxy.to(device), pred_xyxy)
            matched_pred = set()
            for g in range(len(gt_boxes)):
                gt_lbl = gt_labels[g].item()
                best_iou, best_p = iou[g].max(0)
                best_p = best_p.item()
                if best_iou.item() >= iou_thr and best_p not in matched_pred:
                    matched_pred.add(best_p)
                    cm[gt_lbl, pred_lbl[best_p].item()] += 1
                else:
                    cm[gt_lbl, num_cls] += 1  # missed / wrong location

    # Plot
    digit_labels = [str(i) for i in range(num_cls)]
    col_labels = digit_labels + ["miss"]
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_cls + 1))
    ax.set_yticks(range(num_cls))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(digit_labels, fontsize=9)
    ax.set_xlabel("Predicted class  (last col = missed)")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(f"Confusion Matrix  (score_thr={score_thr}, iou_thr={iou_thr})")
    # Annotate cells
    thresh = cm.max() / 2.0
    for r in range(num_cls):
        for c in range(num_cls + 1):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7,
                    color="white" if cm[r, c] > thresh else "black")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    amp_enabled, amp_dtype, use_scaler = _amp_settings(device, args.amp)
    scaler = GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    print(f"Device: {device} | AMP: {args.amp} | GradScaler: {use_scaler}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_ds = DigitDetectionDataset(data_dir / "train", data_dir / "train.json",
                                     img_size=args.img_size, augment=True)
    val_ds = DigitDetectionDataset(data_dir / "valid", data_dir / "valid.json",
                                   img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"lr={args.lr:.1e}  lr_backbone={args.lr_backbone:.1e}  warmup={args.warmup_epochs}  epochs={args.epochs}")

    model = DETR(num_classes=10, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 num_queries=args.num_queries, pretrained=True).to(device)

    # Matcher costs and loss weights are kept identical so Hungarian assignment
    # and gradient signals optimise for the same objective.
    matcher = HungarianMatcher(cost_class=args.loss_ce, cost_bbox=args.loss_bbox,
                               cost_giou=args.loss_giou)
    criterion = DETRLoss(num_classes=10, matcher=matcher,
                         weight_dict={"loss_ce": args.loss_ce, "loss_bbox": args.loss_bbox,
                                      "loss_giou": args.loss_giou},
                         eos_coef=0.1).to(device)

    backbone_names = ("layer1", "layer2", "layer3", "layer4")
    backbone_ids = set(
        id(p) for n, p in model.named_parameters()
        if n.startswith(backbone_names)
    )
    optimizer = optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in backbone_ids],
         "lr": args.lr_backbone},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
         "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    start_epoch, best_map = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for pg, lr in zip(optimizer.param_groups, [args.lr_backbone, args.lr]):
            pg["lr"] = pg["initial_lr"] = lr
        scheduler = _build_scheduler(optimizer, args.warmup_epochs, args.epochs)
        start_epoch = ckpt["epoch"] + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_map = ckpt.get("best_map", -1.0)
        print(f"Resumed epoch {start_epoch} | best mAP {best_map:.4f} | "
              f"lr={scheduler.get_last_lr()[1]:.2e} lr_backbone={scheduler.get_last_lr()[0]:.2e}")

    val_json = str(data_dir / "valid.json")
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device,
                                     epoch, scaler, amp_enabled, amp_dtype, args.grad_clip)
        val_loss, val_ce, val_bbox, val_giou = evaluate_loss(
            model, criterion, val_loader, device, amp_enabled, amp_dtype)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        do_map = (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1)
        val_map = evaluate_map(model, val_loader, device, val_json, args.img_size,
                               amp_enabled, amp_dtype) if do_map else 0.0

        map_str = f" | mAP={val_map:.4f}" if do_map else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} (ce={val_ce:.3f} bbox={val_bbox:.3f} giou={val_giou:.3f})"
              f"{map_str} | lr={scheduler.get_last_lr()[1]:.2e} | {time.time()-t0:.1f}s")

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "best_map": best_map, "args": vars(args)}
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        if do_map and val_map > 0.0 and val_map > best_map:
            best_map = val_map
            ckpt["best_map"] = best_map
            torch.save(ckpt, output_dir / "best.pth")
            print(f"  -> Best model saved (mAP={val_map:.4f})")
        torch.save(ckpt, output_dir / "last.pth")

        # Save loss curve every eval interval so it's available mid-training
        if do_map:
            plot_curves(train_losses, val_losses, output_dir)

    print(f"Training complete. Best mAP = {best_map:.4f}")

    # Final plots using best checkpoint
    plot_curves(train_losses, val_losses, output_dir)
    best_ckpt = output_dir / "best.pth"
    if best_ckpt.exists():
        print("Computing confusion matrix on val set using best.pth …")
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
    compute_confusion_matrix(model, val_loader, device, args.img_size,
                             amp_enabled, amp_dtype, output_dir)


if __name__ == "__main__":
    main()