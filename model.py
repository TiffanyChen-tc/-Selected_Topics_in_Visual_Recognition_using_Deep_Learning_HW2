"""Deformable DETR

Architecture:
    - ResNet-50 backbone (C2–C5), all BatchNorm frozen.
    - Per-level 1×1 Conv + GroupNorm input projections.
    - Multi-scale deformable attention encoder and decoder.
    - Per-layer classification and box heads with iterative refinement.
    - Hungarian matching loss with auxiliary decoder outputs.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import box_convert, generalized_box_iou


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """2-D sin-cos positional embedding, shape (H*W, dim)."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij")
    gx = (gx + 0.5) / max(w, 1)
    gy = (gy + 0.5) / max(h, 1)
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32, device=device)
                              / max(dim // 4, 1)))
    ox = gx.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    oy = gy.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention
# ---------------------------------------------------------------------------

class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Pure-PyTorch , no CUDA).

    Each query predicts sampling offsets over n_levels x n_points locations
    per head, then aggregates bilinearly-interpolated values with learned weights.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_levels: int = 4, n_points: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.offset_proj = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attn_proj = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_proj.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.offset_proj.bias.copy_(grid.reshape(-1))
        nn.init.constant_(self.attn_proj.weight, 0.0)
        nn.init.constant_(self.attn_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, ref_pts: torch.Tensor,
                value: torch.Tensor,
                spatial_shapes: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Args:
            query: (B, Lq, d_model)
            ref_pts: (B, Lq, n_levels, 2) normalised [0, 1]
            value: (B, sum(H*W), d_model)
            spatial_shapes: list of (H, W) per level
        """
        B, Lq, _ = query.shape
        _, Lv, _ = value.shape

        v = self.value_proj(value).view(B, Lv, self.n_heads, self.head_dim)
        offsets = self.offset_proj(query).view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w = F.softmax(
            self.attn_proj(query).view(B, Lq, self.n_heads, self.n_levels * self.n_points), dim=-1
        ).view(B, Lq, self.n_heads, self.n_levels, self.n_points)

        sp = torch.tensor([[w, h] for h, w in spatial_shapes],
                          dtype=torch.float32, device=query.device)
        grids = 2.0 * (ref_pts[:, :, None, :, None, :] + offsets / sp[None, None, None, :, None, :]) - 1.0

        split_sizes = [h * w for h, w in spatial_shapes]
        sampled = []
        for lid, (h, w) in enumerate(spatial_shapes):
            vl = (v.split(split_sizes, dim=1)[lid]
                  .permute(0, 2, 3, 1).reshape(B * self.n_heads, self.head_dim, h, w))
            gl = (grids[:, :, :, lid, :, :]
                  .permute(0, 2, 1, 3, 4)
                  .reshape(B * self.n_heads, Lq, self.n_points, 2))
            sampled.append(F.grid_sample(vl, gl, mode="bilinear",
                                         padding_mode="zeros", align_corners=False))

        sampled = torch.cat(sampled, dim=-1)
        attn_w = (attn_w.view(B, Lq, self.n_heads, self.n_levels * self.n_points)
                  .permute(0, 2, 1, 3)
                  .reshape(B * self.n_heads, 1, Lq, self.n_levels * self.n_points))
        out = ((sampled * attn_w).sum(-1)
               .view(B, self.n_heads, self.head_dim, Lq)
               .permute(0, 3, 1, 2)
               .reshape(B, Lq, self.d_model))
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class DeformableEncoderBlock(nn.Module):
    """Single encoder block: deformable self-attention + FFN."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_levels: int = 4,
                 n_points: int = 4, d_ffn: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, ref_pts, spatial_shapes):
        src = self.norm1(src + self.drop1(self.self_attn(src + pos, ref_pts, src, spatial_shapes)))
        return self.norm2(src + self.drop2(self.ffn(src)))


class DeformableDecoderBlock(nn.Module):
    """Single decoder block: self-attention + deformable cross-attention + FFN."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_levels: int = 4,
                 n_points: int = 4, d_ffn: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.drop3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, memory, ref_pts, spatial_shapes):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.drop1(tgt2))
        tgt = self.norm2(tgt + self.drop2(
            self.cross_attn(tgt + query_pos, ref_pts, memory, spatial_shapes)))
        return self.norm3(tgt + self.drop3(self.ffn(tgt)))


# ---------------------------------------------------------------------------
# Deformable DETR
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """Deformable DETR with ResNet-50 C2-C5 backbone.

    Args:
        num_classes: Number of foreground classes (10 digits).
        num_queries: Number of object queries.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Encoder depth.
        num_decoder_layers: Decoder depth.
        dim_feedforward: FFN hidden size.
        dropout: Dropout probability.
        n_points: Deformable attention sampling points per level.
        pretrained: Whether to load ImageNet weights for the backbone.
    """

    def __init__(self, num_classes: int = 10, num_queries: int = 100,
                 d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 n_points: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_classes = num_classes
        self.n_levels = 4
        self.num_decoder_layers = num_decoder_layers

        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone_stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1)
        self.backbone_c3 = bb.layer2
        self.backbone_c4 = bb.layer3
        self.backbone_c5 = bb.layer4

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, d_model, 1), nn.GroupNorm(32, d_model))
            for ch in [256, 512, 1024, 2048]
        ])
        self.level_embed = nn.Parameter(torch.randn(self.n_levels, d_model))

        self.encoder = nn.ModuleList([
            DeformableEncoderBlock(d_model, nhead, self.n_levels, n_points, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DeformableDecoderBlock(d_model, nhead, self.n_levels, n_points, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        self.query_embed = nn.Embedding(num_queries, d_model * 2)
        self.ref_proj = nn.Linear(d_model, 4)

        self.class_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes + 1) for _ in range(num_decoder_layers)
        ])
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, 4))
            for _ in range(num_decoder_layers)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def _grid_ref_points(self, spatial_shapes, device):
        refs = []
        for h, w in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device) / h,
                torch.linspace(0.5, w - 0.5, w, device=device) / w,
                indexing="ij")
            refs.append(torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1))
        ref = torch.cat(refs, dim=0)
        return ref[:, None, :].repeat(1, len(spatial_shapes), 1)

    def forward(self, x: torch.Tensor) -> dict:
        B, device = x.size(0), x.device

        c2 = self.backbone_stem(x)
        c3 = self.backbone_c3(c2)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)

        srcs, poss, spatial_shapes = [], [], []
        for lid, feat in enumerate([c2, c3, c4, c5]):
            src = self.input_proj[lid](feat)
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            pos = (_sincos_pos_embed(h, w, self.d_model, device)
                   .unsqueeze(0).expand(B, -1, -1)
                   + self.level_embed[lid].view(1, 1, -1))
            srcs.append(src.flatten(2).permute(0, 2, 1))
            poss.append(pos)

        src_flat = torch.cat(srcs, dim=1)
        pos_flat = torch.cat(poss, dim=1)

        enc_ref = self._grid_ref_points(spatial_shapes, device).unsqueeze(0).expand(B, -1, -1, -1)
        memory = src_flat
        for block in self.encoder:
            memory = block(memory, pos_flat, enc_ref, spatial_shapes)

        query_pos, query_content = self.query_embed.weight.split(self.d_model, dim=-1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        tgt = query_content.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self.ref_proj(query_pos).sigmoid()

        dec_logits, dec_boxes = [], []
        for lid, block in enumerate(self.decoder):
            ref_for_attn = ref_pts[:, :, None, :2].expand(-1, -1, self.n_levels, -1)
            tgt = block(tgt, query_pos, memory, ref_for_attn, spatial_shapes)
            out = self.dec_norm(tgt)
            logits = self.class_heads[lid](out)
            boxes = (self.bbox_heads[lid](out) + _inverse_sigmoid(ref_pts)).sigmoid()
            dec_logits.append(logits)
            dec_boxes.append(boxes)
            ref_pts = boxes.detach()

        return {
            "pred_logits": dec_logits[-1],
            "pred_boxes": dec_boxes[-1],
            "aux_outputs": [{"pred_logits": c, "pred_boxes": b}
                            for c, b in zip(dec_logits[:-1], dec_boxes[:-1])],
        }


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Per-image bipartite matching via the Hungarian algorithm."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        indices = []
        for b in range(outputs["pred_logits"].shape[0]):
            tgt_lbl = targets[b]["labels"]
            tgt_box = targets[b]["boxes"]
            if tgt_lbl.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue
            pred_prob = outputs["pred_logits"][b].float().softmax(-1)
            pred_box = outputs["pred_boxes"][b].float()
            cost = (self.cost_class * (-pred_prob[:, tgt_lbl])
                    + self.cost_bbox * torch.cdist(pred_box, tgt_box.float(), p=1)
                    + self.cost_giou * (-generalized_box_iou(
                        box_convert(pred_box, "cxcywh", "xyxy"),
                        box_convert(tgt_box.float(), "cxcywh", "xyxy"))))
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
            r, c = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long),
                            torch.as_tensor(c, dtype=torch.long)))
        return indices


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DETRLoss(nn.Module):
    """Set-based loss: CE + L1 + GIoU with auxiliary decoder losses.

    Args:
        num_classes: Number of foreground classes.
        matcher: HungarianMatcher instance.
        weight_dict: Maps loss name to scalar weight.
        eos_coef: Down-weight factor for the no-object class.
        aux_weight: Scale factor for intermediate decoder layer losses.
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher,
                 weight_dict: dict, eos_coef: float = 0.1, aux_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.aux_weight = aux_weight
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict, targets: list) -> tuple:
        indices = self.matcher(outputs, targets)
        loss_ce = self._cls_loss(outputs, targets, indices)
        loss_bbox, loss_giou = self._box_loss(outputs, targets, indices)
        losses = {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}
        total = sum(self.weight_dict[k] * v for k, v in losses.items())
        for aux in outputs.get("aux_outputs", []):
            aux_idx = self.matcher(aux, targets)
            aux_ce = self._cls_loss(aux, targets, aux_idx)
            aux_bbox, aux_giou = self._box_loss(aux, targets, aux_idx)
            total = total + self.aux_weight * (
                self.weight_dict["loss_ce"] * aux_ce
                + self.weight_dict["loss_bbox"] * aux_bbox
                + self.weight_dict["loss_giou"] * aux_giou)
        return total, losses

    def _cls_loss(self, outputs, targets, indices):
        pred = outputs["pred_logits"]
        B, Q = pred.shape[:2]
        tgt_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=pred.device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                tgt_cls[i, src_idx] = targets[i]["labels"][tgt_idx]
        return F.cross_entropy(pred.flatten(0, 1), tgt_cls.flatten(), weight=self.empty_weight)

    def _box_loss(self, outputs, targets, indices):
        device = outputs["pred_boxes"].device
        src_boxes, tgt_boxes = [], []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes.append(outputs["pred_boxes"][i][src_idx])
            tgt_boxes.append(targets[i]["boxes"][tgt_idx])
        if not src_boxes:
            zero = torch.tensor(0.0, device=device)
            return zero, zero
        src = torch.cat(src_boxes).float().clamp(1e-6, 1 - 1e-6)
        tgt = torch.cat(tgt_boxes).float().clamp(1e-6, 1 - 1e-6)
        n = src.shape[0]
        loss_l1 = F.l1_loss(src, tgt, reduction="sum") / n

        def _to_xyxy(b):
            bx = box_convert(b, "cxcywh", "xyxy")
            return torch.stack([bx[:, 0], bx[:, 1],
                                torch.max(bx[:, 0] + 1e-4, bx[:, 2]),
                                torch.max(bx[:, 1] + 1e-4, bx[:, 3])], dim=1)

        giou = torch.diag(generalized_box_iou(_to_xyxy(src), _to_xyxy(tgt)))
        return loss_l1, (1 - torch.nan_to_num(giou, nan=0.0)).sum() / n