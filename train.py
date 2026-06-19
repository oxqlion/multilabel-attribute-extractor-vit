#!/usr/bin/env python3
# =============================================================================
#  FASHIONPEDIA — ConvNeXt-Tiny Attribute Extraction Fine-Tuning
#  Architecture : ConvNeXt-Tiny (IMAGENET1K_V2) + MLP classification head
#  Loss         : BCEWithLogitsLoss with per-class pos_weight
#  Strategy     : Partial fine-tuning, config-driven frozen/unfrozen stages
#  Device       : Apple Silicon MPS / CUDA / CPU (auto-detected)
#  Metrics      : mAP, macro-F1, Precision, Recall (per epoch)
#  Extras       : Cosine LR scheduler, checkpoint saving, training log CSV
# =============================================================================
#
#  USAGE
#  -----
#  python train.py                          # use defaults in CFG below
#  python train.py --config custom.json    # override CFG via JSON file
#
#  EXPECTED INPUTS  (produced by fashionpedia_preprocessing.py)
#  ──────────────────────────────────────────────────────────────
#  fashionpedia_processed/
#    train.csv                   ← ann_id, image_path, bbox_*, attribute_ids, …
#    val.csv
#    train_labels.npy            ← float32 multi-hot  (N_train, n_classes)
#    val_labels.npy              ← float32 multi-hot  (N_val,   n_classes)
#    metadata/
#      label_mappings.json       ← idx→attr_name, supercat_to_idxs, n_classes
#      pos_weights.json          ← per-class BCE pos_weight array
#
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import json
import time
import math
import shutil
import logging
import argparse
import warnings
import collections
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from PIL import Image

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fashionpedia_train")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

CFG = dict(

    # ── Paths ──────────────────────────────────────────────────────────────
    processed_dir   = "./fashionpedia_processed",   # output of preprocessing
    output_dir      = "./fashionpedia_runs",         # checkpoints + logs

    # ── Model ──────────────────────────────────────────────────────────────
    # ConvNeXt-Tiny has 8 feature blocks inside model.features (indices 0-7):
    #   [0] stem (Conv2d + LayerNorm)
    #   [1] stage-1 blocks      ← deepest, most general features
    #   [2] downsampling
    #   [3] stage-2 blocks
    #   [4] downsampling
    #   [5] stage-3 blocks
    #   [6] downsampling
    #   [7] stage-4 blocks      ← shallowest, most task-specific features
    #
    # 'unfreeze_from_block' controls which feature blocks are trainable.
    # Everything from this index onward (inclusive) is unfrozen.
    # The classifier head is ALWAYS trainable.
    #
    # Recommended settings:
    #   8  → Head only        (fastest, baseline)
    #   6  → Stage-4 + head   (good starting point)
    #   4  → Stages 3-4 + head
    #   2  → Stages 2-4 + head
    #   0  → Full fine-tune   (slow, may overfit without LR warmup)
    unfreeze_from_block = 6,        # ← unfreeze stage-4 (blocks 6,7) + head

    # Classification head architecture
    # Input: ConvNeXt-Tiny produces 768-dim features after adaptive avg pool
    # We add a 2-layer MLP head with dropout before the final linear layer.
    # This is better than a bare linear head for multi-label problems because:
    #   - The intermediate projection allows the model to re-compose features
    #     from ImageNet pretraining into fashion-domain attribute combinations.
    #   - Dropout regularises the head independently of backbone weight decay.
    head_hidden_dim    = 512,        # intermediate MLP projection dimension
    head_dropout       = 0.3,        # dropout probability in head

    # ── Training hyperparameters ───────────────────────────────────────────
    num_epochs         = 30,
    batch_size         = 64,         # adjust if MPS OOMs; try 32 first
    num_workers        = 4,          # DataLoader workers (4 is fine on M5)
    pin_memory         = False,      # must be False for MPS

    # ── Optimiser (AdamW) ──────────────────────────────────────────────────
    # We use two learning rates: one for the backbone (smaller) and one for
    # the head (larger).  This is standard practice in transfer learning —
    # pretrained backbone weights should change slowly.
    lr_head            = 1e-3,       # head learning rate
    lr_backbone        = 1e-4,       # backbone learning rate (10× smaller)
    weight_decay       = 1e-4,       # L2 regularisation

    # ── LR Scheduler (Cosine Annealing with Linear Warmup) ────────────────
    # Warmup is critical when fine-tuning pretrained weights.  Without it,
    # large initial gradients from the randomly-initialised head can destroy
    # the pretrained backbone representations in the first few steps.
    scheduler          = "cosine",   # "cosine" | "step"
    warmup_epochs      = 3,          # linear warmup for first N epochs
    # Cosine scheduler params
    cosine_t_max       = 27,         # cosine period (num_epochs - warmup_epochs)
    cosine_eta_min     = 1e-6,       # minimum LR at end of cosine cycle
    # Step scheduler params (used if scheduler == "step")
    step_size          = 10,
    step_gamma         = 0.1,

    # ── Loss ───────────────────────────────────────────────────────────────
    use_pos_weight     = True,       # load pos_weight from preprocessing
    # Cap pos_weight to avoid numerical instability with extremely rare classes.
    # In our retained label space this should be mild, but it's a safety net.
    pos_weight_cap     = 50.0,

    # ── Image preprocessing ────────────────────────────────────────────────
    img_size           = 224,        # ConvNeXt-Tiny standard input size
    # ImageNet normalisation stats (required when using IMAGENET1K_V2 weights)
    img_mean           = (0.485, 0.456, 0.406),
    img_std            = (0.229, 0.224, 0.225),

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_every_n_epochs = 5,         # save a periodic checkpoint every N epochs
    save_best           = True,      # always save the best-mAP checkpoint
    best_metric         = "val_mAP", # metric to monitor for best checkpoint

    # ── Evaluation ─────────────────────────────────────────────────────────
    # F1 / Precision / Recall threshold (applied to sigmoid output)
    threshold           = 0.5,
    # Also compute metrics at multiple thresholds for the final eval
    eval_thresholds     = [0.3, 0.4, 0.5, 0.6],

    # ── Reproducibility ────────────────────────────────────────────────────
    seed               = 42,
    run_name           = None,       # auto-generated if None

    # ── Debug ──────────────────────────────────────────────────────────────
    # Set to a small integer to run a fast sanity-check pass with N batches.
    # Set to None for full training.
    debug_batches      = None,
)


# =============================================================================
# SECTION 2 — UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Auto-detect the best available device.

    Priority: CUDA → MPS (Apple Silicon) → CPU

    MPS notes for M-series Macs:
    - Requires PyTorch >= 2.0 and macOS >= 12.3
    - pin_memory must be False (set in CFG)
    - Some ops fall back to CPU silently — this is fine
    - AMP (torch.autocast) is supported on MPS from PyTorch 2.3+
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        log.warning("Device: CPU — training will be slow")
    return device


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_run_dir(cfg: Dict) -> Path:
    """
    Create a timestamped run directory under output_dir.
    Returns the path.
    """
    run_name = cfg["run_name"] or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = Path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    # Save the config for this run
    with open(run_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in cfg.items()}, f, indent=2)
    log.info(f"Run directory: {run_dir}")
    return run_dir


# =============================================================================
# SECTION 3 — DATASET
# =============================================================================

class FashionpediaAttrDataset(Dataset):
    """
    Annotation-level crop dataset for Fashionpedia attribute extraction.

    Each item is one garment-part crop (bounding box region of an image)
    paired with its multi-hot attribute label vector.

    The training unit is the crop, not the full image.  This is the correct
    choice because Fashionpedia attributes describe individual garment parts,
    not the whole image.

    Args:
        processed_dir : path to fashionpedia_processed/ (preprocessing output)
        split         : "train" or "val"
        labels        : preloaded float32 numpy array of shape (N, n_classes)
        img_size      : crop resize target (224 for ConvNeXt-Tiny)
        augment       : whether to apply training augmentations
        img_mean      : normalisation mean (ImageNet default)
        img_std       : normalisation std  (ImageNet default)
    """

    def __init__(
        self,
        processed_dir: str,
        split: str,
        labels: np.ndarray,
        img_size: int = 224,
        augment: bool = False,
        img_mean: Tuple = (0.485, 0.456, 0.406),
        img_std:  Tuple = (0.229, 0.224, 0.225),
    ):
        assert split in ("train", "val"), f"Unknown split: {split}"
        processed_dir = Path(processed_dir)

        self.df     = pd.read_csv(processed_dir / f"{split}.csv")
        self.labels = labels
        self.split  = split

        assert len(self.df) == len(self.labels), (
            f"Row count mismatch: CSV={len(self.df)}, NPY={len(self.labels)}"
        )

        # ── Transforms ────────────────────────────────────────────────────
        # Training augmentation strategy — carefully chosen for fashion crops:
        #
        # ✓ RandomHorizontalFlip: garments are symmetric; safe to flip
        # ✓ ColorJitter: lighting/colour variation common in fashion photography
        # ✓ RandomResizedCrop: simulates slight zoom and framing variation
        # ✓ RandomRotation(±10°): small rotational invariance
        # ✗ RandomVerticalFlip: NOT used — upside-down garments are meaningless
        # ✗ Aggressive colour distortion: would remove texture patterns
        #     (stripe, check, floral) which are key attributes
        # ✗ Cutout/Erasing: hides attribute-discriminative regions

        if augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.7, 1.0),   # zoom in 70–100% of crop area
                    ratio=(0.75, 1.33), # aspect ratio variation
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.05,           # very small hue shift — colour matters
                ),
                T.ToTensor(),
                T.Normalize(img_mean, img_std),
            ])
        else:
            # Validation: deterministic resize + centre crop
            self.transform = T.Compose([
                T.Resize(
                    int(img_size * 1.14),  # slightly larger then centre crop
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(img_mean, img_std),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        label = torch.from_numpy(self.labels[idx]).float()

        # ── Load and crop image ───────────────────────────────────────────
        try:
            img = Image.open(row["image_path"]).convert("RGB")

            # Extract bbox, clamp to valid image boundaries
            x  = max(int(row["bbox_x"]), 0)
            y  = max(int(row["bbox_y"]), 0)
            x2 = min(x + int(row["bbox_w"]), img.width)
            y2 = min(y + int(row["bbox_h"]), img.height)

            # Guard against degenerate crops that slipped through
            if x2 <= x or y2 <= y:
                x, y, x2, y2 = 0, 0, img.width, img.height

            crop = img.crop((x, y, x2, y2))

        except (FileNotFoundError, OSError, Exception):
            # Graceful fallback: solid grey crop
            # In production, log this and investigate missing images
            crop = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return self.transform(crop), label


# =============================================================================
# SECTION 4 — MODEL
# =============================================================================

class FashionAttributeClassifier(nn.Module):
    """
    ConvNeXt-Tiny backbone + MLP classification head for multi-label
    attribute prediction.

    Architecture:
        ConvNeXt-Tiny (IMAGENET1K_V2 pretrained)
            ↓ features (8 sequential blocks)
            ↓ adaptive avg pool → (B, 768)
        MLP head:
            Linear(768 → head_hidden_dim)
            GELU
            Dropout(head_dropout)
            Linear(head_hidden_dim → n_classes)
        Output: raw logits (B, n_classes)  ← BCEWithLogitsLoss expects logits

    Head design rationale:
        A 2-layer MLP is preferred over a bare linear layer for this task:
        - Multi-label problems benefit from a non-linear composition step
          before the final per-class sigmoid decision.
        - GELU is used (not ReLU) because ConvNeXt itself uses GELU — keeping
          activation functions consistent aids gradient flow at the boundary.
        - Dropout=0.3 provides regularisation independent of the backbone's
          weight decay, which is important since the head is randomly initialised
          and trained with a higher learning rate than the backbone.

    Partial fine-tuning:
        We freeze backbone feature blocks [0 … unfreeze_from_block-1] and
        leave [unfreeze_from_block … 7] + head trainable.  The LayerNorm
        parameters inside frozen blocks are also frozen — this differs from
        some implementations that unfreeze BN stats; for ConvNeXt (which uses
        LayerNorm, not BatchNorm) full freezing is correct.
    """

    def __init__(
        self,
        n_classes: int,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        unfreeze_from_block: int = 6,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        weights  = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)

        # Remove the original ImageNet classifier (Linear(768 → 1000))
        # Keep: backbone.features (8 blocks) + backbone.avgpool
        self.features = backbone.features  # nn.Sequential of 8 blocks
        self.avgpool  = backbone.avgpool   # AdaptiveAvgPool2d → (B, 768, 1, 1)

        # Feature dimension: ConvNeXt-Tiny outputs 768-d after avgpool
        self.feat_dim = 768

        # ── MLP Classification Head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),                              # (B, 768, 1, 1) → (B, 768)
            nn.LayerNorm(self.feat_dim),               # stabilise features before head
            nn.Linear(self.feat_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, n_classes),     # raw logits
        )

        # ── Partial Freezing ───────────────────────────────────────────────
        self._apply_partial_freeze(unfreeze_from_block)

        # ── Weight initialisation for new head ────────────────────────────
        # Kaiming uniform for Linear layers (standard for ReLU/GELU networks)
        # Small bias init to avoid initial BCE loss explosion
        self._init_head()

    def _apply_partial_freeze(self, unfreeze_from_block: int) -> None:
        """
        Freeze feature blocks [0 … unfreeze_from_block-1].
        Unfreeze blocks [unfreeze_from_block … 7] + avgpool.
        Head is always trainable (initialised randomly).

        unfreeze_from_block=8 → freeze everything (head-only training)
        unfreeze_from_block=0 → unfreeze everything (full fine-tuning)
        """
        # First freeze all backbone params
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

        # Then selectively unfreeze from the specified block onward
        for block_idx in range(unfreeze_from_block, 8):
            for param in self.features[block_idx].parameters():
                param.requires_grad = True
        # avgpool has no learnable params in ConvNeXt-Tiny, but unfreeze anyway
        for param in self.avgpool.parameters():
            param.requires_grad = True

        # Log what was frozen
        frozen_params   = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params     = frozen_params + trainable_params
        log.info(f"Partial freeze: blocks 0–{unfreeze_from_block-1} frozen")
        log.info(f"  Trainable params: {trainable_params/1e6:.2f}M / "
                 f"{total_params/1e6:.2f}M total "
                 f"({trainable_params/total_params*100:.1f}%)")

    def _init_head(self) -> None:
        """Initialise new head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input:  (B, 3, 224, 224)
        Output: (B, n_classes) — raw logits, no sigmoid
        """
        x = self.features(x)   # (B, 768, 7, 7)
        x = self.avgpool(x)    # (B, 768, 1, 1)
        x = self.head(x)       # (B, n_classes)
        return x

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[Dict]:
        """
        Return separate parameter groups for differential learning rates.

        Why differential LRs?
        The pretrained backbone weights are already well-optimised for visual
        feature extraction.  We want to adjust them slowly (lr_backbone) while
        training the new head quickly (lr_head).  Updating both at the same rate
        risks either not adapting the backbone at all, or catastrophically
        forgetting its ImageNet representations.

        We also separate weight decay:
        - Apply weight_decay to all weight parameters
        - Do NOT apply weight_decay to biases or LayerNorm parameters
          (applying decay to these can impair normalisation stability)
        """
        backbone_params_decay  = []
        backbone_params_no_dec = []
        head_params_decay      = []
        head_params_no_dec     = []

        def is_nodecay(name: str) -> bool:
            return any(nd in name for nd in ["bias", "norm.weight", "layernorm"])

        for name, param in self.features.named_parameters():
            if not param.requires_grad:
                continue
            if is_nodecay(name):
                backbone_params_no_dec.append(param)
            else:
                backbone_params_decay.append(param)

        for name, param in self.head.named_parameters():
            if is_nodecay(name):
                head_params_no_dec.append(param)
            else:
                head_params_decay.append(param)

        return [
            {"params": backbone_params_decay,  "lr": lr_backbone, "weight_decay": weight_decay,   "name": "backbone_decay"},
            {"params": backbone_params_no_dec, "lr": lr_backbone, "weight_decay": 0.0,            "name": "backbone_no_decay"},
            {"params": head_params_decay,      "lr": lr_head,     "weight_decay": weight_decay,   "name": "head_decay"},
            {"params": head_params_no_dec,     "lr": lr_head,     "weight_decay": 0.0,            "name": "head_no_decay"},
        ]


# =============================================================================
# SECTION 5 — LOSS
# =============================================================================

def build_loss_fn(
    processed_dir: Path,
    use_pos_weight: bool,
    pos_weight_cap: float,
    device: torch.device,
) -> nn.BCEWithLogitsLoss:
    """
    Build BCEWithLogitsLoss with optional per-class positive weighting.

    pos_weight[i] = n_negative_i / n_positive_i

    This counteracts the extreme class imbalance in Fashionpedia attributes
    (e.g. 'plain (pattern)' appears in ~58K annotations while 'embossed'
    appears in only ~86 — a 675× imbalance).

    Without pos_weight, the model would learn to always predict 0 for rare
    classes (which achieves near-zero BCE loss due to the imbalance) and
    miss most true positives.

    Capping pos_weight at pos_weight_cap prevents numerical instability for
    extremely rare retained classes.  The cap is applied before passing to
    BCEWithLogitsLoss to keep the loss scale bounded.
    """
    if use_pos_weight:
        pw_path = processed_dir / "metadata" / "pos_weights.json"
        if pw_path.exists():
            pw_data     = load_json(pw_path)
            pw_array    = np.array(pw_data["pos_weight"], dtype=np.float32)
            pw_array    = np.clip(pw_array, 1.0, pos_weight_cap)
            pos_weight  = torch.tensor(pw_array, device=device)
            log.info(f"pos_weight loaded: n={len(pw_array)}, "
                     f"min={pw_array.min():.1f}, "
                     f"median={np.median(pw_array):.1f}, "
                     f"max (after cap)={pw_array.max():.1f}")
        else:
            log.warning(f"pos_weights.json not found at {pw_path}. "
                        "Using unweighted BCE.")
            pos_weight = None
    else:
        pos_weight = None
        log.info("Unweighted BCEWithLogitsLoss (use_pos_weight=False)")

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")


# =============================================================================
# SECTION 6 — SCHEDULER (WARMUP + COSINE / STEP)
# =============================================================================

class WarmupScheduler:
    """
    Linear warmup wrapper over a base PyTorch LR scheduler.

    For the first `warmup_epochs` epochs, all learning rates are linearly
    scaled from 0 → their configured value.  After warmup, the base scheduler
    takes over.

    Why warmup?
    At epoch 0, the head weights are random.  The first backward pass produces
    large gradients from the head that propagate into the backbone.  Without
    warmup, these can destabilise the pretrained features before the head has
    learned to output reasonable activations.  Warmup gives the head a few
    epochs to stabilise before the backbone is updated significantly.
    """

    def __init__(
        self,
        optimizer:     optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: optim.lr_scheduler._LRScheduler,
    ):
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.base_scheduler  = base_scheduler
        self._base_lrs       = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch          = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear warmup: scale each group's LR proportionally
            scale = self._epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self.base_scheduler.step()

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        return {
            "_epoch":       self._epoch,
            "_base_lrs":    self._base_lrs,
            "base_sched":   self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self._epoch     = state["_epoch"]
        self._base_lrs  = state["_base_lrs"]
        self.base_scheduler.load_state_dict(state["base_sched"])


def build_scheduler(cfg: Dict, optimizer: optim.Optimizer) -> WarmupScheduler:
    """
    Build the learning rate scheduler based on cfg['scheduler'].

    Supported: "cosine" (recommended), "step"
    """
    if cfg["scheduler"] == "cosine":
        base_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max    = cfg["cosine_t_max"],
            eta_min  = cfg["cosine_eta_min"],
        )
        log.info(f"Scheduler: CosineAnnealingLR "
                 f"(T_max={cfg['cosine_t_max']}, eta_min={cfg['cosine_eta_min']})")
    elif cfg["scheduler"] == "step":
        base_sched = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = cfg["step_size"],
            gamma     = cfg["step_gamma"],
        )
        log.info(f"Scheduler: StepLR "
                 f"(step_size={cfg['step_size']}, gamma={cfg['step_gamma']})")
    else:
        raise ValueError(f"Unknown scheduler: {cfg['scheduler']}. "
                         "Choose 'cosine' or 'step'.")

    scheduler = WarmupScheduler(
        optimizer      = optimizer,
        warmup_epochs  = cfg["warmup_epochs"],
        base_scheduler = base_sched,
    )
    log.info(f"  + LinearWarmup for first {cfg['warmup_epochs']} epochs")
    return scheduler


# =============================================================================
# SECTION 7 — METRICS
# =============================================================================

class MetricTracker:
    """
    Accumulates batch predictions and ground-truth labels, then computes
    epoch-level metrics in one pass.

    Metrics computed:
      - mAP  : mean Average Precision (macro-averaged over classes)
              Standard metric for multi-label classification.
              Threshold-independent (uses precision-recall curve).
      - F1   : macro-F1 at a fixed threshold
      - P    : macro-Precision at the same threshold
      - R    : macro-Recall at the same threshold

    We use sklearn's average_precision_score with average='macro'.
    This treats every class equally regardless of frequency — appropriate
    here because we already removed unlearnable rare classes in preprocessing.

    Note on threshold:
        mAP is threshold-free.
        F1/P/R require a threshold applied to sigmoid(logits).
        We evaluate at multiple thresholds in the final eval report.
    """

    def __init__(self, n_classes: int, threshold: float = 0.5):
        self.n_classes    = n_classes
        self.threshold    = threshold
        self._all_logits: List[np.ndarray] = []
        self._all_labels: List[np.ndarray] = []

    def reset(self) -> None:
        self._all_logits = []
        self._all_labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate batch logits and labels.
        Detaches from computation graph and moves to CPU.
        """
        self._all_logits.append(logits.detach().cpu().numpy())
        self._all_labels.append(labels.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.
        Returns dict with keys: mAP, F1, Precision, Recall.
        """
        logits = np.concatenate(self._all_logits, axis=0)  # (N, C)
        labels = np.concatenate(self._all_labels, axis=0)  # (N, C)
        probs  = 1 / (1 + np.exp(-logits))                 # sigmoid, numerically stable
        preds  = (probs >= self.threshold).astype(np.int32)

        # ── mAP: skip classes with no positive samples in this batch ──────
        # average_precision_score raises if a class has only 0s in labels.
        valid_classes = np.where(labels.sum(axis=0) > 0)[0]
        if len(valid_classes) < self.n_classes:
            # Only a warning — can happen on small val sets or early epochs
            pass

        try:
            mAP = average_precision_score(
                labels[:, valid_classes],
                probs[:, valid_classes],
                average="macro",
            )
        except ValueError:
            mAP = 0.0

        # ── F1, Precision, Recall at fixed threshold ───────────────────────
        # zero_division=0 avoids divide-by-zero warnings for classes that
        # have no predicted positives (common early in training for rare classes)
        f1  = f1_score(labels,  preds, average="macro",  zero_division=0)
        p   = precision_score(labels, preds, average="macro", zero_division=0)
        r   = recall_score(labels,   preds, average="macro",  zero_division=0)

        return {
            "mAP":       float(mAP),
            "F1":        float(f1),
            "Precision": float(p),
            "Recall":    float(r),
        }

    def compute_at_thresholds(
        self, thresholds: List[float]
    ) -> pd.DataFrame:
        """Compute F1/P/R at multiple thresholds (for final eval report)."""
        logits = np.concatenate(self._all_logits, axis=0)
        labels = np.concatenate(self._all_labels, axis=0)
        probs  = 1 / (1 + np.exp(-logits))

        rows = []
        for thr in thresholds:
            preds = (probs >= thr).astype(np.int32)
            rows.append({
                "threshold": thr,
                "F1":        float(f1_score(labels, preds, average="macro",  zero_division=0)),
                "Precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
                "Recall":    float(recall_score(labels,   preds, average="macro",  zero_division=0)),
            })
        return pd.DataFrame(rows)


# =============================================================================
# SECTION 8 — CHECKPOINTING
# =============================================================================

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Saves:
      - last.pt          : most recent epoch (always overwritten)
      - best.pt          : best validation mAP checkpoint
      - epoch_{N:03d}.pt : periodic checkpoint every save_every_n_epochs

    Checkpoint contents:
      - epoch            : int
      - model_state      : model.state_dict()
      - optimizer_state  : optimizer.state_dict()
      - scheduler_state  : scheduler.state_dict()
      - best_metric_val  : float
      - history          : list of epoch metric dicts
      - config           : training config
    """

    def __init__(
        self,
        ckpt_dir:           Path,
        save_every_n_epochs: int = 5,
        best_metric_key:     str = "val_mAP",
    ):
        self.ckpt_dir           = Path(ckpt_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.best_metric_key    = best_metric_key
        self.best_metric_val    = -float("inf")

    def save(
        self,
        epoch:     int,
        model:     nn.Module,
        optimizer: optim.Optimizer,
        scheduler: WarmupScheduler,
        history:   List[Dict],
        config:    Dict,
    ) -> None:
        """Save checkpoint after each epoch."""
        payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric_val": self.best_metric_val,
            "history":         history,
            "config":          {k: str(v) if isinstance(v, Path) else v
                                for k, v in config.items()},
        }

        # Always save last checkpoint
        torch.save(payload, self.ckpt_dir / "last.pt")

        # Periodic checkpoint
        if epoch % self.save_every_n_epochs == 0:
            torch.save(payload, self.ckpt_dir / f"epoch_{epoch:03d}.pt")
            log.info(f"  [ckpt] Saved periodic checkpoint: epoch_{epoch:03d}.pt")

        # Best checkpoint
        current_val = history[-1].get(self.best_metric_key, -float("inf"))
        if current_val > self.best_metric_val:
            self.best_metric_val = current_val
            torch.save(payload, self.ckpt_dir / "best.pt")
            log.info(f"  [ckpt] ★ New best {self.best_metric_key}={current_val:.4f} "
                     f"→ saved best.pt")

    def load(
        self,
        path:      Path,
        model:     nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[WarmupScheduler] = None,
        device:    torch.device = torch.device("cpu"),
    ) -> Tuple[int, List[Dict]]:
        """
        Load a checkpoint.  Returns (start_epoch, history).
        Pass optimizer=None to load model weights only (inference mode).
        """
        log.info(f"Loading checkpoint: {path}")
        payload = torch.load(path, map_location=device)

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])

        start_epoch = payload["epoch"] + 1
        history     = payload.get("history", [])
        self.best_metric_val = payload.get("best_metric_val", -float("inf"))

        log.info(f"  Resumed from epoch {payload['epoch']} | "
                 f"best {self.best_metric_key}={self.best_metric_val:.4f}")
        return start_epoch, history


# =============================================================================
# SECTION 9 — TRAINING & VALIDATION LOOPS
# =============================================================================

def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    criterion:     nn.Module,
    optimizer:     optim.Optimizer,
    device:        torch.device,
    epoch:         int,
    n_epochs:      int,
    debug_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns dict with keys: loss
    (Per-class and per-sample metrics are computed on val, not train,
    to avoid the overhead of accumulating all train predictions.)
    """
    model.train()
    total_loss   = 0.0
    total_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):

        # Early exit for debug mode
        if debug_batches is not None and batch_idx >= debug_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
        logits = model(images)                  # (B, n_classes)
        loss   = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        loss.backward()

        # Gradient clipping — prevents gradient explosion, especially important
        # in the first few epochs when the head gradients are large.
        # max_norm=1.0 is a conservative default for AdamW.
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1

        # ── Progress log every 100 batches ────────────────────────────────
        if (batch_idx + 1) % 100 == 0:
            avg = total_loss / total_batches
            log.info(
                f"  Epoch [{epoch}/{n_epochs}] "
                f"Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {avg:.4f}"
            )

    return {"train_loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def validate(
    model:          nn.Module,
    loader:         DataLoader,
    criterion:      nn.Module,
    device:         torch.device,
    n_classes:      int,
    threshold:      float = 0.5,
    debug_batches:  Optional[int] = None,
) -> Dict[str, float]:
    """
    Run validation.

    Accumulates all logits and labels, then computes:
      - val_loss   : mean BCE loss
      - val_mAP    : macro-average precision
      - val_F1     : macro-F1
      - val_P      : macro-Precision
      - val_R      : macro-Recall
    """
    model.eval()
    total_loss   = 0.0
    total_batches = 0
    tracker = MetricTracker(n_classes=n_classes, threshold=threshold)

    for batch_idx, (images, labels) in enumerate(loader):

        if debug_batches is not None and batch_idx >= debug_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item()
        total_batches += 1
        tracker.update(logits, labels)

    metrics = tracker.compute()
    metrics["val_loss"] = total_loss / max(total_batches, 1)

    # Rename keys with val_ prefix
    return {
        "val_loss":      metrics["val_loss"],
        "val_mAP":       metrics["mAP"],
        "val_F1":        metrics["F1"],
        "val_Precision": metrics["Precision"],
        "val_Recall":    metrics["Recall"],
        "_tracker":      tracker,   # keep tracker for threshold sweep
    }


# =============================================================================
# SECTION 10 — TRAINING LOGGER (CSV)
# =============================================================================

class TrainingLogger:
    """
    Writes per-epoch metrics to a CSV log file.

    Columns: epoch, timestamp, train_loss, val_loss, val_mAP,
             val_F1, val_Precision, val_Recall, lr_head, lr_backbone
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._header_written = False

    def log(self, row: Dict) -> None:
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([row])
        df.to_csv(
            self.log_path,
            mode="a",
            header=not self._header_written,
            index=False,
        )
        self._header_written = True


# =============================================================================
# SECTION 11 — MAIN TRAINING FUNCTION
# =============================================================================

def train(cfg: Dict, resume_from: Optional[str] = None) -> None:
    """
    Main training entry point.

    Args:
        cfg         : configuration dictionary (see CFG at top of file)
        resume_from : path to a checkpoint .pt file to resume training from
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device  = get_device()
    run_dir = make_run_dir(cfg)

    log.info("=" * 70)
    log.info(" FASHIONPEDIA ATTRIBUTE EXTRACTION — TRAINING")
    log.info("=" * 70)
    log.info(f"  Run dir        : {run_dir}")
    log.info(f"  Device         : {device}")
    log.info(f"  Epochs         : {cfg['num_epochs']}")
    log.info(f"  Batch size     : {cfg['batch_size']}")
    log.info(f"  Unfreeze from  : block {cfg['unfreeze_from_block']}")
    log.info(f"  LR (head)      : {cfg['lr_head']}")
    log.info(f"  LR (backbone)  : {cfg['lr_backbone']}")
    log.info("=" * 70)

    processed_dir = Path(cfg["processed_dir"])

    # ── Load label mappings ────────────────────────────────────────────────
    label_mappings = load_json(processed_dir / "metadata" / "label_mappings.json")
    n_classes      = label_mappings["n_classes"]
    log.info(f"Label space: {n_classes} attributes")

    # ── Load pre-built multi-hot label matrices ────────────────────────────
    # Loading from .npy is ~100× faster than re-computing from CSV strings,
    # which matters when num_workers > 0 in the DataLoader.
    train_labels = np.load(processed_dir / "train_labels.npy")
    val_labels   = np.load(processed_dir / "val_labels.npy")
    log.info(f"Labels loaded: train={train_labels.shape}, val={val_labels.shape}")

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = FashionpediaAttrDataset(
        processed_dir = processed_dir,
        split         = "train",
        labels        = train_labels,
        img_size      = cfg["img_size"],
        augment       = True,
        img_mean      = cfg["img_mean"],
        img_std       = cfg["img_std"],
    )
    val_dataset = FashionpediaAttrDataset(
        processed_dir = processed_dir,
        split         = "val",
        labels        = val_labels,
        img_size      = cfg["img_size"],
        augment       = False,
        img_mean      = cfg["img_mean"],
        img_std       = cfg["img_std"],
    )

    # DataLoader notes for MPS (Apple Silicon):
    # - pin_memory=False is required (MPS uses unified memory — pinning is
    #   a no-op and raises a warning if True)
    # - num_workers=4 works well; more can cause issues on some macOS versions
    # - persistent_workers=True avoids worker restart overhead between epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size        = cfg["batch_size"],
        shuffle           = True,
        num_workers       = cfg["num_workers"],
        pin_memory        = cfg["pin_memory"],
        persistent_workers= cfg["num_workers"] > 0,
        drop_last         = True,    # drop last incomplete batch for stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size        = cfg["batch_size"] * 2,  # no grad → 2× batch size
        shuffle           = False,
        num_workers       = cfg["num_workers"],
        pin_memory        = cfg["pin_memory"],
        persistent_workers= cfg["num_workers"] > 0,
        drop_last         = False,
    )
    log.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches/epoch")
    log.info(f"Val  : {len(val_dataset):,} samples, {len(val_loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FashionAttributeClassifier(
        n_classes           = n_classes,
        head_hidden_dim     = cfg["head_hidden_dim"],
        head_dropout        = cfg["head_dropout"],
        unfreeze_from_block = cfg["unfreeze_from_block"],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = build_loss_fn(
        processed_dir  = processed_dir,
        use_pos_weight = cfg["use_pos_weight"],
        pos_weight_cap = cfg["pos_weight_cap"],
        device         = device,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(
        lr_backbone  = cfg["lr_backbone"],
        lr_head      = cfg["lr_head"],
        weight_decay = cfg["weight_decay"],
    )
    optimizer = optim.AdamW(param_groups)
    log.info(f"Optimiser: AdamW | "
             f"LR backbone={cfg['lr_backbone']}, LR head={cfg['lr_head']}, "
             f"WD={cfg['weight_decay']}")

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scheduler = build_scheduler(cfg, optimizer)

    # ── Checkpointing & Logging ───────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        ckpt_dir            = run_dir / "checkpoints",
        save_every_n_epochs = cfg["save_every_n_epochs"],
        best_metric_key     = cfg["best_metric"],
    )
    csv_logger = TrainingLogger(run_dir / "training_log.csv")

    history     : List[Dict] = []
    start_epoch : int        = 1

    # ── Resume from checkpoint ────────────────────────────────────────────
    if resume_from:
        start_epoch, history = ckpt_manager.load(
            path=Path(resume_from), model=model,
            optimizer=optimizer, scheduler=scheduler, device=device,
        )

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("\nStarting training …\n")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model         = model,
            loader        = train_loader,
            criterion     = criterion,
            optimizer     = optimizer,
            device        = device,
            epoch         = epoch,
            n_epochs      = cfg["num_epochs"],
            debug_batches = cfg["debug_batches"],
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = validate(
            model         = model,
            loader        = val_loader,
            criterion     = criterion,
            device        = device,
            n_classes     = n_classes,
            threshold     = cfg["threshold"],
            debug_batches = cfg["debug_batches"],
        )
        tracker = val_metrics.pop("_tracker")  # remove tracker from metrics dict

        # ── Step LR Scheduler ─────────────────────────────────────────────
        scheduler.step()
        current_lrs = scheduler.get_last_lr()

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        epoch_metrics = {
            "epoch":        epoch,
            **train_metrics,
            **val_metrics,
            "lr_head":      current_lrs[2] if len(current_lrs) > 2 else current_lrs[0],
            "lr_backbone":  current_lrs[0],
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_metrics)
        csv_logger.log(epoch_metrics)

        log.info(
            f"Epoch [{epoch:3d}/{cfg['num_epochs']}] "
            f"| Train Loss: {train_metrics['train_loss']:.4f} "
            f"| Val Loss: {val_metrics['val_loss']:.4f} "
            f"| Val mAP: {val_metrics['val_mAP']:.4f} "
            f"| Val F1: {val_metrics['val_F1']:.4f} "
            f"| P: {val_metrics['val_Precision']:.4f} "
            f"| R: {val_metrics['val_Recall']:.4f} "
            f"| LR_head: {epoch_metrics['lr_head']:.2e} "
            f"| {epoch_time:.0f}s"
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt_manager.save(
            epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, history=history, config=cfg,
        )

    # ── Final evaluation at multiple thresholds ───────────────────────────
    log.info("\n── Final threshold sweep on validation set ──")
    log.info("Loading best.pt for final evaluation …")

    best_ckpt_path = run_dir / "checkpoints" / "best.pt"
    if best_ckpt_path.exists():
        ckpt_manager.load(
            path=best_ckpt_path, model=model, device=device,
        )

    # Re-run full validation with the best model
    final_val = validate(
        model=model, loader=val_loader, criterion=criterion,
        device=device, n_classes=n_classes,
        threshold=cfg["threshold"],
    )
    final_tracker = final_val.pop("_tracker")

    thresh_df = final_tracker.compute_at_thresholds(cfg["eval_thresholds"])
    thresh_path = run_dir / "threshold_sweep.csv"
    thresh_df.to_csv(thresh_path, index=False)
    log.info(f"Threshold sweep saved → {thresh_path}")
    log.info("\n" + thresh_df.to_string(index=False))

    # ── Training summary ──────────────────────────────────────────────────
    best_epoch = max(history, key=lambda h: h.get("val_mAP", 0))
    log.info("\n" + "=" * 70)
    log.info(" TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Best epoch     : {best_epoch['epoch']}")
    log.info(f"  Best val mAP   : {best_epoch['val_mAP']:.4f}")
    log.info(f"  Best val F1    : {best_epoch['val_F1']:.4f}")
    log.info(f"  Best Precision : {best_epoch['val_Precision']:.4f}")
    log.info(f"  Best Recall    : {best_epoch['val_Recall']:.4f}")
    log.info(f"  Checkpoints    : {run_dir / 'checkpoints'}")
    log.info(f"  Training log   : {run_dir / 'training_log.csv'}")
    log.info("=" * 70)


# =============================================================================
# SECTION 12 — INFERENCE UTILITY
# =============================================================================

class FashionAttributeInference:
    """
    Convenience class for running inference with a trained checkpoint.

    Usage:
        inferencer = FashionAttributeInference(
            checkpoint_path  = "./fashionpedia_runs/run_xxx/checkpoints/best.pt",
            processed_dir    = "./fashionpedia_processed",
        )
        attrs = inferencer.predict_from_crop(image_pil, bbox=(x, y, w, h))
        print(attrs)  # [{'name': 'plain (pattern)', 'prob': 0.92}, …]
    """

    def __init__(
        self,
        checkpoint_path: str,
        processed_dir:   str,
        threshold:       float = 0.5,
        device:          Optional[str] = None,
    ):
        self.threshold    = threshold
        self.device       = (
            torch.device(device) if device
            else get_device()
        )
        processed_dir     = Path(processed_dir)
        self.mappings     = load_json(processed_dir / "metadata" / "label_mappings.json")
        n_classes         = self.mappings["n_classes"]

        # Load checkpoint
        ckpt              = torch.load(checkpoint_path, map_location=self.device)
        train_cfg         = ckpt.get("config", {})

        self.model = FashionAttributeClassifier(
            n_classes           = n_classes,
            head_hidden_dim     = int(train_cfg.get("head_hidden_dim", 512)),
            head_dropout        = 0.0,  # disable dropout at inference
            unfreeze_from_block = int(train_cfg.get("unfreeze_from_block", 6)),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        log.info(f"Inference model loaded from {checkpoint_path}")
        log.info(f"  n_classes={n_classes}, device={self.device}")

    @torch.no_grad()
    def predict_from_crop(
        self,
        image:     Image.Image,
        bbox:      Optional[Tuple[int, int, int, int]] = None,
        top_k:     Optional[int] = None,
    ) -> List[Dict]:
        """
        Predict attributes from a PIL image and optional bounding box.

        Args:
            image  : PIL.Image (RGB)
            bbox   : (x, y, w, h) bounding box.  If None, uses full image.
            top_k  : if set, return only the top-k highest-probability attrs

        Returns:
            List of dicts: [{'attr_id': int, 'name': str, 'prob': float}, …]
            Sorted by probability descending.
        """
        if bbox is not None:
            x, y, w, h = bbox
            x2 = min(x + w, image.width)
            y2 = min(y + h, image.height)
            crop = image.crop((max(x, 0), max(y, 0), x2, y2))
        else:
            crop = image

        tensor = self.transform(crop).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        logits = self.model(tensor)[0]                               # (n_classes,)
        probs  = torch.sigmoid(logits).cpu().numpy()

        idx_to_attr_id   = {int(k): v for k, v in self.mappings["idx_to_attr_id"].items()}
        idx_to_attr_name = {int(k): v for k, v in self.mappings["idx_to_attr_name"].items()}

        if top_k is not None:
            top_indices = np.argsort(probs)[-top_k:][::-1]
            results = [
                {
                    "attr_id": idx_to_attr_id[i],
                    "name":    idx_to_attr_name[i],
                    "prob":    float(probs[i]),
                }
                for i in top_indices
            ]
        else:
            results = [
                {
                    "attr_id": idx_to_attr_id[i],
                    "name":    idx_to_attr_name[i],
                    "prob":    float(probs[i]),
                }
                for i in range(len(probs))
                if probs[i] >= self.threshold
            ]
            results.sort(key=lambda x: x["prob"], reverse=True)

        return results


# =============================================================================
# SECTION 13 — ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashionpedia ConvNeXt-Tiny attribute extraction training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file to override defaults in CFG",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--unfreeze-from",
        type=int,
        default=None,
        dest="unfreeze_from_block",
        help="Override unfreeze_from_block (0=full, 8=head-only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override batch_size",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Debug mode: run only N batches per epoch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    run_cfg = dict(CFG)  # copy defaults

    # Load JSON config override if provided
    if args.config:
        with open(args.config) as f:
            json_cfg = json.load(f)
        run_cfg.update(json_cfg)
        log.info(f"Config loaded from {args.config}")

    # Apply CLI overrides (highest priority)
    if args.unfreeze_from_block is not None:
        run_cfg["unfreeze_from_block"] = args.unfreeze_from_block
    if args.epochs is not None:
        run_cfg["num_epochs"]  = args.epochs
        run_cfg["cosine_t_max"] = args.epochs - run_cfg["warmup_epochs"]
    if args.batch_size is not None:
        run_cfg["batch_size"] = args.batch_size
    if args.debug is not None:
        run_cfg["debug_batches"] = args.debug
        log.info(f"DEBUG MODE: {args.debug} batches per epoch")

    train(cfg=run_cfg, resume_from=args.resume)
