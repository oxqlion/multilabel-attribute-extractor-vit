"""
FashionpediaAttrDataset — PyTorch Dataset for Attribute Extraction
Requires: torch, torchvision, Pillow, pandas, numpy
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FashionpediaAttrDataset(Dataset):
    """
    PyTorch Dataset for Fashionpedia attribute extraction.

    Returns (crop_tensor, label_tensor) pairs where:
      - crop_tensor : (3, 224, 224) float32 normalized crop
      - label_tensor: (n_classes,) float32 multi-hot vector
    """

    DEFAULT_MEAN = (0.485, 0.456, 0.406)   # ImageNet stats
    DEFAULT_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        processed_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
    ):
        assert split in ("train", "val")
        processed_dir = Path(processed_dir)

        self.df     = pd.read_csv(processed_dir / f"{split}.csv")
        self.labels = np.load(processed_dir / f"{split}_labels.npy")

        with open(processed_dir / "metadata" / "label_mappings.json") as f:
            self.mappings = json.load(f)

        assert len(self.df) == len(self.labels), "CSV/NPY row count mismatch"

        # ── Transforms ────────────────────────────────────────────────────
        if augment and split == "train":
            self.transform = T.Compose([
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(self.DEFAULT_MEAN, self.DEFAULT_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(self.DEFAULT_MEAN, self.DEFAULT_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = torch.from_numpy(self.labels[idx]).float()

        # Load and crop image
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
            # Clamp to image boundaries
            x2 = min(x + w, img.width)
            y2 = min(y + h, img.height)
            x  = max(x, 0)
            y  = max(y, 0)
            crop = img.crop((x, y, x2, y2))
        except (FileNotFoundError, OSError):
            # Graceful fallback: return black crop (log in production)
            crop = Image.new("RGB", (224, 224))

        crop_tensor = self.transform(crop)
        return crop_tensor, label

    @property
    def n_classes(self):
        return self.mappings["n_classes"]

    def get_pos_weight(self, device="cpu"):
        """
        Load pos_weights for BCEWithLogitsLoss.

        Usage:
            ds = FashionpediaAttrDataset(...)
            pos_weight = ds.get_pos_weight(device="cuda")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        """
        with open(Path(self.processed_dir) / "metadata" / "pos_weights.json") as f:
            pw = json.load(f)["pos_weight"]
        return torch.tensor(pw, dtype=torch.float32, device=device)