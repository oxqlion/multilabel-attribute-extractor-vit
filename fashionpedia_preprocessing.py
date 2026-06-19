#!/usr/bin/env python3
# =============================================================================
#  FASHIONPEDIA — Research-Quality Attribute Extraction Preprocessing Pipeline
#  For: ConvNeXt-Tiny + BCEWithLogitsLoss attribute prediction
#  Author: Generated for fashion attribute extraction research
#  Version: 1.0
# =============================================================================
#
#  DESIGN PHILOSOPHY
#  -----------------
#  This pipeline is NOT a generic preprocessing script.  Every decision is
#  grounded in the specific characteristics of the Fashionpedia dataset as
#  revealed by EDA.  Key departures from naive practice:
#
#  1. We DROP zero-attribute annotations (38.1% of data) rather than treating
#     them as all-negative samples, because Fashionpedia annotators skipped
#     many garment-part instances rather than labelling them as attribute-free.
#     Treating skipped instances as confirmed all-negatives would corrupt BCE.
#
#  2. We EXCLUDE the 'nickname' supercategory by default.  Nicknames are
#     garment-type identifiers (set-in sleeve, jeans, blazer), not visual
#     descriptors.  They overlap with category labels and dominate the long
#     tail.  A separate head or downstream model is more appropriate.
#
#  3. We use a PRINCIPLED rarity threshold (min_attr_samples, default 300)
#     derived from expected batch coverage rather than an arbitrary cutoff.
#
#  4. Training unit is ANNOTATION-LEVEL CROP (bbox), not full image, because
#     attributes describe individual garment parts.
#
#  5. We build a CATEGORY-ATTRIBUTE AFFINITY MAP from empirical co-occurrence
#     for optional category-conditioned label masking (ablation-ready).
#
# =============================================================================

# ── Notebook-style cells delimited by  # %%  ─────────────────────────────────
# Run sequentially in Jupyter or as a plain Python script.
# Each "cell" is clearly labelled.

# %% [markdown]
# # Fashionpedia Attribute Extraction — Data Preprocessing Pipeline
# ### Research-Quality Dataset Construction for ConvNeXt-Tiny + BCEWithLogitsLoss

# =============================================================================
# CELL 0 — IMPORTS & OPTIONAL DEPENDENCIES
# =============================================================================
# %%

import json
import os
import sys
import copy
import math
import time
import logging
import warnings
import collections
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# ── Optional heavy dependencies (graceful fallback) ──────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    # Lightweight fallback: plain iterator with progress print every N items
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else None)
            self.desc = desc
            self._n = 0
        def __iter__(self):
            for i, item in enumerate(self.iterable):
                if i % max(1, (self.total or 1) // 10) == 0:
                    pct = 100 * i / self.total if self.total else 0
                    print(f"  {self.desc}: {i}/{self.total} ({pct:.0f}%)", flush=True)
                yield item
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self._n += n

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    warnings.warn("pyarrow not available — Parquet export will be skipped.")

# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fashionpedia")

print("✓ Imports complete")
print(f"  pandas  {pd.__version__}")
print(f"  numpy   {np.__version__}")
print(f"  Parquet support: {HAS_PARQUET}")


# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================
# %%
# ---------------------------------------------------------------------------
# All tunable parameters live here.  Change these before running the pipeline.
# ---------------------------------------------------------------------------

CFG = dict(

    # ── Paths ──────────────────────────────────────────────────────────────
    # Point these to your local Fashionpedia directory.
    data_root            = Path("."),          # root of dataset
    train_json           = Path("./instances_attributes_train2020.json"),
    val_json             = Path("./instances_attributes_val2020.json"),
    train_image_dir      = Path("./train"),
    val_image_dir        = Path("./test"),     # val images in 'test' dir
    output_dir           = Path("./fashionpedia_processed"),

    # ── Attribute filtering ────────────────────────────────────────────────
    # Supercategories to EXCLUDE from the label space.
    # Rationale:
    #   'nickname'  → garment-type IDs, not visual descriptors; overlap with
    #                 category field; dominate the long tail (153 classes).
    #   'animal'    → max 210 samples — unlearnable under BCE.
    #   'leather'   → max 84 samples — unlearnable under BCE.
    exclude_supercats    = {"nickname", "animal", "leather"},

    # Minimum number of TRAINING set occurrences for an attribute to be
    # retained.  Derivation: with ~45K train images, batch_size=64, and
    # wanting ≥10 batches containing a positive, we need ~640 image-level
    # positives.  At annotation-level with mean 2.28 attrs/annotation, a
    # rough floor of 300 annotation-level occurrences is appropriate.
    min_attr_samples     = 300,

    # ── Annotation-level filters ───────────────────────────────────────────
    # Drop crowd annotations (iscrowd=1): attribute labels are ambiguous for
    # crowd instances.
    drop_crowd           = True,

    # Drop annotations with zero attribute labels.
    # CRITICAL: In Fashionpedia, zero-attribute annotations are predominantly
    # unannotated instances (skipped by annotators), NOT confirmed all-negative
    # instances.  Treating them as all-zero BCE targets would corrupt learning.
    drop_zero_attr       = True,

    # Minimum bounding box area (pixels²) for a crop to be considered valid.
    # ConvNeXt-Tiny operates at 224×224; a 32×32 bbox (1024 px²) crops to a
    # tiny patch with virtually no discriminative information.
    min_bbox_area        = 1024,  # 32 × 32

    # Minimum bounding box side length in pixels (applied to both w and h).
    min_bbox_side        = 16,

    # ── Advanced options ───────────────────────────────────────────────────
    # Build category–attribute co-occurrence affinity map.
    # Used for optional category-conditioned label masking (ablation flag).
    build_affinity_map   = True,

    # Affinity threshold: an attribute is considered 'applicable' to a
    # category if it appears in ≥ this fraction of annotated instances for
    # that category.
    affinity_threshold   = 0.01,  # 1% co-occurrence

    # Apply category-conditioned masking to labels.
    # If True, attributes that never (below affinity_threshold) co-occur with
    # a category are forced to 0 in that category's multi-hot vector.
    # Useful for ablation; leave False for baseline experiments.
    apply_category_mask  = False,

    # ── Reproducibility ────────────────────────────────────────────────────
    random_seed          = 42,
    pipeline_version     = "1.0",
)

# Make output dir
CFG["output_dir"].mkdir(parents=True, exist_ok=True)
(CFG["output_dir"] / "figures").mkdir(exist_ok=True)
(CFG["output_dir"] / "metadata").mkdir(exist_ok=True)

print("✓ Configuration loaded")
print(f"  Output directory : {CFG['output_dir'].resolve()}")
print(f"  Excluded supercats: {CFG['exclude_supercats']}")
print(f"  Min attr samples : {CFG['min_attr_samples']}")


# =============================================================================
# CELL 2 — STEP 1: LOAD DATASET
# =============================================================================
# %%

def load_fashionpedia_json(json_path: Path) -> Dict:
    """
    Load a Fashionpedia annotation JSON file.

    Returns the raw dict with keys:
        info, categories, attributes, images, annotations, licenses

    Raises FileNotFoundError if the path does not exist.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    log.info(f"Loading {json_path.name} …")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elapsed = time.time() - t0
    log.info(f"  Loaded in {elapsed:.1f}s | "
             f"images={len(data.get('images', []))}, "
             f"annotations={len(data.get('annotations', []))}, "
             f"categories={len(data.get('categories', []))}, "
             f"attributes={len(data.get('attributes', []))}")
    return data


def build_lookup_maps(data: Dict) -> Dict:
    """
    Build O(1) lookup dictionaries from a loaded Fashionpedia JSON dict.

    Returns a dict with:
        cat_by_id    : {cat_id -> category dict}
        attr_by_id   : {attr_id -> attribute dict}
        img_by_id    : {img_id -> image dict}
        annots_by_img: {img_id -> [annotation dicts]}
    """
    cat_by_id     = {c["id"]: c for c in data["categories"]}
    attr_by_id    = {a["id"]: a for a in data["attributes"]}
    img_by_id     = {i["id"]: i for i in data["images"]}
    annots_by_img = collections.defaultdict(list)
    for ann in data["annotations"]:
        annots_by_img[ann["image_id"]].append(ann)
    return dict(
        cat_by_id=cat_by_id,
        attr_by_id=attr_by_id,
        img_by_id=img_by_id,
        annots_by_img=annots_by_img,
    )


# ── Load train and validation splits ─────────────────────────────────────────
train_data = load_fashionpedia_json(CFG["train_json"])
val_data   = load_fashionpedia_json(CFG["val_json"])

train_maps = build_lookup_maps(train_data)
val_maps   = build_lookup_maps(val_data)

print("\n✓ Dataset loaded")
print(f"  Train: {len(train_data['images'])} images, "
      f"{len(train_data['annotations'])} annotations")
print(f"  Val  : {len(val_data['images'])} images, "
      f"{len(val_data['annotations'])} annotations")


# =============================================================================
# CELL 3 — STEP 2: DATASET INSPECTION & INTEGRITY VERIFICATION
# =============================================================================
# %%

def verify_annotation_integrity(data: Dict, maps: Dict, split_name: str) -> Dict:
    """
    Run a suite of integrity checks on a Fashionpedia split.

    Checks:
      - Every annotation references a valid image_id
      - Every annotation references a valid category_id
      - Every attribute_id in annotations references a valid attribute
      - Bounding box has 4 elements and positive dimensions
      - No duplicate annotation IDs

    Returns a dict summarising findings.
    """
    log.info(f"[{split_name}] Running integrity checks …")

    issues = collections.defaultdict(list)
    ann_ids_seen = set()

    for ann in tqdm(data["annotations"], desc=f"  {split_name} integrity"):
        aid = ann["id"]

        # Duplicate annotation IDs
        if aid in ann_ids_seen:
            issues["duplicate_ann_id"].append(aid)
        ann_ids_seen.add(aid)

        # Orphan annotation (no matching image)
        if ann["image_id"] not in maps["img_by_id"]:
            issues["orphan_image_id"].append(aid)

        # Unknown category
        if ann["category_id"] not in maps["cat_by_id"]:
            issues["unknown_category_id"].append(aid)

        # Unknown attribute IDs
        for attr_id in ann.get("attribute_ids", []):
            if attr_id not in maps["attr_by_id"]:
                issues["unknown_attr_id"].append((aid, attr_id))

        # Bounding box format
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            issues["malformed_bbox"].append(aid)
        elif bbox[2] <= 0 or bbox[3] <= 0:
            issues["degenerate_bbox"].append(aid)

    summary = {k: len(v) for k, v in issues.items()}
    total_issues = sum(summary.values())

    if total_issues == 0:
        log.info(f"  [{split_name}] ✓ No integrity issues found")
    else:
        log.warning(f"  [{split_name}] {total_issues} issues found:")
        for k, cnt in summary.items():
            log.warning(f"    {k}: {cnt}")

    return {"split": split_name, "issues": dict(issues), "summary": summary}


def verify_cross_split_consistency(
    train_data: Dict, val_data: Dict
) -> Dict:
    """
    Verify that category and attribute IDs are consistent across train/val.

    In dataset releases, ID remapping between splits is a known source of
    silent bugs.  We verify:
      - Same set of category IDs
      - Same category names for each ID
      - Same set of attribute IDs
      - Same attribute names for each ID
    """
    log.info("Verifying train/val cross-split consistency …")
    issues = {}

    train_cat_ids = {c["id"] for c in train_data["categories"]}
    val_cat_ids   = {c["id"] for c in val_data["categories"]}
    if train_cat_ids != val_cat_ids:
        issues["category_id_mismatch"] = {
            "only_in_train": train_cat_ids - val_cat_ids,
            "only_in_val":   val_cat_ids   - train_cat_ids,
        }

    # Check names match for shared IDs
    train_cat_map = {c["id"]: c["name"] for c in train_data["categories"]}
    val_cat_map   = {c["id"]: c["name"] for c in val_data["categories"]}
    name_mismatches = {}
    for cid in train_cat_ids & val_cat_ids:
        if train_cat_map[cid] != val_cat_map[cid]:
            name_mismatches[cid] = (train_cat_map[cid], val_cat_map[cid])
    if name_mismatches:
        issues["category_name_mismatch"] = name_mismatches

    train_attr_ids = {a["id"] for a in train_data["attributes"]}
    val_attr_ids   = {a["id"] for a in val_data["attributes"]}
    if train_attr_ids != val_attr_ids:
        issues["attribute_id_mismatch"] = {
            "only_in_train": train_attr_ids - val_attr_ids,
            "only_in_val":   val_attr_ids   - train_attr_ids,
        }

    train_attr_map = {a["id"]: a["name"] for a in train_data["attributes"]}
    val_attr_map   = {a["id"]: a["name"] for a in val_data["attributes"]}
    attr_name_mismatches = {}
    for aid in train_attr_ids & val_attr_ids:
        if train_attr_map[aid] != val_attr_map[aid]:
            attr_name_mismatches[aid] = (train_attr_map[aid], val_attr_map[aid])
    if attr_name_mismatches:
        issues["attribute_name_mismatch"] = attr_name_mismatches

    if not issues:
        log.info("  ✓ Train/val splits are fully consistent")
    else:
        log.warning(f"  Cross-split issues: {list(issues.keys())}")

    return issues


def generate_raw_summary_stats(data: Dict, split_name: str) -> pd.DataFrame:
    """
    Generate summary statistics on the raw (unfiltered) annotations.

    Returns a DataFrame with per-annotation statistics.
    """
    rows = []
    for ann in data["annotations"]:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        n_attrs = len(ann.get("attribute_ids", []))
        rows.append({
            "ann_id":       ann["id"],
            "image_id":     ann["image_id"],
            "category_id":  ann["category_id"],
            "n_attributes": n_attrs,
            "has_attrs":    n_attrs > 0,
            "iscrowd":      ann.get("iscrowd", 0),
            "bbox_x":       bbox[0] if len(bbox) == 4 else np.nan,
            "bbox_y":       bbox[1] if len(bbox) == 4 else np.nan,
            "bbox_w":       bbox[2] if len(bbox) == 4 else np.nan,
            "bbox_h":       bbox[3] if len(bbox) == 4 else np.nan,
            "bbox_area":    (bbox[2] * bbox[3]) if len(bbox) == 4 else np.nan,
        })
    df = pd.DataFrame(rows)
    log.info(f"[{split_name}] Raw annotation stats:")
    log.info(f"  Total annotations : {len(df):,}")
    log.info(f"  With attributes   : {df['has_attrs'].sum():,} "
             f"({df['has_attrs'].mean()*100:.1f}%)")
    log.info(f"  Zero-attribute    : {(~df['has_attrs']).sum():,} "
             f"({(~df['has_attrs']).mean()*100:.1f}%)")
    log.info(f"  Crowd annotations : {df['iscrowd'].sum():,}")
    log.info(f"  Mean attrs/ann    : {df['n_attributes'].mean():.2f}")
    log.info(f"  Median attrs/ann  : {df['n_attributes'].median():.0f}")
    log.info(f"  Max attrs/ann     : {df['n_attributes'].max()}")
    return df


# ── Run checks ────────────────────────────────────────────────────────────────
train_integrity = verify_annotation_integrity(train_data, train_maps, "train")
val_integrity   = verify_annotation_integrity(val_data,   val_maps,   "val")
cross_split     = verify_cross_split_consistency(train_data, val_data)

train_raw_df = generate_raw_summary_stats(train_data, "train")
val_raw_df   = generate_raw_summary_stats(val_data,   "val")

print("\n✓ Inspection complete")


# =============================================================================
# CELL 4 — STEP 3: DATA CLEANING FUNCTIONS
# =============================================================================
# %%
# Each cleaning operation is an independent, configurable function.
# They all accept a list of annotation dicts and return (filtered_list, report).

def drop_crowd_annotations(
    annotations: List[Dict],
    enabled: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Remove crowd annotations (iscrowd == 1).

    Rationale: Crowd instances have aggregated or missing attribute labels.
    Their bounding boxes cover multiple items, making crop-based attribute
    learning noisy.
    """
    if not enabled:
        return annotations, {"op": "drop_crowd", "skipped": True}

    before = len(annotations)
    filtered = [a for a in annotations if a.get("iscrowd", 0) == 0]
    after = len(filtered)
    report = {
        "op":      "drop_crowd",
        "before":  before,
        "after":   after,
        "removed": before - after,
        "pct":     (before - after) / max(before, 1) * 100,
    }
    log.info(f"[drop_crowd] Removed {report['removed']:,} annotations "
             f"({report['pct']:.1f}%)")
    return filtered, report


def drop_zero_attribute_annotations(
    annotations: List[Dict],
    enabled: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Remove annotations with no attribute labels (attribute_ids == []).

    CRITICAL RATIONALE:
    In Fashionpedia, 38.1% of annotations have zero attributes.  These are
    overwhelmingly UNANNOTATED instances (annotators skipped them), NOT
    confirmed attribute-free items.  Evidence: the top annotation categories
    are garment parts (sleeve, neckline, pocket) which MUST have attributes
    like length, opening type, etc.  If they were truly attribute-free, we
    would expect near-zero counts on 'set-in sleeve', 'wrist-length', etc.
    But those are among the most frequent attributes.

    Treating zero-attribute annotations as confirmed all-negative BCE targets
    would inject massive false-negative signal into training.

    Alternative considered: Assign them a 'no_attribute' pseudo-label.
    Rejected: Would require a special loss term and complicates evaluation.
    """
    if not enabled:
        return annotations, {"op": "drop_zero_attr", "skipped": True}

    before = len(annotations)
    filtered = [a for a in annotations if len(a.get("attribute_ids", [])) > 0]
    after = len(filtered)
    report = {
        "op":      "drop_zero_attr",
        "before":  before,
        "after":   after,
        "removed": before - after,
        "pct":     (before - after) / max(before, 1) * 100,
    }
    log.info(f"[drop_zero_attr] Removed {report['removed']:,} annotations "
             f"({report['pct']:.1f}%)")
    return filtered, report


def drop_small_bboxes(
    annotations: List[Dict],
    min_area: int = 1024,
    min_side: int = 16,
    enabled: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Remove annotations whose bounding box is too small to be informative.

    Rationale: ConvNeXt-Tiny processes 224×224 crops.  A bounding box of
    32×32 pixels (area=1024) or smaller produces a severely low-resolution
    patch after resizing — the attribute signal is dominated by interpolation
    artifacts rather than actual garment features.

    We apply both an area threshold and a minimum side length (to also catch
    degenerate thin-strip bboxes like 4×400).
    """
    if not enabled:
        return annotations, {"op": "drop_small_bbox", "skipped": True}

    before = len(annotations)
    def is_valid_bbox(ann):
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            return False
        w, h = bbox[2], bbox[3]
        return (w * h >= min_area) and (w >= min_side) and (h >= min_side)

    filtered = [a for a in annotations if is_valid_bbox(a)]
    after = len(filtered)
    report = {
        "op":        "drop_small_bbox",
        "before":    before,
        "after":     after,
        "removed":   before - after,
        "pct":       (before - after) / max(before, 1) * 100,
        "min_area":  min_area,
        "min_side":  min_side,
    }
    log.info(f"[drop_small_bbox] Removed {report['removed']:,} annotations "
             f"({report['pct']:.1f}%) | min_area={min_area}, min_side={min_side}")
    return filtered, report


def detect_duplicate_crops(
    annotations: List[Dict],
    images: List[Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Detect and remove duplicate annotation crops.

    A duplicate is defined as two annotations sharing the same (image_id,
    bbox) tuple — i.e., two annotations for the exact same crop.

    We keep the annotation with more attribute labels (richer labelling wins).
    This is conservative: if labels differ, the richer one is preferred.

    Note: Full image-level deduplication (perceptual hashing) is out of scope
    here as it requires loading image files.  We do hash-based bbox dedup.
    """
    before = len(annotations)
    seen: Dict[Tuple, Dict] = {}

    for ann in annotations:
        bbox = ann.get("bbox", [])
        key = (ann["image_id"],) + tuple(int(v) for v in bbox)
        if key not in seen:
            seen[key] = ann
        else:
            # Keep the one with more attributes
            if len(ann.get("attribute_ids", [])) > len(seen[key].get("attribute_ids", [])):
                seen[key] = ann

    filtered = list(seen.values())
    after = len(filtered)
    report = {
        "op":      "detect_duplicates",
        "before":  before,
        "after":   after,
        "removed": before - after,
        "pct":     (before - after) / max(before, 1) * 100,
    }
    log.info(f"[detect_duplicates] Removed {report['removed']:,} duplicate crops "
             f"({report['pct']:.1f}%)")
    return filtered, report


# =============================================================================
# CELL 5 — STEP 4: ATTRIBUTE PROCESSING
# =============================================================================
# %%

def build_attribute_metadata(
    attributes: List[Dict],
    exclude_supercats: Set[str],
) -> Tuple[List[Dict], Dict]:
    """
    Build the filtered attribute list and ID→metadata lookup.

    Attributes belonging to excluded supercategories are removed.
    Returns:
        candidate_attrs : list of attribute dicts that passed the supercat filter
        attr_meta       : {attr_id -> {name, supercategory, level}}
    """
    candidate_attrs = [
        a for a in attributes
        if a.get("supercategory", "") not in exclude_supercats
    ]
    attr_meta = {
        a["id"]: {
            "name":          a["name"],
            "supercategory": a.get("supercategory", ""),
            "level":         a.get("level", 0),
            "taxonomy_id":   a.get("taxonomy_id", ""),
        }
        for a in candidate_attrs
    }
    excluded_count = len(attributes) - len(candidate_attrs)
    log.info(f"[attr_metadata] Attributes after supercat filter: "
             f"{len(candidate_attrs)}/{len(attributes)} "
             f"(excluded {excluded_count} from {exclude_supercats})")
    return candidate_attrs, attr_meta


def compute_attribute_frequencies(
    annotations: List[Dict],
    candidate_attr_ids: Set[int],
) -> Dict[int, int]:
    """
    Count how many annotations each attribute appears in,
    restricted to the candidate attribute set.

    This operates on the already-cleaned annotation list (post crowd/zero drop).
    """
    freq = collections.Counter()
    for ann in annotations:
        for attr_id in ann.get("attribute_ids", []):
            if attr_id in candidate_attr_ids:
                freq[attr_id] += 1
    return dict(freq)


def filter_rare_attributes(
    attr_frequencies: Dict[int, int],
    attr_meta: Dict[int, Dict],
    min_samples: int,
) -> Tuple[Set[int], Dict]:
    """
    Remove attributes that appear fewer than min_samples times in training.

    Rationale: With BCE loss, extremely rare classes (< 300 occurrences in
    ~45K images) will appear in fewer than 10 mini-batches (batch_size=64).
    The gradient signal from such classes is negligible and their presence
    inflates the head size, increasing the BCE imbalance problem.

    We do NOT use class-reweighting or oversampling here because:
    - Reweighting extreme minorities distorts the loss for majority classes.
    - Oversampling rare classes in a multi-label setting is complex and can
      introduce label co-occurrence artifacts.

    The clean solution is to remove unlearnable classes and acknowledge this
    as a limitation in the paper.

    Returns:
        kept_attr_ids : set of retained attribute IDs
        report        : frequency DataFrame + summary
    """
    freq_df = pd.DataFrame([
        {
            "attr_id":   attr_id,
            "name":      attr_meta.get(attr_id, {}).get("name", "?"),
            "supercat":  attr_meta.get(attr_id, {}).get("supercategory", "?"),
            "count":     count,
            "retained":  count >= min_samples,
        }
        for attr_id, count in attr_frequencies.items()
    ]).sort_values("count", ascending=False)

    kept_attr_ids = set(freq_df[freq_df["retained"]]["attr_id"].tolist())

    dropped = freq_df[~freq_df["retained"]]
    log.info(f"[filter_rare] Threshold = {min_samples} samples")
    log.info(f"  Retained: {len(kept_attr_ids):,} attributes")
    log.info(f"  Dropped : {len(dropped):,} attributes")
    if len(dropped) > 0:
        log.info(f"  Dropped supercats: "
                 f"{dict(dropped['supercat'].value_counts().head(10))}")

    return kept_attr_ids, {"freq_df": freq_df, "min_samples": min_samples}


def build_category_attribute_affinity(
    annotations: List[Dict],
    kept_attr_ids: Set[int],
    cat_by_id: Dict,
    attr_meta: Dict,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Compute a category × attribute co-occurrence affinity matrix.

    Entry [cat, attr] = fraction of annotated instances of category 'cat'
    that carry attribute 'attr'.

    This reveals:
    1. Which attributes are semantically linked to which categories.
    2. Attributes that appear to be erroneously assigned (very low affinity
       for their expected category might indicate annotation noise).
    3. Can be used as a soft prior or hard mask in the model.

    The affinity matrix is saved to disk for use in evaluation and ablation.
    """
    log.info("Building category–attribute affinity map …")

    # Count per category: total annotated instances & per-attr positives
    cat_total    = collections.defaultdict(int)
    cat_attr_pos = collections.defaultdict(lambda: collections.defaultdict(int))

    for ann in annotations:
        cid = ann["category_id"]
        cat_total[cid] += 1
        for attr_id in ann.get("attribute_ids", []):
            if attr_id in kept_attr_ids:
                cat_attr_pos[cid][attr_id] += 1

    cat_ids  = sorted(cat_total.keys())
    attr_ids = sorted(kept_attr_ids)

    cat_names  = [cat_by_id.get(c, {}).get("name", str(c)) for c in cat_ids]
    attr_names = [attr_meta.get(a, {}).get("name", str(a))  for a in attr_ids]

    matrix = np.zeros((len(cat_ids), len(attr_ids)), dtype=np.float32)
    for i, cid in enumerate(cat_ids):
        total = cat_total[cid]
        if total == 0:
            continue
        for j, attr_id in enumerate(attr_ids):
            matrix[i, j] = cat_attr_pos[cid][attr_id] / total

    affinity_df = pd.DataFrame(
        matrix,
        index=pd.Index(cat_ids, name="category_id"),
        columns=pd.Index(attr_ids, name="attr_id"),
    )
    affinity_df.index.name  = "category_id"
    affinity_df.columns.name = "attr_id"

    log.info(f"  Affinity matrix shape: {affinity_df.shape}")
    log.info(f"  Non-zero entries (>{threshold}): "
             f"{(affinity_df > threshold).values.sum():,}")

    return affinity_df


def apply_supercat_exclude_to_annotation(
    ann: Dict,
    attr_meta: Dict,
    exclude_supercats: Set[str],
) -> Dict:
    """
    Strip excluded-supercat attribute IDs from a single annotation.
    Returns a modified copy.
    """
    ann = copy.copy(ann)
    ann["attribute_ids"] = [
        aid for aid in ann.get("attribute_ids", [])
        if attr_meta.get(aid, {}).get("supercategory", "") not in exclude_supercats
    ]
    return ann


def apply_rarity_filter_to_annotation(
    ann: Dict,
    kept_attr_ids: Set[int],
) -> Dict:
    """
    Strip rare (filtered-out) attribute IDs from a single annotation.
    Returns a modified copy.
    """
    ann = copy.copy(ann)
    ann["attribute_ids"] = [
        aid for aid in ann.get("attribute_ids", [])
        if aid in kept_attr_ids
    ]
    return ann


# =============================================================================
# CELL 6 — RUN THE CLEANING & ATTRIBUTE PROCESSING PIPELINE
# =============================================================================
# %%

print("=" * 70)
print("RUNNING PREPROCESSING PIPELINE")
print("=" * 70)

cleaning_reports = []

# ── Step 3a: Clean training annotations ──────────────────────────────────────
log.info("\n── Cleaning TRAIN annotations ──")
train_anns = train_data["annotations"]

train_anns, r = drop_crowd_annotations(train_anns, enabled=CFG["drop_crowd"])
cleaning_reports.append(("train", r))

train_anns, r = drop_zero_attribute_annotations(train_anns, enabled=CFG["drop_zero_attr"])
cleaning_reports.append(("train", r))

train_anns, r = drop_small_bboxes(
    train_anns,
    min_area=CFG["min_bbox_area"],
    min_side=CFG["min_bbox_side"],
    enabled=True,
)
cleaning_reports.append(("train", r))

train_anns, r = detect_duplicate_crops(train_anns, train_data["images"])
cleaning_reports.append(("train", r))

# ── Step 3b: Clean validation annotations ────────────────────────────────────
log.info("\n── Cleaning VAL annotations ──")
val_anns = val_data["annotations"]

val_anns, r = drop_crowd_annotations(val_anns, enabled=CFG["drop_crowd"])
cleaning_reports.append(("val", r))

val_anns, r = drop_zero_attribute_annotations(val_anns, enabled=CFG["drop_zero_attr"])
cleaning_reports.append(("val", r))

val_anns, r = drop_small_bboxes(
    val_anns,
    min_area=CFG["min_bbox_area"],
    min_side=CFG["min_bbox_side"],
    enabled=True,
)
cleaning_reports.append(("val", r))

val_anns, r = detect_duplicate_crops(val_anns, val_data["images"])
cleaning_reports.append(("val", r))

# ── Step 4a: Build attribute metadata (supercat filter) ───────────────────────
log.info("\n── Processing attributes ──")
candidate_attrs, attr_meta_full = build_attribute_metadata(
    train_data["attributes"],
    exclude_supercats=CFG["exclude_supercats"],
)
candidate_attr_ids = set(a["id"] for a in candidate_attrs)

# ── Step 4b: Strip excluded supercats from annotation attribute_ids ───────────
train_anns = [apply_supercat_exclude_to_annotation(a, attr_meta_full, CFG["exclude_supercats"]) for a in train_anns]
val_anns   = [apply_supercat_exclude_to_annotation(a, attr_meta_full, CFG["exclude_supercats"]) for a in val_anns]

# ── Step 4c: Drop annotations that became zero-attr after supercat stripping ──
# (A 'nickname'-only annotation becomes empty after excluding nicknames)
before_recheck = len(train_anns)
train_anns = [a for a in train_anns if len(a.get("attribute_ids", [])) > 0]
log.info(f"  Post-supercat-strip empty drop (train): "
         f"{before_recheck - len(train_anns):,} removed")

before_recheck = len(val_anns)
val_anns = [a for a in val_anns if len(a.get("attribute_ids", [])) > 0]
log.info(f"  Post-supercat-strip empty drop (val): "
         f"{before_recheck - len(val_anns):,} removed")

# ── Step 4d: Compute attribute frequencies on cleaned TRAINING set ────────────
train_attr_freq = compute_attribute_frequencies(train_anns, candidate_attr_ids)

# ── Step 4e: Rare attribute filter (based on TRAIN frequencies only!) ─────────
# IMPORTANT: We derive the threshold ONLY from training data to avoid
# test-set leakage into the label-space design decision.
kept_attr_ids, rarity_report = filter_rare_attributes(
    train_attr_freq,
    attr_meta_full,
    min_samples=CFG["min_attr_samples"],
)

# ── Step 4f: Apply rarity filter to all annotation attribute_ids ───────────────
train_anns = [apply_rarity_filter_to_annotation(a, kept_attr_ids) for a in train_anns]
val_anns   = [apply_rarity_filter_to_annotation(a, kept_attr_ids) for a in val_anns]

# Drop annotations that became empty after rarity filter
before_recheck = len(train_anns)
train_anns = [a for a in train_anns if len(a.get("attribute_ids", [])) > 0]
log.info(f"  Post-rarity-filter empty drop (train): "
         f"{before_recheck - len(train_anns):,} removed")

before_recheck = len(val_anns)
val_anns = [a for a in val_anns if len(a.get("attribute_ids", [])) > 0]
log.info(f"  Post-rarity-filter empty drop (val): "
         f"{before_recheck - len(val_anns):,} removed")

# ── Step 4g: Build affinity map (from cleaned train, retained attrs) ──────────
if CFG["build_affinity_map"]:
    affinity_df = build_category_attribute_affinity(
        train_anns,
        kept_attr_ids,
        train_maps["cat_by_id"],
        attr_meta_full,
        threshold=CFG["affinity_threshold"],
    )

print(f"\n✓ Pipeline complete")
print(f"  Train annotations: {len(train_anns):,} (retained)")
print(f"  Val   annotations: {len(val_anns):,}   (retained)")
print(f"  Attribute label space: {len(kept_attr_ids)} attributes")


# =============================================================================
# CELL 7 — STEP 5: DATASET CONSTRUCTION
# =============================================================================
# %%

def build_dataset_dataframe(
    annotations: List[Dict],
    img_by_id: Dict,
    cat_by_id: Dict,
    attr_meta: Dict,
    kept_attr_ids: Set[int],
    image_dir: Path,
    split: str,
) -> pd.DataFrame:
    """
    Build a clean, flat DataFrame representing the processed dataset.

    Each row = one annotation crop (the training unit for attribute learning).

    Columns:
        ann_id          : original annotation ID (for traceability)
        image_id        : original image ID
        split           : 'train' or 'val'
        file_name       : original image filename
        image_path      : full path to image (relative to image_dir)
        image_w         : image width
        image_h         : image height
        bbox_x, bbox_y  : top-left corner of bbox
        bbox_w, bbox_h  : width, height of bbox
        bbox_area       : width × height
        category_id     : garment part category ID
        category        : garment part category name
        supercategory   : garment part supercategory
        attribute_ids   : list of retained attribute IDs
        attribute_names : list of corresponding attribute names
        n_attributes    : number of retained attributes
        neg_attr_flags  : boolean list — which attrs are 'negation' markers
    """
    # Negation attribute names (empirically identified from EDA)
    # These are valid labels but semantically inverted — flagging helps
    # researchers decide whether to exclude them in ablation studies.
    NEGATION_MARKER_NAMES = {
        "no non-textile material",
        "no special manufacturing technique",
        "no waistline",
        "no opening",
    }

    rows = []
    for ann in tqdm(annotations, desc=f"  Building {split} df"):
        img  = img_by_id.get(ann["image_id"], {})
        cat  = cat_by_id.get(ann["category_id"], {})
        bbox = ann.get("bbox", [0, 0, 0, 0])

        # Retained attribute IDs for this annotation
        ret_attr_ids = [aid for aid in ann.get("attribute_ids", [])
                        if aid in kept_attr_ids]

        if not ret_attr_ids:
            # Should not happen after prior filtering, but guard anyway
            continue

        attr_names    = [attr_meta.get(aid, {}).get("name", "?") for aid in ret_attr_ids]
        neg_flags     = [n in NEGATION_MARKER_NAMES for n in attr_names]
        has_any_neg   = any(neg_flags)

        rows.append({
            "ann_id":          ann["id"],
            "image_id":        ann["image_id"],
            "split":           split,
            "file_name":       img.get("file_name", ""),
            "image_path":      str(image_dir / img.get("file_name", "")),
            "image_w":         img.get("width", 0),
            "image_h":         img.get("height", 0),
            "bbox_x":          int(bbox[0]),
            "bbox_y":          int(bbox[1]),
            "bbox_w":          int(bbox[2]),
            "bbox_h":          int(bbox[3]),
            "bbox_area":       int(bbox[2]) * int(bbox[3]),
            "category_id":     ann["category_id"],
            "category":        cat.get("name", ""),
            "supercategory":   cat.get("supercategory", ""),
            "attribute_ids":   ret_attr_ids,
            "attribute_names": attr_names,
            "n_attributes":    len(ret_attr_ids),
            "has_negation_attr": has_any_neg,
        })

    df = pd.DataFrame(rows)
    log.info(f"[{split}] Dataset DataFrame: {len(df):,} rows × {len(df.columns)} cols")
    return df


# Build dataframes
train_df = build_dataset_dataframe(
    train_anns,
    train_maps["img_by_id"],
    train_maps["cat_by_id"],
    attr_meta_full,
    kept_attr_ids,
    CFG["train_image_dir"],
    split="train",
)

val_df = build_dataset_dataframe(
    val_anns,
    val_maps["img_by_id"],
    val_maps["cat_by_id"],
    attr_meta_full,
    kept_attr_ids,
    CFG["val_image_dir"],
    split="val",
)

print(f"\n✓ DataFrames constructed")
print(train_df.dtypes)
print(train_df.head(3).to_string())


# =============================================================================
# CELL 8 — STEP 6: MULTI-LABEL ENCODING
# =============================================================================
# %%

def build_label_mappings(kept_attr_ids: Set[int], attr_meta: Dict) -> Dict:
    """
    Build deterministic, sorted label index mappings.

    Returns:
        attr_id_to_idx   : {attr_id -> column index in multi-hot vector}
        idx_to_attr_id   : {column index -> attr_id}
        idx_to_attr_name : {column index -> attribute name}
        attr_id_to_name  : {attr_id -> attribute name}
        supercat_to_idxs : {supercategory -> [column indices]}
        n_classes        : total number of classes
    """
    # Sort by ID for determinism across runs
    sorted_attr_ids = sorted(kept_attr_ids)

    attr_id_to_idx   = {aid: i for i, aid in enumerate(sorted_attr_ids)}
    idx_to_attr_id   = {i: aid for i, aid in enumerate(sorted_attr_ids)}
    idx_to_attr_name = {
        i: attr_meta.get(aid, {}).get("name", f"attr_{aid}")
        for i, aid in enumerate(sorted_attr_ids)
    }
    attr_id_to_name  = {aid: attr_meta.get(aid, {}).get("name", f"attr_{aid}")
                        for aid in sorted_attr_ids}

    # Group indices by supercategory (useful for per-group mAP evaluation)
    supercat_to_idxs = collections.defaultdict(list)
    for i, aid in enumerate(sorted_attr_ids):
        sc = attr_meta.get(aid, {}).get("supercategory", "unknown")
        supercat_to_idxs[sc].append(i)

    mappings = {
        "attr_id_to_idx":   attr_id_to_idx,
        "idx_to_attr_id":   idx_to_attr_id,
        "idx_to_attr_name": idx_to_attr_name,
        "attr_id_to_name":  attr_id_to_name,
        "supercat_to_idxs": dict(supercat_to_idxs),
        "n_classes":        len(sorted_attr_ids),
        "sorted_attr_ids":  sorted_attr_ids,
    }
    log.info(f"Label mapping: {len(sorted_attr_ids)} classes")
    log.info(f"  Supercategories in label space: "
             f"{list(supercat_to_idxs.keys())}")
    return mappings


def encode_multi_hot(
    df: pd.DataFrame,
    mappings: Dict,
    split: str,
) -> np.ndarray:
    """
    Encode attribute_ids column into a multi-hot numpy array.

    Shape: (N_samples, n_classes)
    dtype: float32 (required by BCEWithLogitsLoss)

    Also adds a 'label_vector_idx' column to df referencing rows in the array.
    """
    n = len(df)
    n_classes = mappings["n_classes"]
    attr_id_to_idx = mappings["attr_id_to_idx"]

    multi_hot = np.zeros((n, n_classes), dtype=np.float32)

    for row_idx, attr_ids in enumerate(df["attribute_ids"]):
        for aid in attr_ids:
            col_idx = attr_id_to_idx.get(aid)
            if col_idx is not None:
                multi_hot[row_idx, col_idx] = 1.0

    # Sanity check: every row should have ≥ 1 positive
    empty_rows = (multi_hot.sum(axis=1) == 0).sum()
    if empty_rows > 0:
        log.warning(f"[{split}] {empty_rows} rows have all-zero multi-hot vectors!")
    else:
        log.info(f"[{split}] Multi-hot matrix: {multi_hot.shape}, "
                 f"no all-zero rows (✓)")

    pos_per_sample = multi_hot.sum(axis=1)
    log.info(f"  Positives per sample: mean={pos_per_sample.mean():.2f}, "
             f"median={np.median(pos_per_sample):.0f}, "
             f"max={pos_per_sample.max():.0f}")

    return multi_hot


# ── Build label mappings ──────────────────────────────────────────────────────
label_mappings = build_label_mappings(kept_attr_ids, attr_meta_full)

# ── Encode multi-hot matrices ─────────────────────────────────────────────────
train_labels = encode_multi_hot(train_df, label_mappings, "train")
val_labels   = encode_multi_hot(val_df,   label_mappings, "val")

# ── Compute per-class positive frequencies (for class-weight computation) ─────
class_pos_counts   = train_labels.sum(axis=0)                      # shape (n_classes,)
class_neg_counts   = len(train_labels) - class_pos_counts
pos_weight         = class_neg_counts / np.maximum(class_pos_counts, 1)  # for BCEWithLogitsLoss pos_weight

log.info(f"\nClass balance stats (train, {label_mappings['n_classes']} classes):")
log.info(f"  Median pos_weight : {np.median(pos_weight):.1f}")
log.info(f"  Max pos_weight    : {np.max(pos_weight):.1f}  "
         f"(attr: {label_mappings['idx_to_attr_name'][int(np.argmax(pos_weight))]})")
log.info(f"  Min pos_weight    : {np.min(pos_weight):.2f}  "
         f"(attr: {label_mappings['idx_to_attr_name'][int(np.argmin(pos_weight))]})")

print(f"\n✓ Multi-label encoding complete")
print(f"  Train labels shape: {train_labels.shape}")
print(f"  Val   labels shape: {val_labels.shape}")


# =============================================================================
# CELL 9 — STEP 7: QUALITY ASSURANCE
# =============================================================================
# %%

def qa_check_missing_images(
    df: pd.DataFrame,
    split: str,
) -> Dict:
    """
    Check how many image paths in the dataset actually exist on disk.

    Note: This check is optional and only meaningful when the images are
    locally present.  Reports rather than halts if images are missing.
    """
    paths = df["image_path"].unique()
    missing = [p for p in paths if not Path(p).exists()]
    present = len(paths) - len(missing)
    report = {
        "split":         split,
        "total_images":  len(paths),
        "present":       present,
        "missing":       len(missing),
        "missing_paths": missing[:20],   # Show first 20
    }
    if missing:
        log.warning(f"[{split}] {len(missing)}/{len(paths)} images NOT found on disk")
        log.warning(f"  First missing: {missing[0]}")
    else:
        log.info(f"[{split}] All {len(paths)} images found on disk ✓")
    return report


def qa_check_label_consistency(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_mappings: Dict,
) -> Dict:
    """
    Verify that val labels are a subset of the train label space.

    Any attribute seen in val but not in train's label space is an OOV (out
    of vocabulary) class that the model cannot predict.  These should be zero
    in the multi-hot encoding (they are, by construction), but we flag them
    explicitly for the research report.
    """
    train_attrs_seen = set()
    for aids in train_df["attribute_ids"]:
        train_attrs_seen.update(aids)

    val_attrs_seen = set()
    for aids in val_df["attribute_ids"]:
        val_attrs_seen.update(aids)

    oov = val_attrs_seen - train_attrs_seen
    report = {
        "train_unique_attrs": len(train_attrs_seen),
        "val_unique_attrs":   len(val_attrs_seen),
        "oov_in_val":        len(oov),
        "oov_attr_names":    [
            attr_meta_full.get(aid, {}).get("name", str(aid)) for aid in oov
        ],
    }
    if oov:
        log.warning(f"[QA] {len(oov)} val attributes not seen in train: "
                    f"{report['oov_attr_names'][:5]} …")
    else:
        log.info("[QA] All val attributes are in train label space ✓")
    return report


def qa_check_empty_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    split: str,
) -> Dict:
    """
    Verify there are no all-zero multi-hot vectors.
    Each training example must have at least one positive class.
    """
    all_zero = (labels.sum(axis=1) == 0).sum()
    report = {"split": split, "all_zero_vectors": int(all_zero)}
    if all_zero:
        log.error(f"[QA][{split}] {all_zero} samples have all-zero label vectors!")
    else:
        log.info(f"[QA][{split}] No all-zero label vectors ✓")
    return report


def qa_check_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict:
    """
    Check for image-level data leakage between train and val splits.

    Fashionpedia is officially split at the image level, but we verify
    this holds in our processed dataset.

    We check both image_id overlap and file_name overlap.
    """
    train_img_ids = set(train_df["image_id"].unique())
    val_img_ids   = set(val_df["image_id"].unique())
    leak_ids      = train_img_ids & val_img_ids

    train_fnames  = set(train_df["file_name"].unique())
    val_fnames    = set(val_df["file_name"].unique())
    leak_fnames   = train_fnames & val_fnames

    report = {
        "image_id_overlap":   len(leak_ids),
        "filename_overlap":   len(leak_fnames),
        "leaking_image_ids":  list(leak_ids)[:10],
        "leaking_filenames":  list(leak_fnames)[:10],
    }
    if leak_ids or leak_fnames:
        log.error(f"[QA] DATA LEAKAGE DETECTED! "
                  f"image_id overlap={len(leak_ids)}, "
                  f"filename overlap={len(leak_fnames)}")
    else:
        log.info("[QA] No train/val data leakage detected ✓")
    return report


def qa_check_duplicate_annotations(
    df: pd.DataFrame,
    split: str,
) -> Dict:
    """
    Check for duplicate (image_id, bbox) pairs in the processed dataframe.
    Should be zero after the deduplication step.
    """
    key_cols = ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    dupes = df.duplicated(subset=key_cols, keep=False).sum()
    report = {"split": split, "duplicate_crops": int(dupes)}
    if dupes:
        log.warning(f"[QA][{split}] {dupes} duplicate crop entries remain")
    else:
        log.info(f"[QA][{split}] No duplicate crops ✓")
    return report


# ── Run all QA checks ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("QUALITY ASSURANCE CHECKS")
print("=" * 70)

qa_results = {}
qa_results["missing_images_train"] = qa_check_missing_images(train_df, "train")
qa_results["missing_images_val"]   = qa_check_missing_images(val_df,   "val")
qa_results["label_consistency"]    = qa_check_label_consistency(train_df, val_df, label_mappings)
qa_results["empty_labels_train"]   = qa_check_empty_labels(train_df, train_labels, "train")
qa_results["empty_labels_val"]     = qa_check_empty_labels(val_df,   val_labels,   "val")
qa_results["data_leakage"]         = qa_check_data_leakage(train_df, val_df)
qa_results["duplicates_train"]     = qa_check_duplicate_annotations(train_df, "train")
qa_results["duplicates_val"]       = qa_check_duplicate_annotations(val_df,   "val")

print("\n✓ QA complete")


# =============================================================================
# CELL 10 — STEP 8: EXPORT
# =============================================================================
# %%

def save_label_mappings(mappings: Dict, output_dir: Path) -> None:
    """
    Persist label mappings to JSON for inference-time use.

    All keys converted to strings for JSON compatibility.
    """
    out = {
        "n_classes":         mappings["n_classes"],
        "sorted_attr_ids":   mappings["sorted_attr_ids"],
        "attr_id_to_idx":    {str(k): v for k, v in mappings["attr_id_to_idx"].items()},
        "idx_to_attr_id":    {str(k): v for k, v in mappings["idx_to_attr_id"].items()},
        "idx_to_attr_name":  {str(k): v for k, v in mappings["idx_to_attr_name"].items()},
        "attr_id_to_name":   {str(k): v for k, v in mappings["attr_id_to_name"].items()},
        "supercat_to_idxs":  {k: v for k, v in mappings["supercat_to_idxs"].items()},
        "pipeline_version":  CFG["pipeline_version"],
        "created_at":        datetime.utcnow().isoformat(),
        "config": {
            "exclude_supercats":  list(CFG["exclude_supercats"]),
            "min_attr_samples":   CFG["min_attr_samples"],
            "drop_crowd":         CFG["drop_crowd"],
            "drop_zero_attr":     CFG["drop_zero_attr"],
            "min_bbox_area":      CFG["min_bbox_area"],
            "min_bbox_side":      CFG["min_bbox_side"],
        },
    }
    path = output_dir / "metadata" / "label_mappings.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved label mappings → {path}")


def save_pos_weights(
    pos_weight: np.ndarray,
    mappings: Dict,
    output_dir: Path,
) -> None:
    """
    Save per-class positive weights for BCEWithLogitsLoss.

    File: pos_weights.json — array of floats in class-index order.
    These can be loaded directly as torch.Tensor for training.
    """
    pw_list = pos_weight.tolist()
    out = {
        "pos_weight": pw_list,
        "n_classes":  len(pw_list),
        "description": (
            "Per-class positive weights for torch.nn.BCEWithLogitsLoss(pos_weight=...)."
            " pw[i] = (n_negative_i / n_positive_i).  Higher weight = rarer class."
        ),
        "class_names": [mappings["idx_to_attr_name"][i] for i in range(len(pw_list))],
    }
    path = output_dir / "metadata" / "pos_weights.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved pos_weights → {path}")


def dataframe_to_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list-type columns to JSON strings for CSV/Parquet export.
    """
    df = df.copy()
    for col in ["attribute_ids", "attribute_names"]:
        if col in df.columns:
            df[col] = df[col].apply(json.dumps)
    return df


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path,
    has_parquet: bool,
) -> None:
    """
    Export datasets in multiple formats.

    CSV  — human-readable; list columns stored as JSON strings
    JSON — full fidelity; attribute_ids/names as real arrays
    NPY  — multi-hot label matrices for fast loading in PyTorch DataLoader
    Parquet — if available: best for large-scale training pipelines
    """
    # ── CSV ──────────────────────────────────────────────────────────────────
    train_ser = dataframe_to_serializable(train_df)
    val_ser   = dataframe_to_serializable(val_df)
    train_ser.to_csv(output_dir / "train.csv", index=False)
    val_ser.to_csv(output_dir  / "val.csv",   index=False)
    log.info(f"Saved train.csv ({len(train_df):,} rows)")
    log.info(f"Saved val.csv   ({len(val_df):,} rows)")

    # ── JSON ─────────────────────────────────────────────────────────────────
    train_records = train_df.to_dict(orient="records")
    val_records   = val_df.to_dict(orient="records")
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_records, f, default=str)
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_records, f, default=str)
    log.info("Saved train.json, val.json")

    # ── NumPy label matrices ──────────────────────────────────────────────────
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "val_labels.npy",   val_labels)
    log.info(f"Saved train_labels.npy {train_labels.shape}")
    log.info(f"Saved val_labels.npy   {val_labels.shape}")

    # ── Parquet (optional) ────────────────────────────────────────────────────
    if has_parquet:
        train_ser.to_parquet(output_dir / "train.parquet", index=False)
        val_ser.to_parquet(output_dir   / "val.parquet",   index=False)
        log.info("Saved train.parquet, val.parquet")
    else:
        log.info("Parquet skipped (pyarrow not available)")


# ── Save affinity map ─────────────────────────────────────────────────────────
if CFG["build_affinity_map"]:
    affinity_df.to_csv(CFG["output_dir"] / "metadata" / "category_attr_affinity.csv")
    log.info("Saved category_attr_affinity.csv")

# ── Save label mappings and pos_weights ──────────────────────────────────────
save_label_mappings(label_mappings, CFG["output_dir"])
save_pos_weights(pos_weight, label_mappings, CFG["output_dir"])

# ── Save dataframes ───────────────────────────────────────────────────────────
save_datasets(
    train_df, val_df,
    train_labels, val_labels,
    CFG["output_dir"],
    HAS_PARQUET,
)

# ── Save attribute frequency report ──────────────────────────────────────────
rarity_report["freq_df"].to_csv(
    CFG["output_dir"] / "metadata" / "attribute_frequencies.csv",
    index=False,
)
log.info("Saved attribute_frequencies.csv")

print("\n✓ Export complete")


# =============================================================================
# CELL 11 — STEP 9: VISUALISATIONS & RESEARCH REPORT
# =============================================================================
# %%

FIG_DIR = CFG["output_dir"] / "figures"

def savefig(name: str) -> None:
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()
    log.info(f"Saved figure: {name}")


# ── Figure 1: Attribute distribution before and after filtering ───────────────
def plot_attribute_distribution(rarity_report: Dict) -> None:
    freq_df = rarity_report["freq_df"].copy()
    freq_df = freq_df.sort_values("count", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: full distribution (log scale)
    ax = axes[0]
    colors = ["#2ecc71" if r else "#e74c3c" for r in freq_df["retained"]]
    ax.bar(range(len(freq_df)), freq_df["count"], color=colors, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Attribute rank (sorted by frequency)")
    ax.set_ylabel("Annotation count (log scale)")
    ax.set_title("Attribute Frequency Distribution (Log Scale)\n"
                 f"Green=retained, Red=dropped (< {rarity_report['min_samples']} samples)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ecc71", label=f"Retained ({freq_df['retained'].sum()})"),
        Patch(color="#e74c3c", label=f"Dropped  ({(~freq_df['retained']).sum()})"),
    ])

    # Right: per-supercategory stacked bar
    ax = axes[1]
    sc_counts = freq_df.groupby(["supercat", "retained"])["attr_id"].count().unstack(fill_value=0)
    if True not in sc_counts.columns:
        sc_counts[True] = 0
    if False not in sc_counts.columns:
        sc_counts[False] = 0
    sc_counts = sc_counts.sort_values(True, ascending=True)
    sc_counts[[True, False]].plot(
        kind="barh", stacked=True, ax=ax,
        color=["#2ecc71", "#e74c3c"],
        legend=False,
    )
    ax.set_xlabel("Number of attributes")
    ax.set_title("Retained vs. Dropped per Supercategory")
    ax.legend(["Retained", "Dropped"], loc="lower right")

    plt.tight_layout()
    savefig("fig1_attribute_distribution.png")


# ── Figure 2: Category distribution in processed dataset ──────────────────────
def plot_category_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_cats = train_df["category"].value_counts().head(20)
    val_cats   = val_df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(train_cats))
    width = 0.4
    ax.bar(x - width/2, train_cats.values, width, label="Train", color="#3498db", alpha=0.85)
    ax.bar(x + width/2,
           [val_cats.get(c, 0) for c in train_cats.index],
           width, label="Val", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(train_cats.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Annotation count")
    ax.set_title("Top-20 Categories: Train vs. Val (after preprocessing)")
    ax.legend()
    plt.tight_layout()
    savefig("fig2_category_distribution.png")


# ── Figure 3: Attributes per annotation (after preprocessing) ─────────────────
def plot_attrs_per_annotation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, df, name in zip(axes, [train_df, val_df], ["Train", "Val"]):
        vals = df["n_attributes"].clip(upper=15)
        ax.hist(vals, bins=range(1, 17), color="#9b59b6", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Number of retained attributes per annotation")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: Attributes per Annotation\n"
                     f"(mean={df['n_attributes'].mean():.2f}, "
                     f"median={df['n_attributes'].median():.0f})")
    plt.tight_layout()
    savefig("fig3_attrs_per_annotation.png")


# ── Figure 4: Class imbalance (pos_weight distribution) ───────────────────────
def plot_class_imbalance(pos_weight: np.ndarray, label_mappings: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(pos_weight, bins=50, color="#1abc9c", edgecolor="white", alpha=0.9)
    ax.set_xlabel("pos_weight value  (neg/pos ratio)")
    ax.set_ylabel("Number of attributes")
    ax.set_title("Class Imbalance Distribution\n"
                 "(BCEWithLogitsLoss pos_weight)")
    ax.axvline(np.median(pos_weight), color="red", ls="--",
               label=f"Median={np.median(pos_weight):.0f}")
    ax.legend()

    # Top-20 most imbalanced
    ax = axes[1]
    top20_idx  = np.argsort(pos_weight)[-20:][::-1]
    top20_pw   = pos_weight[top20_idx]
    top20_names = [label_mappings["idx_to_attr_name"][i] for i in top20_idx]
    ax.barh(range(20), top20_pw, color="#e74c3c", alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20_names, fontsize=8)
    ax.set_xlabel("pos_weight")
    ax.set_title("Top-20 Most Imbalanced Attributes")
    plt.tight_layout()
    savefig("fig4_class_imbalance.png")


# ── Figure 5: Affinity heatmap (top categories × supercategory avg) ───────────
def plot_affinity_heatmap(affinity_df: pd.DataFrame, mappings: Dict,
                          attr_meta: Dict, cat_by_id: Dict) -> None:
    """
    Plot a compact affinity heatmap averaged per attribute supercategory.
    Full heatmap is too large to display; this gives an interpretable view.
    """
    # Average affinity per supercategory
    supercat_to_idxs = mappings["supercat_to_idxs"]
    cat_ids   = affinity_df.index.tolist()
    cat_names = [cat_by_id.get(cid, {}).get("name", str(cid)) for cid in cat_ids]

    sc_names   = sorted(supercat_to_idxs.keys())
    avg_matrix = np.zeros((len(cat_ids), len(sc_names)), dtype=np.float32)

    attr_cols = affinity_df.columns.tolist()
    attr_id_to_col = {aid: i for i, aid in enumerate(attr_cols)}

    for j, sc in enumerate(sc_names):
        col_idxs = [attr_id_to_col[aid] for aid in supercat_to_idxs[sc]
                    if aid in attr_id_to_col]
        if col_idxs:
            avg_matrix[:, j] = affinity_df.values[:, col_idxs].mean(axis=1)

    aff_summary = pd.DataFrame(
        avg_matrix,
        index=cat_names,
        columns=sc_names,
    )

    # Select top-25 categories by total annotation count
    top_cats = [cat_by_id.get(cid, {}).get("name", str(cid))
                for cid in sorted(cat_by_id.keys())]
    top_cats = [c for c in top_cats if c in aff_summary.index][:25]
    aff_top  = aff_summary.loc[top_cats]

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        aff_top,
        annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Mean affinity"},
        annot_kws={"size": 7},
    )
    ax.set_title("Category–Attribute Supercategory Affinity\n"
                 "(fraction of category instances with each supercategory's attributes)")
    ax.set_xlabel("Attribute Supercategory")
    ax.set_ylabel("Garment Category")
    plt.tight_layout()
    savefig("fig5_affinity_heatmap.png")


# ── Figure 6: Supercategory label space breakdown ─────────────────────────────
def plot_supercat_breakdown(rarity_report: Dict) -> None:
    freq_df = rarity_report["freq_df"]
    retained = freq_df[freq_df["retained"]]

    sc_counts = retained.groupby("supercat")["attr_id"].count().sort_values(ascending=True)
    sc_total  = retained.groupby("supercat")["count"].sum().reindex(sc_counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(sc_counts.index, sc_counts.values, color="#3498db", alpha=0.85)
    axes[0].set_xlabel("Number of retained attributes")
    axes[0].set_title("Attributes per Supercategory\n(retained label space)")

    axes[1].barh(sc_total.index, sc_total.values, color="#e67e22", alpha=0.85)
    axes[1].set_xlabel("Total annotation count")
    axes[1].set_title("Total Annotation Volume per Supercategory\n(retained label space)")

    plt.tight_layout()
    savefig("fig6_supercat_breakdown.png")


# ── Generate all figures ──────────────────────────────────────────────────────
print("\nGenerating figures …")
plot_attribute_distribution(rarity_report)
plot_category_distribution(train_df, val_df)
plot_attrs_per_annotation(train_df, val_df)
plot_class_imbalance(pos_weight, label_mappings)
if CFG["build_affinity_map"]:
    plot_affinity_heatmap(
        affinity_df, label_mappings,
        attr_meta_full, train_maps["cat_by_id"]
    )
plot_supercat_breakdown(rarity_report)
print("✓ All figures saved")


# =============================================================================
# CELL 12 — PREPROCESSING REPORT (RESEARCH-QUALITY)
# =============================================================================
# %%

def generate_preprocessing_report(
    train_raw_df: pd.DataFrame,
    val_raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rarity_report: Dict,
    label_mappings: Dict,
    pos_weight: np.ndarray,
    cleaning_reports: List[Tuple[str, Dict]],
    qa_results: Dict,
    cfg: Dict,
    output_dir: Path,
) -> str:
    """
    Generate a full preprocessing report as a Markdown string and save to disk.

    Suitable for inclusion in a research paper's data section or appendix.
    """
    freq_df    = rarity_report["freq_df"]
    retained   = freq_df[freq_df["retained"]]
    dropped    = freq_df[~freq_df["retained"]]

    lines = []
    A = lines.append  # shorthand

    A("# Fashionpedia Attribute Extraction — Preprocessing Report")
    A(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    A(f"**Pipeline version:** {cfg['pipeline_version']}")
    A("")
    A("---")
    A("")

    # ── Config ────────────────────────────────────────────────────────────────
    A("## 1. Configuration")
    A("")
    A("| Parameter | Value |")
    A("|---|---|")
    A(f"| Excluded supercategories | {', '.join(sorted(cfg['exclude_supercats']))} |")
    A(f"| Min attribute samples (train) | {cfg['min_attr_samples']} |")
    A(f"| Drop crowd annotations | {cfg['drop_crowd']} |")
    A(f"| Drop zero-attribute annotations | {cfg['drop_zero_attr']} |")
    A(f"| Min bbox area (px²) | {cfg['min_bbox_area']} |")
    A(f"| Min bbox side (px) | {cfg['min_bbox_side']} |")
    A(f"| Apply category-attr masking | {cfg['apply_category_mask']} |")
    A("")

    # ── Raw dataset statistics ────────────────────────────────────────────────
    A("## 2. Raw Dataset Statistics")
    A("")
    A("| Metric | Train | Val |")
    A("|---|---|---|")
    A(f"| Total annotations | {len(train_raw_df):,} | {len(val_raw_df):,} |")
    A(f"| Total images | {train_raw_df['image_id'].nunique():,} | {val_raw_df['image_id'].nunique():,} |")
    A(f"| Annotations w/ attributes | {train_raw_df['has_attrs'].sum():,} | {val_raw_df['has_attrs'].sum():,} |")
    A(f"| % zero-attribute | {(~train_raw_df['has_attrs']).mean()*100:.1f}% | {(~val_raw_df['has_attrs']).mean()*100:.1f}% |")
    A(f"| Mean attrs per annotation | {train_raw_df['n_attributes'].mean():.2f} | {val_raw_df['n_attributes'].mean():.2f} |")
    A(f"| Crowd annotations | {int(train_raw_df['iscrowd'].sum()):,} | {int(val_raw_df['iscrowd'].sum()):,} |")
    A(f"| Total categories | 46 | 46 |")
    A(f"| Total attributes | 294 | 294 |")
    A("")

    # ── Cleaning steps ────────────────────────────────────────────────────────
    A("## 3. Cleaning Steps & Impact")
    A("")
    A("| Step | Split | Before | After | Removed | % Removed |")
    A("|---|---|---|---|---|---|")
    for split, r in cleaning_reports:
        if r.get("skipped"):
            continue
        A(f"| {r['op']} | {split} | {r['before']:,} | {r['after']:,} | {r['removed']:,} | {r['pct']:.1f}% |")
    A("")

    # ── Attribute label space ─────────────────────────────────────────────────
    A("## 4. Attribute Label Space")
    A("")
    A(f"- **Original attributes:** 294")
    A(f"- **After supercategory exclusion:** {len(freq_df)} "
      f"(excluded {294 - len(freq_df)} from {{{', '.join(sorted(cfg['exclude_supercats']))}}}) ")
    A(f"- **After rarity filter (≥{cfg['min_attr_samples']} samples):** {len(retained)}")
    A(f"- **Total attributes removed:** {294 - len(retained)}")
    A("")
    A("### Retained Attributes per Supercategory")
    A("")
    A("| Supercategory | Retained | Total Freq (train) |")
    A("|---|---|---|")
    for sc, grp in retained.groupby("supercat"):
        A(f"| {sc} | {len(grp)} | {grp['count'].sum():,} |")
    A("")
    A("### Dropped Attributes (sample, n_dropped lowest frequency first)")
    A("")
    A("| Attribute | Supercategory | Train Freq |")
    A("|---|---|---|")
    for _, row in dropped.sort_values("count").head(20).iterrows():
        A(f"| {row['name']} | {row['supercat']} | {row['count']} |")
    A("")

    # ── Processed dataset statistics ──────────────────────────────────────────
    A("## 5. Processed Dataset Statistics")
    A("")
    A("| Metric | Train | Val |")
    A("|---|---|---|")
    A(f"| Total annotations (crops) | {len(train_df):,} | {len(val_df):,} |")
    A(f"| Unique images | {train_df['image_id'].nunique():,} | {val_df['image_id'].nunique():,} |")
    A(f"| Unique categories | {train_df['category_id'].nunique()} | {val_df['category_id'].nunique()} |")
    A(f"| Label space size | {label_mappings['n_classes']} | {label_mappings['n_classes']} |")
    A(f"| Mean attrs per annotation | {train_df['n_attributes'].mean():.2f} | {val_df['n_attributes'].mean():.2f} |")
    A(f"| Median attrs per annotation | {train_df['n_attributes'].median():.0f} | {val_df['n_attributes'].median():.0f} |")
    A(f"| Max attrs per annotation | {train_df['n_attributes'].max()} | {val_df['n_attributes'].max()} |")
    A(f"| Annotations with negation attrs | "
      f"{train_df['has_negation_attr'].sum():,} ({train_df['has_negation_attr'].mean()*100:.1f}%) | "
      f"{val_df['has_negation_attr'].sum():,} ({val_df['has_negation_attr'].mean()*100:.1f}%) |")
    A("")

    # ── Class imbalance ───────────────────────────────────────────────────────
    A("## 6. Class Imbalance Analysis")
    A("")
    A(f"BCEWithLogitsLoss `pos_weight` statistics (n={label_mappings['n_classes']} classes):")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Min pos_weight | {pos_weight.min():.2f} |")
    A(f"| Median pos_weight | {np.median(pos_weight):.1f} |")
    A(f"| Mean pos_weight | {pos_weight.mean():.1f} |")
    A(f"| Max pos_weight | {pos_weight.max():.1f} |")
    A(f"| Classes with pw > 100 | {(pos_weight > 100).sum()} |")
    A(f"| Classes with pw > 50 | {(pos_weight > 50).sum()} |")
    A("")
    A("**Top-5 most imbalanced attributes:**")
    A("")
    A("| Attribute | pos_weight |")
    A("|---|---|")
    for i in np.argsort(pos_weight)[-5:][::-1]:
        A(f"| {label_mappings['idx_to_attr_name'][i]} | {pos_weight[i]:.0f} |")
    A("")
    A("**Top-5 most balanced attributes:**")
    A("")
    A("| Attribute | pos_weight |")
    A("|---|---|")
    for i in np.argsort(pos_weight)[:5]:
        A(f"| {label_mappings['idx_to_attr_name'][i]} | {pos_weight[i]:.2f} |")
    A("")

    # ── QA summary ────────────────────────────────────────────────────────────
    A("## 7. Quality Assurance Summary")
    A("")
    A("| Check | Train | Val |")
    A("|---|---|---|")
    A(f"| All-zero label vectors | "
      f"{qa_results['empty_labels_train']['all_zero_vectors']} | "
      f"{qa_results['empty_labels_val']['all_zero_vectors']} |")
    A(f"| Duplicate crops | "
      f"{qa_results['duplicates_train']['duplicate_crops']} | "
      f"{qa_results['duplicates_val']['duplicate_crops']} |")
    A(f"| Train/val image leakage | "
      f"image_id: {qa_results['data_leakage']['image_id_overlap']}, "
      f"filename: {qa_results['data_leakage']['filename_overlap']} | — |")
    A(f"| OOV attributes in val | — | "
      f"{qa_results['label_consistency']['oov_in_val']} |")
    A(f"| Missing images (train) | "
      f"{qa_results['missing_images_train']['missing']} / "
      f"{qa_results['missing_images_train']['total_images']} | — |")
    A(f"| Missing images (val) | — | "
      f"{qa_results['missing_images_val']['missing']} / "
      f"{qa_results['missing_images_val']['total_images']} |")
    A("")

    # ── Recommendations for training ─────────────────────────────────────────
    A("## 8. Recommendations for Training")
    A("")
    A("1. **Use `pos_weight` in BCEWithLogitsLoss.** "
      "The median class imbalance ratio is significant.  "
      "Load `metadata/pos_weights.json` and pass as `pos_weight` tensor.")
    A("")
    A("2. **Training unit is annotation-level crop.**  "
      "Use `bbox_x, bbox_y, bbox_w, bbox_h` to crop images before resizing to 224×224.")
    A("")
    A("3. **Category-conditioned evaluation.**  "
      "The `supercat_to_idxs` mapping in `label_mappings.json` allows computing "
      "per-supercategory mAP, which is more informative than flat mAP across "
      f"{label_mappings['n_classes']} classes.")
    A("")
    A("4. **Negation attribute awareness.**  "
      "The `has_negation_attr` column flags samples carrying negation-type labels "
      "('no non-textile material', 'no waistline', etc.).  "
      "Consider ablating with/without these labels to assess their impact.")
    A("")
    A("5. **Affinity map for advanced experiments.**  "
      "`metadata/category_attr_affinity.csv` provides empirical category–attribute "
      "co-occurrence.  This can be used as a soft prior (multiply logits by affinity) "
      "or as a hard mask (zero-out logits for inapplicable attributes).")
    A("")
    A("6. **`nickname` supercategory (excluded by default).**  "
      "If you wish to include nickname prediction, re-run with "
      "`exclude_supercats = set()` and increase `min_attr_samples` appropriately.  "
      "Consider a separate classification head for nicknames.")
    A("")

    report_str = "\n".join(lines)

    report_path = output_dir / "preprocessing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    log.info(f"Preprocessing report saved → {report_path}")

    return report_str


# ── Generate report ───────────────────────────────────────────────────────────
report_md = generate_preprocessing_report(
    train_raw_df, val_raw_df,
    train_df, val_df,
    rarity_report,
    label_mappings,
    pos_weight,
    cleaning_reports,
    qa_results,
    CFG,
    CFG["output_dir"],
)

print(report_md[:3000])  # Print first 3000 chars as preview


# =============================================================================
# CELL 13 — PYTORCH DATASET CLASS (READY-TO-USE)
# =============================================================================
# %%
# This class is provided as a reference implementation.
# It reads the exported CSV + NPY files produced by this pipeline.
#
# Usage in your training script:
#
#   from fashionpedia_preprocessing import FashionpediaAttrDataset
#   train_dataset = FashionpediaAttrDataset("./fashionpedia_processed", split="train")
#
# The __getitem__ method returns (crop_tensor, multi_hot_label).
# Crop is loaded from image_path, bbox is applied, then resized to 224×224.

PYTORCH_DATASET_CODE = '''
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
            x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), \
                         int(row["bbox_w"]), int(row["bbox_h"])
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
'''

# Save PyTorch dataset class
ds_path = CFG["output_dir"] / "fashionpedia_dataset.py"
with open(ds_path, "w", encoding="utf-8") as f:
    f.write(PYTORCH_DATASET_CODE.strip())
log.info(f"Saved PyTorch dataset class → {ds_path}")


# =============================================================================
# CELL 14 — FINAL SUMMARY PRINTOUT
# =============================================================================
# %%

print("\n" + "=" * 70)
print("PREPROCESSING PIPELINE — FINAL SUMMARY")
print("=" * 70)
print(f"""
  Raw dataset:
    Train: {len(train_raw_df):,} annotations over {train_raw_df['image_id'].nunique():,} images
    Val  : {len(val_raw_df):,} annotations over {val_raw_df['image_id'].nunique():,} images

  Processed dataset:
    Train: {len(train_df):,} annotation crops  ({len(train_df)/len(train_raw_df)*100:.1f}% of raw)
    Val  : {len(val_df):,} annotation crops  ({len(val_df)/len(val_raw_df)*100:.1f}% of raw)

  Label space:
    Original attributes          : 294
    After supercategory exclusion: {len(rarity_report['freq_df'])}  (excluded: {', '.join(sorted(CFG['exclude_supercats']))})
    After rarity filter (≥{CFG['min_attr_samples']})  : {label_mappings['n_classes']}  ← FINAL LABEL SPACE

  Class balance (BCEWithLogitsLoss pos_weight):
    Min    : {pos_weight.min():.2f}
    Median : {np.median(pos_weight):.1f}
    Max    : {pos_weight.max():.1f}

  QA:
    All-zero label vectors : train={qa_results['empty_labels_train']['all_zero_vectors']}, val={qa_results['empty_labels_val']['all_zero_vectors']}
    Train/val leakage      : {qa_results['data_leakage']['image_id_overlap']} overlapping images
    Duplicate crops        : train={qa_results['duplicates_train']['duplicate_crops']}, val={qa_results['duplicates_val']['duplicate_crops']}

  Output files:
    {CFG['output_dir'] / 'train.csv'}
    {CFG['output_dir'] / 'val.csv'}
    {CFG['output_dir'] / 'train_labels.npy'}
    {CFG['output_dir'] / 'val_labels.npy'}
    {CFG['output_dir'] / 'metadata' / 'label_mappings.json'}
    {CFG['output_dir'] / 'metadata' / 'pos_weights.json'}
    {CFG['output_dir'] / 'metadata' / 'category_attr_affinity.csv'}
    {CFG['output_dir'] / 'metadata' / 'attribute_frequencies.csv'}
    {CFG['output_dir'] / 'preprocessing_report.md'}
    {CFG['output_dir'] / 'fashionpedia_dataset.py'}
    {CFG['output_dir'] / 'figures'}/  (6 research figures)
""")
print("=" * 70)
print("✓ PIPELINE COMPLETE")
