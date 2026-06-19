# Fashionpedia Attribute Extraction — Preprocessing Report
**Generated:** 2026-06-17 09:06 UTC
**Pipeline version:** 1.0

---

## 1. Configuration

| Parameter | Value |
|---|---|
| Excluded supercategories | animal, leather, nickname |
| Min attribute samples (train) | 300 |
| Drop crowd annotations | True |
| Drop zero-attribute annotations | True |
| Min bbox area (px²) | 1024 |
| Min bbox side (px) | 16 |
| Apply category-attr masking | False |

## 2. Raw Dataset Statistics

| Metric | Train | Val |
|---|---|---|
| Total annotations | 333,401 | 8,781 |
| Total images | 45,623 | 1,158 |
| Annotations w/ attributes | 206,410 | 5,259 |
| % zero-attribute | 38.1% | 40.1% |
| Mean attrs per annotation | 2.28 | 2.36 |
| Crowd annotations | 0 | 0 |
| Total categories | 46 | 46 |
| Total attributes | 294 | 294 |

## 3. Cleaning Steps & Impact

| Step | Split | Before | After | Removed | % Removed |
|---|---|---|---|---|---|
| drop_crowd | train | 333,401 | 333,401 | 0 | 0.0% |
| drop_zero_attr | train | 333,401 | 206,410 | 126,991 | 38.1% |
| drop_small_bbox | train | 206,410 | 190,292 | 16,118 | 7.8% |
| detect_duplicates | train | 190,292 | 190,254 | 38 | 0.0% |
| drop_crowd | val | 8,781 | 8,781 | 0 | 0.0% |
| drop_zero_attr | val | 8,781 | 5,259 | 3,522 | 40.1% |
| drop_small_bbox | val | 5,259 | 4,777 | 482 | 9.2% |
| detect_duplicates | val | 4,777 | 4,776 | 1 | 0.0% |

## 4. Attribute Label Space

- **Original attributes:** 294
- **After supercategory exclusion:** 131 (excluded 163 from {animal, leather, nickname}) 
- **After rarity filter (≥300 samples):** 101
- **Total attributes removed:** 193

### Retained Attributes per Supercategory

| Supercategory | Retained | Total Freq (train) |
|---|---|---|
| length | 14 | 119,558 |
| neckline type | 20 | 26,555 |
| non-textile material type | 5 | 71,466 |
| opening type | 7 | 44,526 |
| silhouette | 19 | 131,383 |
| textile finishing, manufacturing techniques | 17 | 78,467 |
| textile pattern | 13 | 74,927 |
| waistline | 6 | 60,359 |

### Dropped Attributes (sample, n_dropped lowest frequency first)

| Attribute | Supercategory | Train Freq |
|---|---|---|
| ivory | non-textile material type | 1 |
| bone | non-textile material type | 2 |
| wood | non-textile material type | 4 |
| chained (opening) | opening type | 19 |
| bell bottom | silhouette | 43 |
| feather | non-textile material type | 43 |
| toggled (opening) | opening type | 58 |
| baggy | silhouette | 77 |
| herringbone (pattern) | textile pattern | 78 |
| embossed | textile finishing, manufacturing techniques | 86 |
| choker (neck) | neckline type | 88 |
| crossover (neck) | neckline type | 90 |
| buckled (opening) | opening type | 91 |
| houndstooth (pattern) | textile pattern | 97 |
| keyhole (neck) | neckline type | 108 |
| curved (fit) | silhouette | 111 |
| argyle | textile pattern | 114 |
| basque (wasitline) | waistline | 134 |
| camouflage | textile pattern | 147 |
| smocking | textile finishing, manufacturing techniques | 152 |

## 5. Processed Dataset Statistics

| Metric | Train | Val |
|---|---|---|
| Total annotations (crops) | 156,496 | 4,066 |
| Unique images | 45,589 | 1,158 |
| Unique categories | 21 | 16 |
| Label space size | 101 | 101 |
| Mean attrs per annotation | 3.88 | 4.07 |
| Median attrs per annotation | 1 | 1 |
| Max attrs per annotation | 14 | 12 |
| Annotations with negation attrs | 70,613 (45.1%) | 1,900 (46.7%) |

## 6. Class Imbalance Analysis

BCEWithLogitsLoss `pos_weight` statistics (n=101 classes):

| Metric | Value |
|---|---|
| Min pos_weight | 1.38 |
| Median pos_weight | 92.5 |
| Mean pos_weight | 135.8 |
| Max pos_weight | 515.5 |
| Classes with pw > 100 | 50 |
| Classes with pw > 50 | 64 |

**Top-5 most imbalanced attributes:**

| Attribute | pos_weight |
|---|---|
| paisley | 515 |
| queen anne (neck) | 473 |
| lace up | 445 |
| fair isle | 440 |
| toile de jouy | 429 |

**Top-5 most balanced attributes:**

| Attribute | pos_weight |
|---|---|
| no non-textile material | 1.38 |
| symmetrical | 1.61 |
| plain (pattern) | 1.68 |
| no special manufacturing technique | 3.09 |
| wrist-length | 3.86 |

## 7. Quality Assurance Summary

| Check | Train | Val |
|---|---|---|
| All-zero label vectors | 0 | 0 |
| Duplicate crops | 0 | 0 |
| Train/val image leakage | image_id: 0, filename: 0 | — |
| OOV attributes in val | — | 0 |
| Missing images (train) | 0 / 45589 | — |
| Missing images (val) | — | 0 / 1158 |

## 8. Recommendations for Training

1. **Use `pos_weight` in BCEWithLogitsLoss.** The median class imbalance ratio is significant.  Load `metadata/pos_weights.json` and pass as `pos_weight` tensor.

2. **Training unit is annotation-level crop.**  Use `bbox_x, bbox_y, bbox_w, bbox_h` to crop images before resizing to 224×224.

3. **Category-conditioned evaluation.**  The `supercat_to_idxs` mapping in `label_mappings.json` allows computing per-supercategory mAP, which is more informative than flat mAP across 101 classes.

4. **Negation attribute awareness.**  The `has_negation_attr` column flags samples carrying negation-type labels ('no non-textile material', 'no waistline', etc.).  Consider ablating with/without these labels to assess their impact.

5. **Affinity map for advanced experiments.**  `metadata/category_attr_affinity.csv` provides empirical category–attribute co-occurrence.  This can be used as a soft prior (multiply logits by affinity) or as a hard mask (zero-out logits for inapplicable attributes).

6. **`nickname` supercategory (excluded by default).**  If you wish to include nickname prediction, re-run with `exclude_supercats = set()` and increase `min_attr_samples` appropriately.  Consider a separate classification head for nicknames.
