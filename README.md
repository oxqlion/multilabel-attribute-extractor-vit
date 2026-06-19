# Fashionpedia Attribute & Description API

End-to-end backend for fashion product description generation.

```
POST /describe  →  image → attributes (JSON) + product description (string)
```

## Architecture

```
Frontend (any)
     │  POST /describe  multipart/form-data  image file
     ▼
FastAPI  (uvicorn, async)
     │
     ├─ Stage 1 ─ GarmentDetector
     │             Full photo → centre crop (85% of image)
     │             Zero extra model, <5ms
     │
     ├─ Stage 2 ─ AttributeClassifier  (ConvNeXt-Tiny, MPS)
     │             Crop → 294-class attribute sigmoid scores
     │             Filter by threshold → top-K attributes
     │             ~180ms on M5 MPS
     │
     └─ Stage 3 ─ DescriptionGenerator
                   Grouped attributes → product description
                   Local:  Qwen2.5-0.5B-Instruct (CPU, ~1.5s)
                   Fallback: Claude Haiku API (~0.8s)
```

## Quick Start

### 1. Install

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install PyTorch (M5 Mac — MPS included in standard wheel)
pip install torch torchvision

# Install API dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set CHECKPOINT_PATH and PROCESSED_DIR
```

### 3. Verify your checkpoint first (no server needed)

```bash
python test_inference.py \
  --image path/to/your/product_photo.jpg \
  --checkpoint ./fashionpedia_runs/ablation_B_stage4_head/checkpoints/best.pt \
  --processed-dir ./fashionpedia_processed
```

Output:
```
Device: MPS (Apple Silicon)
Label space: 141 attributes
Image size: 800×1200
Checkpoint loaded ✓

── Attributes (12 detected in 183ms) ──

  LENGTH:
    ████████████████     0.872  midi length
    ████████████         0.601  below-knee length

  TEXTILE PATTERN:
    █████████████████    0.812  floral (pattern)
    ██████████████       0.701  printed

  SILHOUETTE:
    ██████████████       0.743  A-line

── Description (generated in 1.4s) ──

  "A charming midi-length dress featuring a vibrant floral print in an elegant
   A-line silhouette. The below-knee hem and flowing printed fabric make this
   piece perfect for warm-weather occasions."

✓ Full JSON output saved → product_photo_output.json
```

### 4. Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API is live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## API Reference

### `POST /describe`

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image file | JPEG / PNG / WEBP, max 15MB |

**Response:** `application/json`

```json
{
  "attributes": [
    {
      "attr_id": 47,
      "name": "midi length",
      "supercategory": "length",
      "confidence": 0.872
    }
  ],
  "attributes_by_group": {
    "length": [
      { "attr_id": 47, "name": "midi length", "supercategory": "length", "confidence": 0.872 }
    ],
    "textile pattern": [
      { "attr_id": 102, "name": "floral (pattern)", "supercategory": "textile pattern", "confidence": 0.812 }
    ]
  },
  "description": "A charming midi-length dress featuring a vibrant floral print...",
  "meta": {
    "image_size": "800x1200",
    "crop_box": { "x": 60, "y": 90, "width": 680, "height": 1020 },
    "detection_strategy": "centrecrop",
    "n_attributes_raw": 14,
    "threshold_used": 0.45,
    "attribute_model": "ConvNeXt-Tiny (IMAGENET1K_V2 + Fashionpedia)",
    "llm_backend": "local",
    "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "timing": {
      "detection_s": 0.004,
      "attribute_s": 0.183,
      "llm_s": 1.42,
      "total_s": 1.607
    }
  }
}
```

### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "attribute_model": "ConvNeXt-Tiny (IMAGENET1K_V2 + Fashionpedia fine-tune)",
  "llm_backend": "local",
  "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "mps"
}
```

---

## Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Describe a product image
curl -X POST http://localhost:8000/describe \
  -F "file=@/path/to/your/dress.jpg" \
  | python3 -m json.tool

# Save output to file
curl -X POST http://localhost:8000/describe \
  -F "file=@dress.jpg" \
  -o output.json
```

---

## Switching to Claude Haiku (faster, no local download)

```bash
# In .env:
LLM_BACKEND=claude
ANTHROPIC_API_KEY=sk-ant-...
```

Restart the server — no code changes needed.

---

## Project Structure

```
fashionpedia_api/
├── main.py                  ← FastAPI app, routes, lifespan
├── app/
│   ├── __init__.py
│   ├── config.py            ← Settings (pydantic-settings + .env)
│   ├── models.py            ← PipelineManager, all 3 stages
│   └── schemas.py           ← Pydantic request/response models
├── test_inference.py        ← Standalone test (no server needed)
├── requirements.txt
├── .env.example             ← Copy to .env and configure
└── README.md
```

---

## Expected Latency on M5 MacBook Pro

| Stage | Time |
|---|---|
| Image decode + crop | ~5ms |
| ConvNeXt-Tiny (MPS) | ~150–200ms |
| Qwen2.5-0.5B (CPU) | ~1.2–2.0s |
| **Total** | **~1.5–2.2s** |

Switching to Claude Haiku brings total to ~0.9–1.2s (network dependent).
