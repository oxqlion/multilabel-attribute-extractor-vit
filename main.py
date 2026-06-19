#!/usr/bin/env python3
# =============================================================================
#  FASHIONPEDIA — Fashion Attribute Extraction + Description Generation API
#  Stack  : FastAPI + ConvNeXt-Tiny (MPS) + Qwen2.5-0.5B-Instruct (CPU/MPS)
#  Fallback: Claude Haiku API (if local LLM disabled)
#
#  POST /describe   → full image → attributes + product description (JSON)
#  GET  /health     → liveness check
#  GET  /docs       → Swagger UI (auto-generated)
# =============================================================================

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings
from app.models import PipelineManager
from app.schemas import DescribeResponse, HealthResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fashionpedia.api")

# ── Settings ──────────────────────────────────────────────────────────────────
settings = Settings()

# ── Pipeline (loaded once at startup, shared across requests) ─────────────────
pipeline: PipelineManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models at startup, release at shutdown.

    FastAPI's lifespan context manager replaces the deprecated @app.on_event
    pattern.  All heavy model loading happens here so the first request is
    not penalised.
    """
    global pipeline
    log.info("=" * 60)
    log.info(" FASHIONPEDIA API — Starting up")
    log.info("=" * 60)
    t0 = time.time()

    pipeline = PipelineManager(settings)
    await pipeline.load()

    log.info(f"All models loaded in {time.time() - t0:.1f}s")
    log.info("API ready to serve requests")
    log.info("=" * 60)

    yield  # ← API is live here

    log.info("Shutting down — releasing models")
    pipeline.release()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fashionpedia Attribute & Description API",
    description = (
        "Upload a full fashion product photo.  "
        "The API extracts garment attributes using a fine-tuned ConvNeXt-Tiny "
        "model and generates a product description using a local LLM."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (allow all origins for local dev; tighten for production) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# ROUTES
# =============================================================================

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness & readiness check",
    tags           = ["System"],
)
async def health() -> HealthResponse:
    """Returns API status and which models are loaded."""
    return HealthResponse(
        status          = "ok",
        models_loaded   = pipeline is not None and pipeline.is_ready,
        attribute_model = "ConvNeXt-Tiny (IMAGENET1K_V2 + Fashionpedia fine-tune)",
        llm_backend     = settings.llm_backend,
        llm_model       = settings.llm_model_name,
        device          = str(pipeline.device) if pipeline else "not loaded",
    )


@app.post(
    "/describe",
    response_model = DescribeResponse,
    summary        = "Extract attributes + generate product description",
    tags           = ["Inference"],
)
async def describe(
    file: UploadFile = File(
        ...,
        description = (
            "Full fashion product image (JPEG / PNG / WEBP). "
            "No cropping needed — the API auto-detects the garment region."
        ),
    ),
) -> DescribeResponse:
    """
    End-to-end fashion product description pipeline.

    **Steps (all server-side):**
    1. Validate and decode uploaded image
    2. Auto-detect the primary garment region (largest salient crop)
    3. Run ConvNeXt-Tiny attribute classifier on the crop
    4. Feed structured attributes to LLM → generate product description
    5. Return attributes + description as JSON

    **Returns:**
    - `attributes` — list of detected attributes with confidence scores
    - `attributes_by_group` — attributes grouped by supercategory
    - `description` — generated product description (2–3 sentences)
    - `meta` — timing, crop box, model info
    """
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(503, "Models not yet loaded — please retry in a moment")

    # ── Validate file type ────────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            400,
            f"Unsupported file type: {file.content_type}. "
            f"Accepted: {', '.join(allowed_types)}",
        )

    # ── Read raw bytes ────────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            413,
            f"Image too large: {len(image_bytes)/1e6:.1f}MB. "
            f"Maximum: {settings.max_image_bytes/1e6:.0f}MB",
        )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = await pipeline.run(image_bytes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        log.exception("Pipeline error")
        raise HTTPException(500, f"Internal pipeline error: {e}")

    return result
