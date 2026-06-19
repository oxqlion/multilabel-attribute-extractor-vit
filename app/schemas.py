# app/schemas.py
# =============================================================================
#  Pydantic models for API request / response validation and documentation.
# =============================================================================

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DetectedAttribute(BaseModel):
    """A single predicted fashion attribute."""
    attr_id:       int   = Field(..., description="Attribute ID from Fashionpedia taxonomy")
    name:          str   = Field(..., description="Human-readable attribute name")
    supercategory: str   = Field(..., description="Attribute supercategory (e.g. 'length', 'silhouette')")
    confidence:    float = Field(..., ge=0.0, le=1.0, description="Sigmoid confidence score [0, 1]")

    model_config = {"json_schema_extra": {
        "example": {
            "attr_id": 47,
            "name": "midi length",
            "supercategory": "length",
            "confidence": 0.87,
        }
    }}


class CropBox(BaseModel):
    """Bounding box of the garment region used for attribute extraction."""
    x:      int = Field(..., description="Left pixel coordinate")
    y:      int = Field(..., description="Top pixel coordinate")
    width:  int = Field(..., description="Crop width in pixels")
    height: int = Field(..., description="Crop height in pixels")


class InferenceMeta(BaseModel):
    """Timing and model metadata for the inference run."""
    image_size:         str   = Field(..., description="Original image dimensions WxH")
    crop_box:           CropBox
    detection_strategy: str   = Field(..., description="How the garment region was located")
    n_attributes_raw:   int   = Field(..., description="Total attributes above threshold before top-k")
    threshold_used:     float = Field(..., description="Sigmoid threshold applied")
    attribute_model:    str   = Field(..., description="ConvNeXt model identifier")
    llm_backend:        str   = Field(..., description="LLM used for description generation")
    llm_model:          str   = Field(..., description="LLM model identifier")
    timing: Dict[str, float]  = Field(..., description="Per-stage latency in seconds")


class DescribeResponse(BaseModel):
    """
    Full response from POST /describe.

    Contains:
    - `attributes`          → flat list sorted by confidence (highest first)
    - `attributes_by_group` → same attributes grouped by supercategory
    - `description`         → LLM-generated product description
    - `meta`                → timing + model info
    """
    attributes: List[DetectedAttribute] = Field(
        ...,
        description="All detected attributes, sorted by confidence descending",
    )
    attributes_by_group: Dict[str, List[DetectedAttribute]] = Field(
        ...,
        description="Attributes grouped by supercategory (e.g. 'length', 'pattern')",
    )
    description: str = Field(
        ...,
        description="LLM-generated product description based on extracted attributes",
    )
    meta: InferenceMeta

    model_config = {"json_schema_extra": {
        "example": {
            "attributes": [
                {"attr_id": 47,  "name": "midi length",      "supercategory": "length",           "confidence": 0.87},
                {"attr_id": 102, "name": "floral (pattern)", "supercategory": "textile pattern",  "confidence": 0.81},
                {"attr_id": 58,  "name": "A-line",           "supercategory": "silhouette",        "confidence": 0.74},
            ],
            "attributes_by_group": {
                "length":          [{"attr_id": 47,  "name": "midi length",      "supercategory": "length",          "confidence": 0.87}],
                "textile pattern": [{"attr_id": 102, "name": "floral (pattern)", "supercategory": "textile pattern", "confidence": 0.81}],
                "silhouette":      [{"attr_id": 58,  "name": "A-line",           "supercategory": "silhouette",       "confidence": 0.74}],
            },
            "description": (
                "A charming midi-length dress featuring a vibrant floral print "
                "in an elegant A-line silhouette. Perfect for both casual daytime "
                "outings and semi-formal occasions, this piece effortlessly combines "
                "femininity with versatile style."
            ),
            "meta": {
                "image_size": "800x1200",
                "crop_box": {"x": 60, "y": 90, "width": 680, "height": 1020},
                "detection_strategy": "centrecrop",
                "n_attributes_raw": 14,
                "threshold_used": 0.45,
                "attribute_model": "ConvNeXt-Tiny (IMAGENET1K_V2 + Fashionpedia)",
                "llm_backend": "local",
                "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "timing": {"detection_s": 0.01, "attribute_s": 0.18, "llm_s": 1.42, "total_s": 1.61},
            },
        }
    }}


class HealthResponse(BaseModel):
    """GET /health response."""
    status:          str  = Field(..., description="'ok' if all models are loaded")
    models_loaded:   bool
    attribute_model: str
    llm_backend:     str
    llm_model:       str
    device:          str
