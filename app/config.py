# app/config.py
# =============================================================================
#  All configuration is read from environment variables (with defaults).
#  Create a .env file in the project root to override without touching code.
#
#  Example .env:
#      CHECKPOINT_PATH=./fashionpedia_runs/ablation_B_stage4_head/checkpoints/best.pt
#      PROCESSED_DIR=./fashionpedia_processed
#      LLM_BACKEND=local
#      ATTRIBUTE_THRESHOLD=0.45
# =============================================================================

from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.  All values can be overridden via environment
    variables or a .env file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_file        = ".env",
        env_file_encoding = "utf-8",
        case_sensitive  = False,
        extra           = "ignore",
    )

    # ── Model paths ───────────────────────────────────────────────────────
    checkpoint_path: Path = Field(
        default = Path("./fashionpedia_runs/ablation_B_stage4_head/checkpoints/best.pt"),
        description = "Path to trained ConvNeXt-Tiny checkpoint (.pt)",
    )
    processed_dir: Path = Field(
        default = Path("./fashionpedia_processed"),
        description = "Path to preprocessing output directory (label_mappings.json etc.)",
    )

    # ── Attribute classifier settings ─────────────────────────────────────
    attribute_threshold: float = Field(
        default     = 0.45,
        ge          = 0.0,
        le          = 1.0,
        description = "Sigmoid threshold for positive attribute prediction",
    )
    top_k_attributes: int = Field(
        default     = 20,
        description = "Maximum number of attributes to return (sorted by confidence)",
    )
    img_size: int = Field(default=224)

    # ── LLM backend ───────────────────────────────────────────────────────
    llm_backend: Literal["local", "claude"] = Field(
        default     = "local",
        description = "LLM backend: 'local' (Qwen2.5-0.5B) or 'claude' (Haiku API)",
    )

    # Local LLM (Qwen2.5-0.5B-Instruct)
    llm_model_name: str = Field(
        default     = "Qwen/Qwen2.5-0.5B-Instruct",
        description = "HuggingFace model ID for local LLM",
    )
    llm_max_new_tokens: int  = Field(default=220)
    llm_temperature:    float = Field(default=0.7)
    llm_device:         str  = Field(
        default     = "auto",
        description = "Device for local LLM: 'auto', 'mps', 'cpu'",
    )

    # Claude API fallback
    anthropic_api_key: str = Field(
        default     = "",
        description = "Anthropic API key (required if llm_backend=claude)",
    )
    claude_model: str = Field(
        default     = "claude-haiku-4-5-20251001",
        description = "Claude model ID to use when llm_backend=claude",
    )

    # ── Detection settings ────────────────────────────────────────────────
    # Strategy for finding the garment region in a full product photo.
    # 'centrecrop'  : fast, zero extra model — use centre 80% of image.
    #                 Works well for standard e-commerce white-bg photos.
    # 'saliency'    : use GradCAM-lite saliency on the classifier itself.
    #                 Best quality but slightly slower.
    detection_strategy: Literal["centrecrop", "saliency"] = Field(
        default     = "centrecrop",
        description = "Garment region detection strategy",
    )
    # Fraction of image to keep in centrecrop mode (0.8 = centre 80%)
    centrecrop_fraction: float = Field(default=0.85)

    # ── API settings ──────────────────────────────────────────────────────
    max_image_bytes: int = Field(
        default     = 15 * 1024 * 1024,   # 15 MB
        description = "Maximum accepted image upload size in bytes",
    )
    cors_origins: List[str] = Field(
        default     = ["*"],
        description = "Allowed CORS origins",
    )

    # ── Model architecture (must match training config) ───────────────────
    head_hidden_dim:     int   = Field(default=512)
    head_dropout:        float = Field(default=0.0)   # always 0 at inference
    unfreeze_from_block: int   = Field(default=6)

    @field_validator("checkpoint_path", "processed_dir", mode="before")
    @classmethod
    def to_path(cls, v):
        return Path(v)
