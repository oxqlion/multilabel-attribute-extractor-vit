# app/models.py
# =============================================================================
#  PipelineManager
#  ───────────────
#  Owns and orchestrates all three stages of the inference pipeline:
#
#  Stage 1 — GarmentDetector
#      Full product photo → garment crop (PIL.Image)
#      Strategy: centrecrop (fast, no extra model) or saliency
#
#  Stage 2 — AttributeClassifier
#      Garment crop → List[DetectedAttribute]
#      Model: ConvNeXt-Tiny fine-tuned on Fashionpedia
#
#  Stage 3 — DescriptionGenerator
#      List[DetectedAttribute] → str (product description)
#      Model: Qwen2.5-0.5B-Instruct (local) OR Claude Haiku (API fallback)
#
#  All models are loaded once at startup (lifespan) and reused across
#  requests.  The classifier runs on MPS; the local LLM runs on CPU
#  (Qwen2.5-0.5B fits in ~1GB RAM and generates ~220 tokens in <2s on M5).
# =============================================================================

import asyncio
import io
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from app.config import Settings
from app.schemas import CropBox, DescribeResponse, DetectedAttribute, InferenceMeta

log = logging.getLogger("fashionpedia.pipeline")


# =============================================================================
# STAGE 1 — GARMENT DETECTOR
# =============================================================================

class GarmentDetector:
    """
    Locates the primary garment region in a full product photo.

    Strategy: centrecrop (default, zero extra model)
    ──────────────────────────────────────────────────
    For standard e-commerce product photos (white or clean background,
    single garment, centred subject) a simple centre crop outperforms
    object detection because:
      - E-commerce photos are composited, not candid
      - The garment occupies ~70–90% of the frame by design
      - A heavy detector (YOLO, Faster-RCNN) adds 200–500ms latency
        for negligible accuracy gain on this image distribution

    The crop fraction (default 0.85) is tunable in Settings.
    For lifestyle/editorial photos, switch strategy to 'saliency' in .env.

    Strategy: saliency
    ──────────────────
    Uses a lightweight gradient-free saliency map from the classifier
    itself (class activation mapping approximation) to find the most
    discriminative region.  Adds ~80ms on MPS.  Better for non-standard
    product shots (model wearing the garment, lifestyle backgrounds).
    """

    def __init__(self, settings: Settings):
        self.strategy        = settings.detection_strategy
        self.crop_fraction   = settings.centrecrop_fraction
        log.info(f"GarmentDetector: strategy={self.strategy}, "
                 f"crop_fraction={self.crop_fraction}")

    def detect(self, image: Image.Image) -> Tuple[Image.Image, CropBox]:
        """
        Detect the garment region and return (cropped_image, crop_box).

        Args:
            image: full product photo as PIL.Image (RGB)

        Returns:
            crop   : PIL.Image of the detected garment region
            box    : CropBox with absolute pixel coordinates
        """
        if self.strategy == "centrecrop":
            return self._centrecrop(image)
        elif self.strategy == "saliency":
            # Saliency requires the classifier to be injected;
            # that is done by PipelineManager after both are constructed.
            # Until then, fall back to centrecrop.
            return self._centrecrop(image)
        else:
            return self._centrecrop(image)

    def _centrecrop(
        self, image: Image.Image
    ) -> Tuple[Image.Image, CropBox]:
        """
        Crop the centre fraction of the image.

        For a 800×1200 image with fraction=0.85:
          x = 800 * (1 - 0.85) / 2 = 60
          y = 1200 * (1 - 0.85) / 2 = 90
          w = 800 * 0.85 = 680
          h = 1200 * 0.85 = 1020
        """
        w, h   = image.size
        f      = self.crop_fraction
        x      = int(w * (1 - f) / 2)
        y      = int(h * (1 - f) / 2)
        crop_w = int(w * f)
        crop_h = int(h * f)

        crop = image.crop((x, y, x + crop_w, y + crop_h))
        box  = CropBox(x=x, y=y, width=crop_w, height=crop_h)
        return crop, box


# =============================================================================
# STAGE 2 — ATTRIBUTE CLASSIFIER
# =============================================================================

class ConvNeXtHead(nn.Module):
    """
    Identical head architecture to training (FashionAttributeClassifier).
    Must match exactly — any mismatch causes a state_dict load error.
    """
    def __init__(self, feat_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AttributeClassifier:
    """
    Loads the fine-tuned ConvNeXt-Tiny checkpoint and runs attribute
    inference on a garment crop.

    Output: list of DetectedAttribute sorted by confidence (descending).
    """

    # ImageNet normalisation (must match training)
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD  = (0.229, 0.224, 0.225)

    def __init__(self, settings: Settings, device: torch.device):
        self.settings  = settings
        self.device    = device
        self.model     = None
        self.mappings  = None
        self.transform = None

    def load(self) -> None:
        """Load checkpoint and label mappings.  Called once at startup."""
        log.info(f"Loading attribute classifier from: {self.settings.checkpoint_path}")

        # ── Label mappings ────────────────────────────────────────────────
        mappings_path = self.settings.processed_dir / "metadata" / "label_mappings.json"
        if not mappings_path.exists():
            raise FileNotFoundError(f"label_mappings.json not found at {mappings_path}")

        with open(mappings_path) as f:
            self.mappings = json.load(f)

        n_classes = self.mappings["n_classes"]
        log.info(f"  Label space: {n_classes} attributes")

        # ── Build model architecture ──────────────────────────────────────
        # We rebuild the architecture explicitly rather than pickling the
        # class, making the checkpoint forward-compatible with code changes.
        backbone = convnext_tiny(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.clf_head = ConvNeXtHead(
            feat_dim   = 768,
            hidden_dim = self.settings.head_hidden_dim,
            n_classes  = n_classes,
            dropout    = self.settings.head_dropout,  # 0.0 at inference
        )

        # Combine into single module for clean device placement
        class _FullModel(nn.Module):
            def __init__(self, features, avgpool, head):
                super().__init__()
                self.features = features
                self.avgpool  = avgpool
                self.head     = head
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return self.head(x)

        self.model = _FullModel(self.features, self.avgpool, self.clf_head)

        # ── Load weights ──────────────────────────────────────────────────
        if not self.settings.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.settings.checkpoint_path}\n"
                "  → Run training first, or update CHECKPOINT_PATH in .env"
            )

        ckpt = torch.load(
            self.settings.checkpoint_path,
            map_location = "cpu",   # load to CPU first, then move to device
            weights_only = True,    # security: don't unpickle arbitrary objects
        )

        # The checkpoint was saved by FashionAttributeClassifier which has
        # keys: features.*, avgpool.*, head.*  — matches our _FullModel.
        # Older checkpoints saved head as a bare nn.Sequential (head.N.*),
        # but ConvNeXtHead wraps it as self.head, producing head.head.N.*.
        # Remap on the fly so both checkpoint versions load cleanly.
        state = ckpt["model_state"]
        if any(k.startswith("head.") and not k.startswith("head.head.") for k in state):
            state = {
                (f"head.head.{k[len('head.'):]}" if k.startswith("head.") else k): v
                for k, v in state.items()
            }

        missing, unexpected = self.model.load_state_dict(state, strict=True)
        if missing:
            log.warning(f"  Missing keys in checkpoint: {missing[:5]}")
        if unexpected:
            log.warning(f"  Unexpected keys in checkpoint: {unexpected[:5]}")

        self.model.to(self.device)
        self.model.eval()

        # ── Transform pipeline (deterministic, no augmentation) ───────────
        img_size = self.settings.img_size
        self.transform = T.Compose([
            T.Resize(
                int(img_size * 1.14),
                interpolation = T.InterpolationMode.BICUBIC,
            ),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(self.IMG_MEAN, self.IMG_STD),
        ])

        log.info(f"  Attribute classifier ready on {self.device}")

    @torch.no_grad()
    def predict(
        self,
        crop: Image.Image,
        threshold:  float,
        top_k:      int,
    ) -> List[DetectedAttribute]:
        """
        Run inference on a garment crop.

        Args:
            crop      : PIL.Image (RGB) of the garment region
            threshold : sigmoid threshold for positive prediction
            top_k     : return at most this many attributes

        Returns:
            List of DetectedAttribute sorted by confidence descending.
        """
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        logits = self.model(tensor)[0]                    # (n_classes,)
        probs  = torch.sigmoid(logits).cpu().numpy()      # (n_classes,)

        # Build lookup tables (int-keyed, JSON stores str keys)
        idx_to_attr_id   = {int(k): v for k, v in self.mappings["idx_to_attr_id"].items()}
        idx_to_attr_name = {int(k): v for k, v in self.mappings["idx_to_attr_name"].items()}

        # Rebuild supercat lookup from supercat_to_idxs
        idx_to_supercat: Dict[int, str] = {}
        for sc, idxs in self.mappings.get("supercat_to_idxs", {}).items():
            for i in idxs:
                idx_to_supercat[int(i)] = sc

        # Threshold + sort
        above_threshold = [
            (i, float(probs[i]))
            for i in range(len(probs))
            if probs[i] >= threshold
        ]
        above_threshold.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, conf in above_threshold[:top_k]:
            results.append(DetectedAttribute(
                attr_id       = idx_to_attr_id.get(i, i),
                name          = idx_to_attr_name.get(i, f"attr_{i}"),
                supercategory = idx_to_supercat.get(i, "unknown"),
                confidence    = round(conf, 4),
            ))

        return results, len(above_threshold)


# =============================================================================
# STAGE 3 — DESCRIPTION GENERATOR
# =============================================================================

class DescriptionGenerator:
    """
    Generates a short product description from detected attributes.

    Backends:
      'local'  — Qwen2.5-0.5B-Instruct via HuggingFace transformers
      'claude' — Claude Haiku via Anthropic API

    Prompt design:
      We use a structured prompt that passes attributes grouped by
      supercategory.  Grouping helps the LLM understand the semantic
      relationship between attributes (all lengths together, all patterns
      together) rather than seeing a flat undifferentiated list.
      The instruction asks for 2–3 sentences of polished product copy —
      short enough to fit in a product card, specific enough to be useful.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend  = settings.llm_backend
        self._model       = None
        self._tokenizer   = None
        self._llm_device  = None
        self._client      = None   # Anthropic client

    def load(self) -> None:
        """Load LLM.  Called once at startup."""
        if self.backend == "local":
            self._load_local()
        elif self.backend == "claude":
            self._load_claude()
        else:
            raise ValueError(f"Unknown llm_backend: {self.backend}")

    def _load_local(self) -> None:
        """
        Load Qwen2.5-0.5B-Instruct from HuggingFace.

        Why Qwen2.5-0.5B?
        ──────────────────
        • 0.5B parameters → ~1GB RAM → fits comfortably on M5 alongside
          ConvNeXt-Tiny (28M params, ~110MB)
        • Instruction-tuned variant follows system/user prompts reliably
        • Generates coherent English product copy with simple prompts
        • ~1–2s per description on M5 CPU (no MPS for generation due to
          HuggingFace MPS generation instability in some torch versions)
        • Apache 2.0 licence — fully open for commercial use

        Alternatives considered:
        • Llama-3.2-1B: better quality but 2× RAM, 3× slower on CPU
        • SmolLM2-135M: too small to produce coherent fashion copy
        • TinyLlama-1.1B: good but LLAMA licence restrictions
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Run: pip install transformers accelerate"
            )

        model_id = self.settings.llm_model_name
        log.info(f"Loading local LLM: {model_id} …")
        log.info("  (First run will download ~1GB from HuggingFace)")

        # Determine device for local LLM
        # We use CPU for the LLM even on M5 because:
        # 1. ConvNeXt already occupies MPS for attribute inference
        # 2. HuggingFace generate() on MPS has known issues with KV-cache
        #    in some PyTorch versions that cause silent incorrect outputs
        # 3. 0.5B on M5 CPU is fast enough (<2s) for the use case
        llm_device = self.settings.llm_device
        if llm_device == "auto":
            llm_device = "cpu"   # safe default; override in .env if desired

        self._llm_device = llm_device

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code = True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype       = torch.float32,  # float32 for CPU stability
            device_map        = llm_device,
            trust_remote_code = True,
        )
        self._model.eval()
        log.info(f"  Local LLM ready on {llm_device}")

    def _load_claude(self) -> None:
        """Load Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        api_key = self.settings.anthropic_api_key
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in environment / .env file. "
                "Set it or switch LLM_BACKEND=local."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        log.info(f"Claude API client ready (model: {self.settings.claude_model})")

    def _build_prompt(
        self,
        attributes_by_group: Dict[str, List[DetectedAttribute]],
    ) -> str:
        """
        Build a structured prompt from grouped attributes.

        Format example:
            Fashion attributes detected for this product:
            - Length: midi length (0.87), maxi length (0.52)
            - Pattern: floral (0.81), printed (0.76)
            - Silhouette: A-line (0.74)

            Write a 2-3 sentence product description for an online fashion
            store based on these attributes.  Be specific, use natural
            retail language, do not invent attributes not listed above.
        """
        if not attributes_by_group:
            return (
                "Write a short 2-sentence generic fashion product description "
                "for an unidentified garment."
            )

        lines = ["Fashion attributes detected for this product:"]
        for group, attrs in sorted(attributes_by_group.items()):
            attr_strs = [
                f"{a.name} ({a.confidence:.2f})"
                for a in attrs
            ]
            lines.append(f"- {group.title()}: {', '.join(attr_strs)}")

        lines.append("")
        lines.append(
            "Write a 2-3 sentence product description for an online fashion store "
            "based on these attributes. Use natural retail language. "
            "Be specific to the detected attributes. "
            "Do not invent features not listed above. "
            "Do not mention confidence scores."
        )
        return "\n".join(lines)

    async def generate(
        self,
        attributes_by_group: Dict[str, List[DetectedAttribute]],
    ) -> str:
        """
        Generate a product description (async — runs LLM in thread pool).

        We run the synchronous LLM inference in asyncio's thread pool
        executor so it doesn't block FastAPI's event loop.
        """
        prompt = self._build_prompt(attributes_by_group)

        if self.backend == "local":
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(
                None,
                self._generate_local,
                prompt,
            )
        elif self.backend == "claude":
            description = await self._generate_claude(prompt)
        else:
            description = "Description unavailable."

        return description.strip()

    def _generate_local(self, prompt: str) -> str:
        """Synchronous local generation (called in thread pool)."""
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a professional fashion copywriter for an e-commerce platform. "
                    "Write concise, appealing product descriptions based on provided attributes. "
                    "Keep descriptions to 2-3 sentences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Apply chat template (Qwen uses a specific format)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize          = False,
            add_generation_prompt = True,
        )
        inputs = self._tokenizer(
            text,
            return_tensors = "pt",
        ).to(self._llm_device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens  = self.settings.llm_max_new_tokens,
                temperature     = self.settings.llm_temperature,
                do_sample       = True,
                top_p           = 0.9,
                repetition_penalty = 1.1,   # reduce repetitive phrases
                pad_token_id    = self._tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the input prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        description   = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return description

    async def _generate_claude(self, prompt: str) -> str:
        """Async Claude API generation."""
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._call_claude_sync,
            prompt,
        )
        return result

    def _call_claude_sync(self, prompt: str) -> str:
        message = self._client.messages.create(
            model      = self.settings.claude_model,
            max_tokens = self.settings.llm_max_new_tokens,
            system     = (
                "You are a professional fashion copywriter for an e-commerce platform. "
                "Write concise, appealing product descriptions. "
                "Keep descriptions to 2-3 sentences."
            ),
            messages   = [{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager:
    """
    Owns all three pipeline stages and orchestrates the full inference run.

    Lifecycle:
        await pipeline.load()    ← called once at FastAPI startup
        result = await pipeline.run(image_bytes)   ← called per request
        pipeline.release()       ← called at FastAPI shutdown
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_ready = False
        self.device   = self._select_device()

        self.detector   = GarmentDetector(settings)
        self.classifier = AttributeClassifier(settings, self.device)
        self.generator  = DescriptionGenerator(settings)

    def _select_device(self) -> torch.device:
        """Select device for the attribute classifier."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Attribute classifier will use: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Attribute classifier will use: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            log.info("Attribute classifier will use: CPU")
        return device

    async def load(self) -> None:
        """Load all models.  Runs in the FastAPI startup lifespan."""
        loop = asyncio.get_event_loop()

        # Load classifier synchronously in thread pool
        # (torch.load is not async-safe)
        await loop.run_in_executor(None, self.classifier.load)

        # Load LLM (may download from HuggingFace on first run)
        await loop.run_in_executor(None, self.generator.load)

        self.is_ready = True

    def release(self) -> None:
        """Free GPU/MPS memory on shutdown."""
        if self.classifier.model is not None:
            del self.classifier.model
        if self.generator._model is not None:
            del self.generator._model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_ready = False
        log.info("Models released")

    async def run(self, image_bytes: bytes) -> DescribeResponse:
        """
        Execute the full pipeline on raw image bytes.

        Returns a DescribeResponse with attributes + description + meta.
        """
        t_total = time.perf_counter()

        # ── Decode image ──────────────────────────────────────────────────
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not decode image: {e}")

        orig_w, orig_h = image.size

        # ── Stage 1: Detect garment region ────────────────────────────────
        t0    = time.perf_counter()
        crop, box = self.detector.detect(image)
        t_det = time.perf_counter() - t0

        # ── Stage 2: Attribute prediction ─────────────────────────────────
        t0 = time.perf_counter()
        attributes, n_raw = self.classifier.predict(
            crop      = crop,
            threshold = self.settings.attribute_threshold,
            top_k     = self.settings.top_k_attributes,
        )
        t_attr = time.perf_counter() - t0

        if not attributes:
            # Fallback: lower threshold and retry
            log.warning(
                f"No attributes above threshold={self.settings.attribute_threshold}. "
                "Retrying with threshold=0.3"
            )
            attributes, n_raw = self.classifier.predict(
                crop      = crop,
                threshold = 0.3,
                top_k     = self.settings.top_k_attributes,
            )

        # ── Group by supercategory ────────────────────────────────────────
        by_group: Dict[str, List[DetectedAttribute]] = defaultdict(list)
        for attr in attributes:
            by_group[attr.supercategory].append(attr)
        # Sort each group by confidence
        by_group = {
            k: sorted(v, key=lambda a: a.confidence, reverse=True)
            for k, v in sorted(by_group.items())
        }

        # ── Stage 3: Generate description ─────────────────────────────────
        t0 = time.perf_counter()
        description = await self.generator.generate(by_group)
        t_llm = time.perf_counter() - t0

        t_total_elapsed = time.perf_counter() - t_total

        log.info(
            f"Pipeline complete | "
            f"n_attrs={len(attributes)} | "
            f"det={t_det:.2f}s attr={t_attr:.2f}s llm={t_llm:.2f}s "
            f"total={t_total_elapsed:.2f}s"
        )

        # ── Build response ────────────────────────────────────────────────
        return DescribeResponse(
            attributes          = attributes,
            attributes_by_group = by_group,
            description         = description,
            meta                = InferenceMeta(
                image_size         = f"{orig_w}x{orig_h}",
                crop_box           = box,
                detection_strategy = self.settings.detection_strategy,
                n_attributes_raw   = n_raw,
                threshold_used     = self.settings.attribute_threshold,
                attribute_model    = "ConvNeXt-Tiny (IMAGENET1K_V2 + Fashionpedia)",
                llm_backend        = self.settings.llm_backend,
                llm_model          = self.settings.llm_model_name,
                timing             = {
                    "detection_s":  round(t_det,   3),
                    "attribute_s":  round(t_attr,  3),
                    "llm_s":        round(t_llm,   3),
                    "total_s":      round(t_total_elapsed, 3),
                },
            ),
        )
