#!/usr/bin/env python3
# =============================================================================
#  test_inference.py — Standalone inference test (no API server needed)
#
#  Use this to verify your checkpoint works BEFORE starting the API.
#
#  Usage:
#      python test_inference.py --image path/to/product_photo.jpg
#      python test_inference.py --image photo.jpg --threshold 0.4 --top-k 15
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

# ── Minimal standalone inference (mirrors app/models.py but without FastAPI) ─

def run_inference(
    image_path:      str,
    checkpoint_path: str,
    processed_dir:   str,
    threshold:       float = 0.45,
    top_k:           int   = 20,
    llm_backend:     str   = "local",
    anthropic_key:   str   = "",
):
    import io
    import json
    import torch
    import torch.nn as nn
    import torchvision.transforms.v2 as T
    from collections import defaultdict
    from PIL import Image
    from torchvision.models import convnext_tiny

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU")

    # ── Load label mappings ───────────────────────────────────────────────
    mappings_path = Path(processed_dir) / "metadata" / "label_mappings.json"
    if not mappings_path.exists():
        print(f"ERROR: label_mappings.json not found at {mappings_path}")
        sys.exit(1)

    with open(mappings_path) as f:
        mappings = json.load(f)

    n_classes        = mappings["n_classes"]
    idx_to_attr_id   = {int(k): v for k, v in mappings["idx_to_attr_id"].items()}
    idx_to_attr_name = {int(k): v for k, v in mappings["idx_to_attr_name"].items()}
    idx_to_supercat  = {}
    for sc, idxs in mappings.get("supercat_to_idxs", {}).items():
        for i in idxs:
            idx_to_supercat[int(i)] = sc

    print(f"Label space: {n_classes} attributes")

    # ── Build & load model ────────────────────────────────────────────────
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone      = convnext_tiny(weights=None)
            self.features = backbone.features
            self.avgpool  = backbone.avgpool
            self.head     = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(768),
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(512, n_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return self.head(x)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("  → Train first, or update --checkpoint argument")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} …")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = FullModel()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    print("Checkpoint loaded ✓")

    # ── Transform ─────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ── Load & crop image ─────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size
    print(f"Image size: {w}×{h}")

    # Centre crop (85% of image)
    f    = 0.85
    x    = int(w * (1 - f) / 2)
    y    = int(h * (1 - f) / 2)
    crop = image.crop((x, y, x + int(w * f), y + int(h * f)))

    # ── Inference ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        tensor = transform(crop).unsqueeze(0).to(device)
        logits = model(tensor)[0]
        probs  = torch.sigmoid(logits).cpu().numpy()
    t_attr = time.perf_counter() - t0

    # Threshold + sort
    results = sorted(
        [(i, float(probs[i])) for i in range(len(probs)) if probs[i] >= threshold],
        key=lambda x: x[1], reverse=True,
    )[:top_k]

    attributes = [
        {
            "attr_id":       idx_to_attr_id.get(i, i),
            "name":          idx_to_attr_name.get(i, f"attr_{i}"),
            "supercategory": idx_to_supercat.get(i, "unknown"),
            "confidence":    round(conf, 4),
        }
        for i, conf in results
    ]

    # Group by supercategory
    by_group = defaultdict(list)
    for a in attributes:
        by_group[a["supercategory"]].append(a)

    print(f"\n── Attributes ({len(attributes)} detected in {t_attr*1000:.0f}ms) ──")
    for group, attrs in sorted(by_group.items()):
        print(f"\n  {group.upper()}:")
        for a in attrs:
            bar = "█" * int(a["confidence"] * 20)
            print(f"    {bar:<20} {a['confidence']:.3f}  {a['name']}")

    # ── Description generation ────────────────────────────────────────────
    # Build prompt
    prompt_lines = ["Fashion attributes detected for this product:"]
    for group, attrs in sorted(by_group.items()):
        attr_strs = [f"{a['name']} ({a['confidence']:.2f})" for a in attrs]
        prompt_lines.append(f"- {group.title()}: {', '.join(attr_strs)}")
    prompt_lines += [
        "",
        "Write a 2-3 sentence product description for an online fashion store "
        "based on these attributes. Use natural retail language. "
        "Do not mention confidence scores.",
    ]
    prompt = "\n".join(prompt_lines)

    print(f"\n── Generating description (backend={llm_backend}) ──")
    t0 = time.perf_counter()

    if llm_backend == "local":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("  transformers not installed — pip install transformers accelerate")
            sys.exit(1)

        model_id  = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"  Loading {model_id} (first run downloads ~1GB) …")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm       = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype  = torch.float32,
            device_map   = "cpu",
        )
        llm.eval()

        messages = [
            {"role": "system", "content": "You are a professional fashion copywriter. Write concise 2-3 sentence product descriptions."},
            {"role": "user",   "content": prompt},
        ]
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out_ids = llm.generate(
                **inputs,
                max_new_tokens    = 220,
                temperature       = 0.7,
                do_sample         = True,
                top_p             = 0.9,
                repetition_penalty = 1.1,
                pad_token_id      = tokenizer.eos_token_id,
            )
        gen_ids     = out_ids[0][inputs["input_ids"].shape[1]:]
        description = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    elif llm_backend == "claude":
        try:
            import anthropic
        except ImportError:
            print("  anthropic not installed — pip install anthropic")
            sys.exit(1)
        if not anthropic_key:
            print("  ERROR: --anthropic-key required for claude backend")
            sys.exit(1)
        client  = anthropic.Anthropic(api_key=anthropic_key)
        message = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 220,
            system     = "You are a professional fashion copywriter. Write concise 2-3 sentence product descriptions.",
            messages   = [{"role": "user", "content": prompt}],
        )
        description = message.content[0].text.strip()
    else:
        description = "(LLM backend not configured)"

    t_llm = time.perf_counter() - t0
    print(f"\n── Description (generated in {t_llm:.1f}s) ──\n")
    print(f'  "{description}"')

    # ── Final JSON output ──────────────────────────────────────────────────
    output = {
        "attributes":          attributes,
        "attributes_by_group": {k: v for k, v in sorted(by_group.items())},
        "description":         description,
        "meta": {
            "image_size":       f"{w}x{h}",
            "crop_box":         {"x": x, "y": y, "width": int(w * f), "height": int(h * f)},
            "n_attributes":     len(attributes),
            "threshold":        threshold,
            "attr_time_ms":     round(t_attr * 1000, 1),
            "llm_time_s":       round(t_llm, 2),
        }
    }

    out_path = Path(image_path).stem + "_output.json"
    with open(out_path, "w") as f_out:
        json.dump(output, f_out, indent=2)
    print(f"\n✓ Full JSON output saved → {out_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone Fashionpedia inference test"
    )
    parser.add_argument("--image",      required=True,  help="Path to product image")
    parser.add_argument("--checkpoint", default="./fashionpedia_runs/ablation_B_stage4_head/checkpoints/best.pt")
    parser.add_argument("--processed-dir", default="./fashionpedia_processed")
    parser.add_argument("--threshold",  type=float, default=0.45)
    parser.add_argument("--top-k",      type=int,   default=20)
    parser.add_argument("--llm-backend", default="local", choices=["local", "claude"])
    parser.add_argument("--anthropic-key", default="")
    args = parser.parse_args()

    run_inference(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        processed_dir   = args.processed_dir,
        threshold       = args.threshold,
        top_k           = args.top_k,
        llm_backend     = args.llm_backend,
        anthropic_key   = args.anthropic_key,
    )
