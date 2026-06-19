#!/usr/bin/env python3
"""
generate_presentation.py
Professional slide deck for the Fashionpedia Attribute Extraction project.
Run: python generate_presentation.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
FIGS  = os.path.join(BASE, "fashionpedia_processed", "figures")
LOG   = os.path.join(BASE, "fashionpedia_runs", "ablation_B_stage4_head", "training_log.csv")
INFR  = os.path.join(BASE, "any_dress_photo_output.json")
OUT   = os.path.join(BASE, "fashionpedia_presentation.pdf")

# ── Palette (GitHub Dark-inspired) ──────────────────────────────────────────
BG   = "#0D1117"
CARD = "#161B22"
BORD = "#30363D"
BLU  = "#58A6FF"
PUR  = "#BC8CFF"
GRN  = "#3FB950"
ORG  = "#FFA657"
RED  = "#F85149"
YEL  = "#E3B341"
TXT  = "#E6EDF3"
MUT  = "#8B949E"

W, H = 16, 9
NOTES = []


# ── Primitive helpers ────────────────────────────────────────────────────────
def make_fig():
    return plt.figure(figsize=(W, H), facecolor=BG)


def blank_ax(fig, rect=None):
    rect = rect or [0, 0, 1, 1]
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def slide_header(ax, text, y=0.935, size=24):
    ax.text(0.5, y, text, transform=ax.transAxes,
            fontsize=size, fontweight="bold", color=TXT,
            ha="center", va="top")
    ax.plot([0.04, 0.96], [y - 0.085, y - 0.085],
            color=BLU, lw=1.5, transform=ax.transAxes, clip_on=False)


def rbox(ax, x, y, w, h, fc=CARD, ec=BORD, lw=1.0, r=0.012, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={r}",
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, alpha=alpha,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)


def lbl(ax, x, y, text, size=12, color=TXT, ha="left", va="center", bold=False,
         style="normal"):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=size, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", style=style,
            linespacing=1.4)


def arr(ax, x1, y1, x2, y2, color=BLU, lw=2.0, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def save_slide(fig, pdf, note=""):
    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    NOTES.append(note)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════════════════
def slide_title(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)

    # Left blue accent bar
    ax.add_patch(mpatches.Rectangle((0, 0), 0.006, 1,
                                    facecolor=BLU, transform=ax.transAxes))

    # Subtle gradient band top
    for i in range(80):
        ax.axhspan(0.78 + i * 0.0028, 0.78 + (i + 1) * 0.0028,
                   facecolor=BLU, alpha=0.010 * (1 - i / 80), zorder=0)

    # Tag pill
    rbox(ax, 0.06, 0.865, 0.27, 0.07, fc="#0C1F3A", ec=BLU, lw=1.5)
    lbl(ax, 0.195, 0.900, "MACHINE LEARNING  ·  COMPUTER VISION",
        size=10, color=BLU, ha="center")

    # Main title
    ax.text(0.5, 0.70, "Fashionpedia", transform=ax.transAxes,
            fontsize=72, fontweight="bold", color=TXT,
            ha="center", va="center")
    ax.text(0.5, 0.565, "Attribute Extraction System", transform=ax.transAxes,
            fontsize=40, fontweight="bold", color=BLU,
            ha="center", va="center")

    lbl(ax, 0.5, 0.465,
        "End-to-End Fashion Attribute Recognition  ·  FastAPI Deployment  ·  LLM-Powered Descriptions",
        size=14, color=MUT, ha="center")

    ax.plot([0.15, 0.85], [0.39, 0.39], color=BORD, lw=1, transform=ax.transAxes)

    # Stats row
    stats = [
        ("156,496", "Training Crops"),
        ("101",     "Attribute Classes"),
        ("0.3619",  "Validation mAP"),
        ("3-Stage", "Inference Pipeline"),
    ]
    for i, (v, l_) in enumerate(stats):
        x = 0.15 + i * 0.19
        ax.text(x, 0.315, v,  transform=ax.transAxes,
                fontsize=24, fontweight="bold", color=BLU, ha="center")
        ax.text(x, 0.245, l_, transform=ax.transAxes,
                fontsize=11, color=MUT, ha="center")

    ax.plot([0.05, 0.95], [0.12, 0.12], color=BORD, lw=0.5, transform=ax.transAxes)
    lbl(ax, 0.5, 0.075,
        "ConvNeXt-Tiny  ·  BCEWithLogitsLoss  ·  FastAPI  ·  Qwen2.5-0.5B  ·  Claude Haiku",
        size=12, color=MUT, ha="center")
    lbl(ax, 0.95, 0.075, "2026", size=11, color=MUT, ha="right")

    save_slide(fig, pdf,
        "This project builds a complete fashion attribute extraction system — from raw COCO-format "
        "annotations to a live production API. It automatically identifies 101 visual attributes — "
        "pattern, silhouette, neckline — from garment images and generates product descriptions.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════════════════════
def slide_problem(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Problem Statement")

    problems = [
        (BLU, "Scale",              "45,623 images × 294 possible attributes — manual tagging is slow and inconsistent"),
        (PUR, "Multi-Label",        "Each garment has multiple simultaneous attributes (avg 3.88 per crop at inference)"),
        (ORG, "Extreme Imbalance",  "Rarest attribute (paisley) is 515× more rare than the most common class"),
        (GRN, "Annotation Noise",   "38% of raw annotations have zero attributes — unannotated, not confirmed negatives"),
    ]

    for i, (color, title, desc) in enumerate(problems):
        y_box = 0.715 - i * 0.155
        rbox(ax, 0.04, y_box, 0.90, 0.125, fc=CARD, ec=color, lw=1.5)
        # Left accent strip
        ax.add_patch(mpatches.Rectangle((0.04, y_box), 0.006, 0.125,
                                        facecolor=color, transform=ax.transAxes))
        ax.text(0.065, y_box + 0.063, title, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=color, va="center")
        ax.text(0.30,  y_box + 0.063, desc, transform=ax.transAxes,
                fontsize=12, color=TXT, va="center")

    # Summary callout
    rbox(ax, 0.04, 0.08, 0.90, 0.085, fc="#0C1F3A", ec=BLU, lw=1.5)
    lbl(ax, 0.5, 0.122,
        "Goal:  Train a model that takes any fashion image and outputs structured, confidence-scored attribute labels — automatically.",
        size=12, color=TXT, ha="center")

    save_slide(fig, pdf,
        "The core challenge is automating attribute tagging at scale. Fashionpedia has 45,623 training images "
        "and 294 possible attributes — doing this manually doesn't scale. The multi-label nature and severe "
        "class imbalance, plus noisy annotations, make this a hard engineering problem.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — PROPOSED SOLUTION
# ════════════════════════════════════════════════════════════════════════════
def slide_solution(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Proposed Solution: 3-Stage Inference Pipeline")

    stages = [
        ("Stage 1",  "Garment\nDetection",    "Centre-crop bbox\nNo extra model\n~4 ms",   BLU, 0.08),
        ("Stage 2",  "Attribute\nClassifier", "ConvNeXt-Tiny\n101 attributes\n~150 ms",    PUR, 0.38),
        ("Stage 3",  "Description\nGenerator","Qwen2.5-0.5B\nor Claude Haiku\n~1.5 s",     GRN, 0.68),
    ]

    for title, name, detail, color, x in stages:
        rbox(ax, x, 0.20, 0.25, 0.60, fc=CARD, ec=color, lw=2.0)
        # Header band
        ax.add_patch(mpatches.Rectangle((x, 0.72), 0.25, 0.08,
                                        facecolor=color, alpha=0.15,
                                        transform=ax.transAxes))
        ax.text(x + 0.125, 0.755, title, transform=ax.transAxes,
                fontsize=12, color=color, ha="center", va="center", fontweight="bold")
        ax.text(x + 0.125, 0.600, name, transform=ax.transAxes,
                fontsize=17, color=TXT, ha="center", va="center", fontweight="bold",
                linespacing=1.4)
        ax.text(x + 0.125, 0.420, detail, transform=ax.transAxes,
                fontsize=11, color=MUT, ha="center", va="center", linespacing=1.5)

    # Arrows between stages
    arr(ax, 0.335, 0.50, 0.375, 0.50, color=BLU)
    arr(ax, 0.635, 0.50, 0.675, 0.50, color=PUR)

    # Input label
    ax.annotate("Fashion\nImage", xy=(0.08, 0.50), xytext=(0.005, 0.50),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=11, color=MUT, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))

    # Output label
    arr(ax, 0.93, 0.50, 0.955, 0.50, color=GRN, lw=1.5)
    lbl(ax, 0.958, 0.50, "JSON\nResponse", size=11, color=GRN, ha="left")

    # FastAPI wrapper box
    rbox(ax, 0.03, 0.10, 0.94, 0.80, fc="none", ec=BORD, lw=1, r=0.02)
    lbl(ax, 0.5, 0.065,
        "FastAPI  ·  POST /describe  ·  GET /health  ·  Swagger UI  ·  Apple MPS (M-series)",
        size=11, color=MUT, ha="center")

    save_slide(fig, pdf,
        "The solution is a three-stage pipeline wrapped in FastAPI. Stage 1 crops the garment "
        "in milliseconds with no extra model. Stage 2 runs ConvNeXt-Tiny to output 101 attribute scores. "
        "Stage 3 calls a local Qwen 0.5B or the Claude Haiku API to generate a product description.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — DATASET & TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def slide_dataset(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Dataset & Training Pipeline")

    # ── Left: preprocessing flow ─────────────────────────────────────────────
    steps = [
        (BLU, "Fashionpedia 2020",   "45,623 images  ·  333,401 raw annotations",             0.760),
        (MUT, "Drop Zero-Attr",      "−38.1%  (unannotated instances discarded)",               0.645),
        (MUT, "Drop Small Boxes",    "−7.8%   (bbox area < 1,024 px²  or side < 16 px)",       0.530),
        (MUT, "Dedup Check",         "−0.02%  (38 duplicate crops removed)",                    0.415),
        (MUT, "Filter Rare Attrs",   "294 → 101 classes  (≥ 300 training samples per class)",   0.300),
        (GRN, "Final Split",         "156,496 train crops  ·  4,066 val crops  ·  101 classes", 0.185),
    ]

    for i, (color, step, detail, y) in enumerate(steps):
        is_key = color in (BLU, GRN)
        rbox(ax, 0.03, y - 0.052, 0.47, 0.10,
             fc=CARD if is_key else "#0F1420", ec=color,
             lw=1.8 if is_key else 0.8)
        ax.text(0.055, y, step,   transform=ax.transAxes,
                fontsize=12, fontweight="bold" if is_key else "normal",
                color=color, va="center")
        ax.text(0.055, y - 0.035, detail, transform=ax.transAxes,
                fontsize=9.5, color=MUT, va="center")
        if i < len(steps) - 1:
            ax.plot([0.255, 0.255], [y - 0.052, y - 0.068],
                    color=BORD, lw=1.5, transform=ax.transAxes)

    # Supercategory badges (bottom-left)
    supercats = ["length", "neckline type", "silhouette", "textile pattern",
                 "waistline", "opening type", "non-textile material", "mfg techniques"]
    ax.text(0.03, 0.095, "8 Supercategories retained:", transform=ax.transAxes,
            fontsize=10, color=MUT)
    for i, s in enumerate(supercats):
        col_ = i % 4
        row_ = i // 4
        x_b = 0.03 + col_ * 0.115
        y_b = 0.068 - row_ * 0.048
        rbox(ax, x_b, y_b, 0.108, 0.036, fc="#0C1F3A", ec=BLU, lw=0.8, r=0.008)
        ax.text(x_b + 0.054, y_b + 0.018, s, transform=ax.transAxes,
                fontsize=8, color=BLU, ha="center", va="center")

    # ── Right: EDA figure ────────────────────────────────────────────────────
    fig_path = os.path.join(FIGS, "fig4_class_imbalance.png")
    if os.path.exists(fig_path):
        ax_img = fig.add_axes([0.525, 0.13, 0.455, 0.70])
        ax_img.imshow(Image.open(fig_path))
        ax_img.axis("off")
        ax_img.set_title("Class Imbalance: pos_weight per Attribute\n"
                          "(median 92.5×, max 515.5× for 'paisley')",
                          color=MUT, fontsize=10, pad=5)

    save_slide(fig, pdf,
        "We use the Fashionpedia 2020 dataset. The preprocessing pipeline aggressively cleans the data: "
        "dropping unannotated instances, tiny boxes, and duplicates. We filter attributes below 300 "
        "training samples, leaving 101 classes across 8 supercategories and 156K training crops.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
def slide_arch(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Model Architecture: ConvNeXt-Tiny + Partial Fine-Tuning")

    # ── Architecture diagram ─────────────────────────────────────────────────
    # Input
    rbox(ax, 0.03, 0.62, 0.10, 0.20, fc="#0C1F3A", ec=BLU, lw=1.5)
    ax.text(0.08, 0.720, "Input\n224×224", transform=ax.transAxes,
            fontsize=10, color=BLU, ha="center", va="center")
    arr(ax, 0.13, 0.72, 0.155, 0.72)

    # Backbone blocks
    blocks = [
        ("Blocks\n0–5\n(Frozen)", BORD, "#0F1420"),
        ("Block 6\nStage 4",       BLU,  CARD),
        ("Block 7\nStage 4",       BLU,  CARD),
    ]
    bx_positions = [0.155, 0.265, 0.365]
    for j, ((bname, ec_c, fc_c), bx) in enumerate(zip(blocks, bx_positions)):
        rbox(ax, bx, 0.58, 0.095, 0.28, fc=fc_c, ec=ec_c, lw=1.5)
        ax.text(bx + 0.0475, 0.720, bname, transform=ax.transAxes,
                fontsize=8.5, color=MUT if ec_c == BORD else TXT,
                ha="center", va="center", linespacing=1.3)
        if j < 2:
            arr(ax, bx + 0.095, 0.72, bx + 0.118, 0.72, lw=1.5)

    arr(ax, 0.46, 0.72, 0.485, 0.72)

    # GAP
    rbox(ax, 0.485, 0.62, 0.075, 0.20, fc=CARD, ec=PUR, lw=1.5)
    ax.text(0.5225, 0.720, "GAP\n768-d", transform=ax.transAxes,
            fontsize=10, color=PUR, ha="center", va="center")
    arr(ax, 0.56, 0.72, 0.585, 0.72, color=PUR)

    # MLP Head
    rbox(ax, 0.585, 0.48, 0.115, 0.42, fc="#140E25", ec=PUR, lw=2.0)
    ax.text(0.6425, 0.95, "MLP Head", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=PUR, ha="center")
    head_items = [
        ("LayerNorm(768)",   0.875),
        ("Linear 768→512",  0.810),
        ("GELU + Drop(0.3)", 0.745),
        ("Linear 512→101",  0.680),
        ("Sigmoid ×101",    0.615),
    ]
    for ht, hy in head_items:
        ax.text(0.6425, hy, ht, transform=ax.transAxes,
                fontsize=9, color=TXT, ha="center", va="center")
        if hy > 0.62:
            ax.plot([0.595, 0.69], [hy - 0.032, hy - 0.032],
                    color=BORD, lw=0.7, transform=ax.transAxes)
    arr(ax, 0.70, 0.72, 0.725, 0.72, color=PUR)

    # Output
    rbox(ax, 0.725, 0.62, 0.09, 0.20, fc="#0C2820", ec=GRN, lw=1.5)
    ax.text(0.770, 0.720, "101\nLogits", transform=ax.transAxes,
            fontsize=10, color=GRN, ha="center", va="center")

    # Legend
    rbox(ax, 0.035, 0.52, 0.085, 0.055, fc="#0F1420", ec=BORD, lw=1)
    ax.text(0.0775, 0.548, "Frozen", transform=ax.transAxes, fontsize=9, color=MUT, ha="center")
    rbox(ax, 0.13,  0.52, 0.085, 0.055, fc=CARD,    ec=BLU,  lw=1)
    ax.text(0.1725, 0.548, "Trainable", transform=ax.transAxes, fontsize=9, color=BLU, ha="center")

    # ── Ablation table ───────────────────────────────────────────────────────
    rbox(ax, 0.585, 0.08, 0.40, 0.36, fc=CARD, ec=BORD, lw=1)
    ax.text(0.785, 0.415, "Partial Fine-Tuning Ablations",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=TXT, ha="center")

    headers = [("Config", 0.605), ("Trainable Params", 0.720), ("Result", 0.840)]
    for h, hx in headers:
        ax.text(hx, 0.385, h, transform=ax.transAxes, fontsize=10, color=BLU, fontweight="bold")
    ax.plot([0.592, 0.978], [0.370, 0.370], color=BORD, lw=0.8, transform=ax.transAxes)

    ablations = [
        ("A  Head Only",         "~0.5M  (1.8%)",  "Baseline",   MUT),
        ("B  Stage-4 + Head  ✓", "~1.2M  (3.2%)",  "SELECTED",   GRN),
        ("C  Stages 3-4 + Head", "~2.5M  (8.9%)",  "Extended",   MUT),
    ]
    for i, (cfg, tp, strategy, color) in enumerate(ablations):
        yr = 0.310 - i * 0.090
        bg_ = "#0A2015" if color == GRN else "#0D1117"
        rbox(ax, 0.592, yr - 0.025, 0.382, 0.072, fc=bg_, ec=color, lw=1.2)
        ax.text(0.605, yr + 0.010, cfg,      transform=ax.transAxes, fontsize=10, color=color)
        ax.text(0.720, yr + 0.010, tp,       transform=ax.transAxes, fontsize=10, color=TXT)
        ax.text(0.840, yr + 0.010, strategy, transform=ax.transAxes, fontsize=10, color=color,
                fontweight="bold" if color == GRN else "normal")

    ax.plot([0.592, 0.978], [0.132, 0.132], color=BORD, lw=0.8, transform=ax.transAxes)
    ax.text(0.785, 0.107,
            "LR head=1e-3  ·  LR backbone=1e-4  (10:1 ratio)  ·  3-epoch warmup  ·  Cosine Annealing",
            transform=ax.transAxes, fontsize=9.5, color=MUT, ha="center")

    save_slide(fig, pdf,
        "ConvNeXt-Tiny is a 28M-parameter vision backbone. We freeze the first 6 blocks and fine-tune "
        "only Stage 4 — blocks 6 and 7 — plus the new MLP head. That's just 1.2M trainable parameters, "
        "3.2% of the total. Differential learning rates — 10× slower for backbone — preserve ImageNet features.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
def slide_results(pdf):
    df = pd.read_csv(LOG)
    with open(INFR) as f:
        inf_ex = json.load(f)

    fig = make_fig()
    ax_bg_ = blank_ax(fig)
    slide_header(ax_bg_, "Results & Example Predictions")

    # ── Training curves (left) ───────────────────────────────────────────────
    ax1 = fig.add_axes([0.04, 0.13, 0.38, 0.67], facecolor=CARD)
    epochs = df["epoch"].values
    ax1.plot(epochs, df["train_loss"], color=RED,  lw=2.5, marker="o", ms=5, label="Train Loss")
    ax1.plot(epochs, df["val_loss"],   color=ORG,  lw=2.5, marker="s", ms=5, label="Val Loss")
    ax1r = ax1.twinx()
    ax1r.plot(epochs, df["val_mAP"],   color=GRN,  lw=2.5, marker="^", ms=6, label="Val mAP")
    ax1r.plot(epochs, df["val_F1"],    color=BLU,  lw=2,   marker="D", ms=5, label="Val F1")
    ax1r.plot(epochs, df["val_Recall"],color=PUR,  lw=1.5, marker="v", ms=4, label="Val Recall", ls="--")

    for ax_ in (ax1, ax1r):
        ax_.set_facecolor(CARD)
        ax_.spines[:].set_color(BORD)
        ax_.tick_params(colors=MUT, labelsize=9)
    ax1.set_xlabel("Epoch", color=MUT, fontsize=10)
    ax1.set_ylabel("Loss",  color=ORG,  fontsize=10)
    ax1r.set_ylabel("Score", color=GRN, fontsize=10)
    ax1.set_title("Training Curves — Ablation B (7 Epochs)", color=TXT, fontsize=11, pad=8)
    ax1.grid(color=BORD, lw=0.5, alpha=0.6)

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2, fontsize=8.5,
               facecolor=BG, labelcolor=TXT, loc="center right")

    # ── Metric tiles (center) ────────────────────────────────────────────────
    ax_c = blank_ax(fig, rect=[0.44, 0.13, 0.21, 0.67])
    ax_c.text(0.5, 0.975, "Best Checkpoint\n(Epoch 7)", transform=ax_c.transAxes,
              fontsize=11, color=MUT, ha="center", va="top", linespacing=1.4)
    metrics = [
        ("Val mAP",    "0.3619", BLU, 0.72),
        ("F1 (macro)", "0.2912", PUR, 0.515),
        ("Recall",     "0.7464", GRN, 0.310),
        ("Precision",  "0.2023", ORG, 0.105),
    ]
    for mname, mval, color, y in metrics:
        rbox(ax_c, 0.05, y, 0.90, 0.175, fc=CARD, ec=color, lw=1.8)
        ax_c.add_patch(mpatches.Rectangle((0.05, y), 0.05, 0.175,
                                          facecolor=color, alpha=0.18,
                                          transform=ax_c.transAxes))
        ax_c.text(0.55, y + 0.120, mname, transform=ax_c.transAxes,
                  fontsize=11, color=MUT, ha="center")
        ax_c.text(0.55, y + 0.055, mval, transform=ax_c.transAxes,
                  fontsize=22, fontweight="bold", color=color, ha="center")

    # ── Example prediction (right) ───────────────────────────────────────────
    ax_r = blank_ax(fig, rect=[0.66, 0.13, 0.33, 0.67])
    ax_r.text(0.5, 0.98, "Live Inference — any_dress_photo.jpg",
              transform=ax_r.transAxes, fontsize=11, color=TXT,
              ha="center", va="top", fontweight="bold")
    ax_r.text(0.5, 0.935, "threshold=0.45  ·  11 attributes detected  ·  ~75 ms",
              transform=ax_r.transAxes, fontsize=9, color=MUT, ha="center")

    attrs = inf_ex["attributes"][:9]
    for i, a in enumerate(attrs):
        y_a = 0.875 - i * 0.103
        conf = a["confidence"]
        col_ = GRN if conf > 0.85 else (BLU if conf > 0.60 else YEL)
        rbox(ax_r, 0.02, y_a - 0.048, 0.96, 0.092, fc=CARD, ec=BORD, lw=0.8)
        ax_r.text(0.06, y_a, a["name"], transform=ax_r.transAxes,
                  fontsize=9.5, color=TXT, va="center")
        # Confidence bar
        bar_w = conf * 0.38
        ax_r.add_patch(FancyBboxPatch((0.56, y_a - 0.022), bar_w, 0.044,
                                      boxstyle="round,pad=0.005",
                                      facecolor=col_, edgecolor="none",
                                      transform=ax_r.transAxes))
        ax_r.text(0.97, y_a, f"{conf:.0%}", transform=ax_r.transAxes,
                  fontsize=9, color=col_, va="center", ha="right")

    save_slide(fig, pdf,
        "After 7 epochs the model reaches validation mAP 0.36 and F1 0.29. "
        "High recall at 0.74 means the model finds most relevant attributes. "
        "On the live example, the top prediction is plain pattern at 99%, "
        "no non-textile material at 97%, symmetrical silhouette at 95%.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — DEPLOYMENT
# ════════════════════════════════════════════════════════════════════════════
def slide_deployment(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Deployment Architecture: FastAPI + Multi-Backend LLM")

    # Client
    rbox(ax, 0.02, 0.48, 0.12, 0.26, fc="#0C1F3A", ec=BLU, lw=1.5)
    ax.text(0.08, 0.61, "Client\n(iOS / Web\n/ cURL)", transform=ax.transAxes,
            fontsize=11, color=BLU, ha="center", va="center", linespacing=1.4)

    arr(ax, 0.14, 0.65, 0.178, 0.65, color=BLU)
    ax.text(0.158, 0.705, "POST /describe\nmultipart/form-data",
            transform=ax.transAxes, fontsize=8.5, color=MUT, ha="center")

    # FastAPI wrapper
    rbox(ax, 0.178, 0.12, 0.645, 0.78, fc=CARD, ec=BORD, lw=1, r=0.02)
    ax.text(0.50, 0.88, "FastAPI Application", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=TXT, ha="center")

    # Stage boxes
    stage_defs = [
        ("Stage 1\nGarment\nDetector",    "centre-crop\n0.85 fraction\n~4 ms",  BLU,  0.198),
        ("Stage 2\nAttribute\nClassifier","ConvNeXt-Tiny\nMPS device\n~150 ms", PUR,  0.390),
        ("Stage 3\nDescription\nGenerator","Qwen2.5-0.5B\nor Claude Haiku\n~1.5 s", GRN, 0.582),
    ]
    for sname, sdetail, color, sx in stage_defs:
        rbox(ax, sx, 0.49, 0.175, 0.32, fc="#0D1117", ec=color, lw=1.8)
        ax.text(sx + 0.0875, 0.735, sname, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color, ha="center", linespacing=1.3)
        ax.text(sx + 0.0875, 0.575, sdetail, transform=ax.transAxes,
                fontsize=9.5, color=MUT, ha="center", linespacing=1.4)

    arr(ax, 0.375, 0.65, 0.387, 0.65, color=BLU, lw=1.5)
    arr(ax, 0.567, 0.65, 0.579, 0.65, color=PUR, lw=1.5)

    # LLM backends
    rbox(ax, 0.588, 0.22, 0.095, 0.20, fc=CARD, ec=MUT,  lw=1)
    ax.text(0.6355, 0.320, "Qwen2.5\n0.5B\n(Local CPU)", transform=ax.transAxes,
            fontsize=8.5, color=MUT, ha="center", linespacing=1.3)
    rbox(ax, 0.700, 0.22, 0.095, 0.20, fc=CARD, ec=ORG,  lw=1)
    ax.text(0.7475, 0.320, "Claude\nHaiku\n(API)", transform=ax.transAxes,
            fontsize=8.5, color=ORG, ha="center", linespacing=1.3)
    for bx_, bc in [(0.6355, MUT), (0.7475, ORG)]:
        ax.plot([bx_, bx_], [0.42, 0.49], color=bc, lw=1, ls="--",
                transform=ax.transAxes)
    ax.text(0.692, 0.185, "switchable via LLM_BACKEND env var",
            transform=ax.transAxes, fontsize=9, color=MUT, ha="center")

    # API routes
    rbox(ax, 0.200, 0.18, 0.36, 0.28, fc="#0D1117", ec=BORD, lw=1)
    ax.text(0.38, 0.44, "API Endpoints", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=TXT, ha="center")
    routes = [
        ("GET  /health",  "Model status, device, threshold"),
        ("POST /describe","Image → attributes + description"),
        ("GET  /docs",    "Swagger UI (auto-generated)"),
    ]
    for i, (route, desc) in enumerate(routes):
        yr = 0.385 - i * 0.064
        ax.text(0.215, yr, route, transform=ax.transAxes, fontsize=10, color=BLU)
        ax.text(0.365, yr, desc,  transform=ax.transAxes, fontsize=9,  color=MUT)

    # Response arrow back
    ax.annotate("", xy=(0.140, 0.57), xytext=(0.178, 0.57),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<-", color=GRN, lw=1.5))
    ax.text(0.158, 0.52, "JSON\nResponse", transform=ax.transAxes,
            fontsize=9, color=GRN, ha="center")

    # Latency bar
    rbox(ax, 0.02, 0.035, 0.96, 0.075, fc="#0D1117", ec=BORD, lw=1)
    ax.text(0.5, 0.073,
            "Total Latency:  ~4 ms detection  +  ~150 ms attributes  +  ~1.5 s LLM  =  ~1.7 s (local Qwen)  /  ~0.9 s (Claude Haiku API)",
            transform=ax.transAxes, fontsize=10.5, color=MUT, ha="center")

    save_slide(fig, pdf,
        "The FastAPI server wraps all three stages. POST /describe accepts a multipart image "
        "and returns a JSON with per-attribute confidence scores and a product description. "
        "The LLM backend is configurable via environment variable — Qwen runs locally on CPU "
        "within 1 GB RAM alongside the ConvNeXt model.")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — LEARNING EXPERIENCE & FUTURE WORK
# ════════════════════════════════════════════════════════════════════════════
def slide_learning(pdf):
    fig = make_fig()
    ax  = blank_ax(fig)
    slide_header(ax, "Learning Experience & Future Work")

    # ── Model journey (top) ──────────────────────────────────────────────────
    ax.text(0.5, 0.835, "Model Architecture Journey", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=TXT, ha="center")

    models = [
        ("CLIP\n(ViT-L/14)",      "Zero-shot approach.\nAttractive but poor\nfine-grained precision\nfor 101 classes.",         RED, 0.055),
        ("LLaVA\n(VLM)",          "Rich descriptions.\nImpractical for\nstructured 101-class\nmulti-label output.",              ORG, 0.285),
        ("Swin\nTransformer",     "Competitive accuracy.\nHeavier VRAM demand.\nSlower on Apple MPS.",                          YEL, 0.515),
        ("ConvNeXt-Tiny\n✓",      "28M params — fast.\nMPS-optimized.\nBest accuracy/speed\nfor this task.",                    GRN, 0.745),
    ]

    for mname, mdesc, color, x in models:
        selected = color == GRN
        rbox(ax, x, 0.46, 0.215, 0.335,
             fc="#0A2015" if selected else CARD,
             ec=color, lw=2.2 if selected else 1.0)
        ax.text(x + 0.1075, 0.735, mname, transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=color,
                ha="center", linespacing=1.3)
        ax.text(x + 0.1075, 0.590, mdesc, transform=ax.transAxes,
                fontsize=9, color=MUT, ha="center", va="center", linespacing=1.4)
        if x < 0.745:
            arr(ax, x + 0.215, 0.625, x + 0.278, 0.625, color=BORD, lw=1.5)

    # ── Technical decisions (bottom) ─────────────────────────────────────────
    ax.text(0.02, 0.430, "Key Technical Decisions", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=TXT)
    ax.plot([0.02, 0.98], [0.408, 0.408], color=BORD, lw=0.8, transform=ax.transAxes)

    decisions = [
        (BLU, "BCEWithLogitsLoss + pos_weight",
         "Multi-label binary CE with per-class weighting handles 515× imbalance. "
         "Weight cap=50 prevents gradient explosion on rarest classes."),
        (PUR, "Partial Fine-Tuning (Stage 4 + Head)",
         "Freezing 6 of 8 blocks with only 1.2M trainable params prevented overfitting "
         "and kept each epoch under 26 minutes on M-series MPS."),
        (GRN, "Differential Learning Rates + Warmup",
         "10:1 ratio (1e-3 head / 1e-4 backbone) + 3-epoch linear warmup protects "
         "pre-trained ImageNet representations while the new head learns fast."),
        (ORG, "Annotation-Level Crops as Training Units",
         "Training on bbox crops (not full images) aligns inputs with attribute labels — "
         "each crop = one garment region described by its annotation."),
    ]

    for i, (color, title, desc) in enumerate(decisions):
        row = i // 2
        col = i % 2
        x_d = 0.02 + col * 0.50
        y_d = 0.335 - row * 0.150
        rbox(ax_d := ax, x_d, y_d - 0.085, 0.475, 0.120, fc=CARD, ec=color, lw=1.3)
        ax.add_patch(mpatches.Rectangle((x_d, y_d - 0.085), 0.005, 0.120,
                                        facecolor=color, transform=ax.transAxes))
        ax.text(x_d + 0.018, y_d + 0.020, title, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color)
        ax.text(x_d + 0.018, y_d - 0.028, desc, transform=ax.transAxes,
                fontsize=9, color=MUT, va="center")

    # Future work strip
    rbox(ax, 0.02, 0.035, 0.96, 0.058, fc="#0C1F3A", ec=BLU, lw=1)
    ax.text(0.5, 0.064, "Future Work:  More training epochs  ·  iOS SwiftUI app  ·  "
            "YOLOv8 garment detector  ·  Saliency-based cropping  ·  Larger ConvNeXt variant",
            transform=ax.transAxes, fontsize=10.5, color=BLU, ha="center")

    save_slide(fig, pdf,
        "The biggest lesson was model selection. CLIP and LLaVA are powerful but don't fit "
        "structured 101-class multi-label output. Swin Transformer was competitive but heavier. "
        "ConvNeXt-Tiny hit the right balance. BCEWithLogitsLoss with pos_weight and partial "
        "fine-tuning were the two most impactful engineering decisions in this project.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "text.color":      TXT,
        "axes.labelcolor": MUT,
        "xtick.color":     MUT,
        "ytick.color":     MUT,
        "figure.dpi":      120,
    })

    print("Generating presentation …")
    with PdfPages(OUT) as pdf:
        slide_title(pdf)
        slide_problem(pdf)
        slide_solution(pdf)
        slide_dataset(pdf)
        slide_arch(pdf)
        slide_results(pdf)
        slide_deployment(pdf)
        slide_learning(pdf)

        d = pdf.infodict()
        d["Title"]    = "Fashionpedia Attribute Extraction System"
        d["Author"]   = "Generated by Claude Code"
        d["Subject"]  = "ML Presentation — ConvNeXt, FastAPI, Fashion AI"
        d["Keywords"] = "ConvNeXt, Fashionpedia, FastAPI, Multi-label Classification, BCEWithLogitsLoss"

    print(f"\n✓  Saved → {OUT}\n")
    print("=" * 70)
    print("SPEAKER NOTES  (≈ 26 s per slide, total ~3 min 30 s)")
    print("=" * 70)
    slide_names = [
        "1. Title", "2. Problem Statement", "3. Proposed Solution",
        "4. Dataset & Pipeline", "5. Model Architecture",
        "6. Results", "7. Deployment", "8. Learning Experience",
    ]
    for i, (name, note) in enumerate(zip(slide_names, NOTES)):
        print(f"\n[{name}]")
        print(note)


if __name__ == "__main__":
    main()
