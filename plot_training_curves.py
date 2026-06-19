#!/usr/bin/env python3
# =============================================================================
#  FASHIONPEDIA — Post-Training Analysis & Visualisation
#
#  Reads training_log.csv and threshold_sweep.csv produced by train.py
#  and generates publication-quality figures.
#
#  Usage:
#      python plot_training_curves.py --run-dir ./fashionpedia_runs/run_xxx
# =============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

COLORS = {
    "train": "#3498db",
    "val":   "#e74c3c",
    "mAP":   "#2ecc71",
    "F1":    "#9b59b6",
    "P":     "#f39c12",
    "R":     "#1abc9c",
}


def load_run(run_dir: Path):
    """Load all artefacts from a training run directory."""
    log_path    = run_dir / "training_log.csv"
    thresh_path = run_dir / "threshold_sweep.csv"
    cfg_path    = run_dir / "config.json"

    assert log_path.exists(), f"training_log.csv not found in {run_dir}"

    log_df    = pd.read_csv(log_path)
    thresh_df = pd.read_csv(thresh_path) if thresh_path.exists() else None
    cfg       = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    return log_df, thresh_df, cfg


def plot_loss_curves(log_df: pd.DataFrame, out_dir: Path) -> None:
    """Train and val loss curves."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss",
            color=COLORS["train"], lw=2)
    ax.plot(log_df["epoch"], log_df["val_loss"],   label="Val Loss",
            color=COLORS["val"],   lw=2)
    best_epoch = log_df.loc[log_df["val_mAP"].idxmax(), "epoch"]
    ax.axvline(best_epoch, ls="--", color="grey", alpha=0.6,
               label=f"Best epoch ({int(best_epoch)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: loss_curves.png")


def plot_metric_curves(log_df: pd.DataFrame, out_dir: Path) -> None:
    """Val mAP, F1, Precision, Recall over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics_axes = [
        ("val_mAP",       "Val mAP",       COLORS["mAP"],   axes[0, 0]),
        ("val_F1",        "Val F1 (macro)", COLORS["F1"],    axes[0, 1]),
        ("val_Precision", "Val Precision",  COLORS["P"],     axes[1, 0]),
        ("val_Recall",    "Val Recall",     COLORS["R"],     axes[1, 1]),
    ]
    best_epoch = log_df.loc[log_df["val_mAP"].idxmax(), "epoch"]

    for col, label, color, ax in metrics_axes:
        if col not in log_df.columns:
            continue
        ax.plot(log_df["epoch"], log_df[col], color=color, lw=2, label=label)
        best_val = log_df.loc[log_df["val_mAP"].idxmax(), col]
        ax.axvline(best_epoch, ls="--", color="grey", alpha=0.5)
        ax.scatter([best_epoch], [best_val], color=color, zorder=5,
                   s=80, label=f"Best: {best_val:.4f}")
        ax.set_ylabel(label)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle("Validation Metrics Over Training", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "metric_curves.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: metric_curves.png")


def plot_lr_curve(log_df: pd.DataFrame, out_dir: Path) -> None:
    """Learning rate schedule visualisation."""
    if "lr_head" not in log_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(log_df["epoch"], log_df["lr_head"],
            label="LR (head)", color=COLORS["train"], lw=2)
    if "lr_backbone" in log_df.columns:
        ax.plot(log_df["epoch"], log_df["lr_backbone"],
                label="LR (backbone)", color=COLORS["val"], lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule (log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lr_schedule.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: lr_schedule.png")


def plot_threshold_sweep(thresh_df: pd.DataFrame, out_dir: Path) -> None:
    """F1, Precision, Recall across thresholds."""
    if thresh_df is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresh_df["threshold"], thresh_df["F1"],
            label="F1",        color=COLORS["F1"], lw=2, marker="o")
    ax.plot(thresh_df["threshold"], thresh_df["Precision"],
            label="Precision", color=COLORS["P"],  lw=2, marker="s")
    ax.plot(thresh_df["threshold"], thresh_df["Recall"],
            label="Recall",    color=COLORS["R"],  lw=2, marker="^")

    best_thr = thresh_df.loc[thresh_df["F1"].idxmax(), "threshold"]
    ax.axvline(best_thr, ls="--", color="grey", alpha=0.7,
               label=f"Best F1 @ thr={best_thr:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score (macro)")
    ax.set_title("Threshold Sweep — Best Model Checkpoint")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: threshold_sweep.png")


def print_summary_table(log_df: pd.DataFrame, thresh_df: pd.DataFrame) -> None:
    """Print a tidy summary to stdout."""
    best_row = log_df.loc[log_df["val_mAP"].idxmax()]
    print("\n" + "=" * 60)
    print(" TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Best epoch      : {int(best_row['epoch'])}")
    print(f"  val mAP         : {best_row['val_mAP']:.4f}")
    print(f"  val F1 (macro)  : {best_row['val_F1']:.4f}")
    print(f"  val Precision   : {best_row['val_Precision']:.4f}")
    print(f"  val Recall      : {best_row['val_Recall']:.4f}")
    print(f"  val Loss        : {best_row['val_loss']:.4f}")
    print(f"  train Loss      : {best_row['train_loss']:.4f}")

    if thresh_df is not None:
        best_thr_row = thresh_df.loc[thresh_df["F1"].idxmax()]
        print(f"\n  Best threshold  : {best_thr_row['threshold']:.2f}")
        print(f"  F1 @ best thr   : {best_thr_row['F1']:.4f}")
        print(f"  P  @ best thr   : {best_thr_row['Precision']:.4f}")
        print(f"  R  @ best thr   : {best_thr_row['Recall']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to run directory (contains training_log.csv)"
    )
    args    = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    log_df, thresh_df, cfg = load_run(run_dir)

    print(f"Loaded {len(log_df)} epochs from {run_dir.name}")
    plot_loss_curves(log_df, out_dir)
    plot_metric_curves(log_df, out_dir)
    plot_lr_curve(log_df, out_dir)
    plot_threshold_sweep(thresh_df, out_dir)
    print_summary_table(log_df, thresh_df)
    print(f"\nAll figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
