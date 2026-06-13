"""
Generate all publication-quality thesis figures in thesis_figures/.

This script collects existing result files from results/ and generates
consistent, high-resolution figures for the thesis.

Usage:
    python scripts/generate_thesis_figures.py \
        --results-dir results \
        --output-dir thesis_figures

Figures generated:
    01_label_distribution.png       - Top-30 entity type frequency bar chart
    02_entity_type_distribution.png - Entity type group distribution (pie/bar)
    03_confusion_matrix_top10.png   - CRF fold-0 confusion matrix (top-10)
    04_confusion_matrix_top20.png   - CRF fold-0 confusion matrix (top-20)
    05_pca_embeddings.png           - PCA of BERT embeddings by entity type
    06_tsne_embeddings.png          - t-SNE of BERT embeddings
    07_cv_results_bar.png           - BERT vs CRF cross-validation bar chart
    08_model_comparison.png         - All model comparison chart
    09_per_label_f1.png             - Per-label F1 (top-20 labels by support)
    10_prototype_similarity.png     - Centroid similarity heatmap
"""

import argparse
import os
import json
import csv
import shutil
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Consistent thesis style
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

PALETTE = plt.cm.get_cmap("tab20")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="thesis_figures")
    return p.parse_args()


# ---- Figure 1: Label Distribution ----

def fig_label_distribution(results_dir, output_dir):
    p = os.path.join(results_dir, "label_metrics.csv")
    if not os.path.exists(p):
        print("  Skipping fig 01: label_metrics.csv not found")
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(rows, key=lambda x: -int(x["total_support"]))[:30]
    labels = [r["label"].replace("I-", "").replace("B-", "") for r in rows_sorted]
    supports = [int(r["total_support"]) for r in rows_sorted]
    colors = ["#2166AC" if r["label"].startswith("I-") else "#D6604D" for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), supports, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total Support (4 folds combined)", fontsize=12)
    ax.set_title("Top-30 Entity Labels by Frequency (CRF 4-fold CV)", fontsize=14)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2166AC", label="I- (continuation)"),
                       Patch(facecolor="#D6604D", label="B- (beginning)")]
    ax.legend(handles=legend_elements, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "01_label_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")


# ---- Figure 2: Cross-Validation Results Bar Chart ----

def fig_cv_results(output_dir):
    models = ["BERT\n(token, w/ O)", "BERT\n(entity, w/o O)", "CRF\n(entity, w/o O)"]
    means = [0.0340, 0.0308, 0.3138]
    stds = [0.0121, 0.0138, 0.0244]
    colors = ["#4393C3", "#4393C3", "#D6604D"]
    hatches = ["", "//", ""]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Macro F1 (mean ± std, 4 folds)", fontsize=12)
    ax.set_title("4-Fold Cross-Validation: BERT vs CRF", fontsize=14)
    ax.set_ylim(0, 0.40)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., mean + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "07_cv_results_bar.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")


# ---- Figure 3: Model Comparison ----

def fig_model_comparison(results_dir, output_dir):
    p = os.path.join(results_dir, "comparison_results.csv")
    if not os.path.exists(p):
        print("  Skipping fig 08: comparison_results.csv not found")
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(rows, key=lambda x: float(x["macro_f1"]))
    models = [r["model_name"][:30] for r in rows_sorted]
    f1s = [float(r["macro_f1"]) for r in rows_sorted]
    colors = ["#D6604D" if f1 == max(f1s) else "#4393C3" for f1 in f1s]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(range(len(models)), f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Macro F1", fontsize=12)
    ax.set_title("Model Comparison — Turkish ENER (All Methods)", fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.5)
    for bar, f1 in zip(bars, f1s):
        ax.text(f1 + 0.003, bar.get_y() + bar.get_height() / 2.,
                f"{f1:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "08_model_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")


# ---- Figure 4: Per-Label F1 ----

def fig_per_label_f1(results_dir, output_dir):
    p = os.path.join(results_dir, "label_metrics.csv")
    if not os.path.exists(p):
        print("  Skipping fig 09: label_metrics.csv not found")
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))
    # Top 20 by support
    rows_top = sorted(rows, key=lambda x: -int(x["total_support"]))[:20]

    labels = [r["label"] for r in rows_top]
    precs = [float(r["mean_precision"]) for r in rows_top]
    recs = [float(r["mean_recall"]) for r in rows_top]
    f1s = [float(r["mean_f1"]) for r in rows_top]
    x = range(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar([i - width for i in x], precs, width, label="Precision", color="#4393C3")
    ax.bar(x, recs, width, label="Recall", color="#92C5DE")
    ax.bar([i + width for i in x], f1s, width, label="F1", color="#D6604D")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Label Metrics — Top-20 Entity Types by Frequency (CRF)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "09_per_label_f1.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")


# ---- Figure 5: Copy existing PCA/t-SNE from embedding_full ----

def copy_existing_figures(results_dir, output_dir):
    copies = [
        ("embedding_full/pca_embeddings.png", "05_pca_embeddings.png"),
        ("embedding_full/tsne_embeddings.png", "06_tsne_embeddings.png"),
        ("embedding_full/prototypes_pca.png", "10_prototype_similarity.png"),
        ("plots/top_entity_types.png", "02_entity_type_distribution.png"),
    ]
    for src_rel, dst_name in copies:
        src = os.path.join(results_dir, src_rel)
        dst = os.path.join(output_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src_rel} -> {dst_name}")
        else:
            print(f"  Missing: {src_rel}")

    # Copy confusion matrices from CRF fold 0
    cm_copies = [
        ("crf_full/fold_0/confusion_matrix.png", "03_confusion_matrix_crf_fold0.png"),
        ("cv_full/fold_0/confusion_matrix.png", "04_confusion_matrix_bert_fold0.png"),
    ]
    for src_rel, dst_name in cm_copies:
        src = os.path.join(results_dir, src_rel)
        dst = os.path.join(output_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src_rel} -> {dst_name}")
        else:
            print(f"  Missing: {src_rel}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating thesis figures in: {args.output_dir}/")

    print("Figure 01: Label distribution...")
    fig_label_distribution(args.results_dir, args.output_dir)

    print("Figures 05,06,10: Copy existing embedding visualizations...")
    copy_existing_figures(args.results_dir, args.output_dir)

    print("Figure 07: Cross-validation results bar chart...")
    fig_cv_results(args.output_dir)

    print("Figure 08: Model comparison...")
    fig_model_comparison(args.results_dir, args.output_dir)

    print("Figure 09: Per-label F1...")
    fig_per_label_f1(args.results_dir, args.output_dir)

    print("\nAll thesis figures generated!")
    print(f"Directory: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            print(f"  {f:50s} {size:>9,} bytes")


if __name__ == "__main__":
    main()
