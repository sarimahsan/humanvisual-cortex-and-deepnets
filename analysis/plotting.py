"""
Static Figure Generation for Visual Cortex Brain Encoding.

Produces publication-quality figures using pure Matplotlib:
1. Fig 1: Layer x ROI Heatmap per model.
2. Fig 2: Model comparison per ROI with untrained & Gabor controls.
3. Fig 3: Hierarchy analysis (Best-layer depth vs. anatomical ROI order with Spearman rho).
4. Fig 4: Domain selectivity profiles in FFA, PPA, and EBA (Measured vs. Predicted).
5. Fig 5: Multi-subject robustness across Subjects 1-4.
"""

from typing import Dict, List, Optional, Any
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data.rois import ROI_PALETTE, ROI_HIERARCHY_RANK

# Set clean aesthetic defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_layer_roi_heatmap(
    summary_data: Dict[str, Any],
    model_key: str,
    output_path: str = "figures/fig1_layer_roi_heatmap.png",
) -> None:
    """Generates a 2D heatmap showing prediction accuracy (Pearson r) across layers and ROIs."""
    entries = [v for v in summary_data.values() if v.get("model_key") == model_key]
    if not entries:
        print(f"No entries found for {model_key}. Skipping heatmap.")
        return

    # Sort entries by normalized depth
    entries.sort(key=lambda x: x.get("normalized_depth", 0.0))

    layer_labels = [f"d={e.get('normalized_depth', 0.0):.2f}\n{e.get('layer_name', '')}" for e in entries]
    rois = [r for r in ROI_HIERARCHY_RANK.keys() if any(r in e.get("roi_medians", {}) for e in entries)]
    if not rois:
        # Fallback to any present ROI keys
        all_keys = set()
        for e in entries:
            all_keys.update(e.get("roi_medians", {}).keys())
        rois = sorted(list(all_keys))

    if not rois:
        print(f"Notice: No ROI-specific metrics found for {model_key}. Skipping heatmap.")
        return

    matrix = np.zeros((len(rois), len(entries)))
    for j, e in enumerate(entries):
        for i, r in enumerate(rois):
            matrix[i, j] = e.get("roi_medians", {}).get(r, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = max(0.6, float(matrix.max())) if matrix.size > 0 else 0.6
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_yticks(np.arange(len(rois)))
    ax.set_xticklabels(layer_labels, rotation=35, ha="right")
    ax.set_yticklabels(rois)

    # Annotate numeric values inside cells
    for i in range(len(rois)):
        for j in range(len(entries)):
            val = matrix[i, j]
            color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Held-Out Test Pearson $r$")

    ax.set_title(f"Representational Encoding Alignment: {model_key.upper()}")
    ax.set_xlabel("Layer Depth (Normalized [0, 1])")
    ax.set_ylabel("Cortical Region of Interest (Early -> High)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison_per_roi(
    model_best_scores: Dict[str, Dict[str, float]],
    output_path: str = "figures/fig2_model_comparison.png",
) -> None:
    """Plots grouped bar chart comparing peak performance of each model and control per ROI."""
    rois = [r for r in ROI_HIERARCHY_RANK.keys() if any(r in sc for sc in model_best_scores.values())]
    models = list(model_best_scores.keys())

    x = np.arange(len(rois))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    default_colors = ["#4299e1", "#48bb78", "#ed8936", "#9f7aea", "#e53e3e", "#a0aec0", "#718096"]

    for i, m_name in enumerate(models):
        y_vals = [model_best_scores[m_name].get(r, 0.0) for r in rois]
        hatch = "//" if "untrained" in m_name.lower() or "gabor" in m_name.lower() else ""
        color = default_colors[i % len(default_colors)]
        ax.bar(x + i * width, y_vals, width, label=m_name, color=color, hatch=hatch, alpha=0.9)

    ax.set_ylabel("Peak Prediction Accuracy (Pearson $r$)")
    ax.set_title("Peak Encoding Accuracy Across Human Visual ROIs")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(rois)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hierarchy_regression(
    hierarchy_results: Dict[str, Dict[str, Any]],
    output_path: str = "figures/fig3_hierarchy_alignment.png",
) -> None:
    """Plots Best-layer depth vs. Anatomical ROI order for key models, with Spearman rho."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for m_key, h_res in hierarchy_results.items():
        ranks = np.array(h_res["ranks"])
        depths = np.array(h_res["depths"])
        rho = h_res.get("spearman_rho", 0.0)
        ci = h_res.get("ci_95", (0.0, 0.0))

        ax.scatter(ranks, depths, s=60, alpha=0.8, label=f"{m_key} ($\\rho = {rho:.2f}$, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}])")

        if len(ranks) > 1:
            z = np.polyfit(ranks, depths, 1)
            p = np.poly1d(z)
            x_line = np.linspace(ranks.min(), ranks.max(), 100)
            ax.plot(x_line, p(x_line), linestyle="--", alpha=0.7)

    sample_rois = list(hierarchy_results.values())[0]["rois"]
    sample_ranks = list(hierarchy_results.values())[0]["ranks"]
    ax.set_xticks(sample_ranks)
    ax.set_xticklabels(sample_rois)

    ax.set_xlabel("Anatomical Cortical Hierarchy (Early -> Intermediate -> Category-Selective)")
    ax.set_ylabel("Peak-Predicting Layer Depth (Normalized [0, 1])")
    ax.set_title("Hierarchical Correspondence Between Deep Nets and Human Visual Cortex")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_selectivity_bars(
    profiles: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str = "figures/fig4_category_selectivity.png",
) -> None:
    """Plots measured vs. predicted mean category responses for FFA, PPA, and EBA."""
    target_rois = [r for r in ["FFA", "PPA", "EBA"] if r in profiles]
    if not target_rois:
        return

    fig, axes = plt.subplots(1, len(target_rois), figsize=(5 * len(target_rois), 4.5), sharey=True)
    if len(target_rois) == 1:
        axes = [axes]

    for ax, roi in zip(axes, target_rois):
        cats_data = profiles[roi]
        cats = list(cats_data.keys())
        x = np.arange(len(cats))
        width = 0.35

        m_means = [cats_data[c]["measured_mean"] for c in cats]
        m_sems = [cats_data[c]["measured_sem"] for c in cats]
        p_means = [cats_data[c]["predicted_mean"] for c in cats]
        p_sems = [cats_data[c]["predicted_sem"] for c in cats]

        ax.bar(x - width / 2, m_means, width, yerr=m_sems, capsize=3, label="Measured fMRI", color="#3182ce")
        ax.bar(x + width / 2, p_means, width, yerr=p_sems, capsize=3, label="Predicted (Model)", color="#e53e3e")

        ax.set_title(f"ROI: {roi} Tuning Profile")
        ax.set_xticks(x)
        clean_cat_labels = [c.replace("_", "\n") for c in cats]
        ax.set_xticklabels(clean_cat_labels, rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Normalized Response")
            ax.legend()

    plt.suptitle("Domain Selectivity Check: Measured vs. Predicted Category Responses", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cross_subject_robustness(
    subject_rhos: Dict[str, Dict[str, float]],
    output_path: str = "figures/fig5_cross_subject_robustness.png",
) -> None:
    """Plots hierarchy rank correlation Spearman rho across Subjects 1 to 4+."""
    subjects = list(subject_rhos.keys())
    models = list(list(subject_rhos.values())[0].keys())

    x = np.arange(len(subjects))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2b6cb0", "#319795", "#d69e2e", "#dd6b20", "#805ad5"]

    for i, m in enumerate(models):
        y_vals = [subject_rhos[s].get(m, 0.0) for s in subjects]
        color = colors[i % len(colors)]
        ax.bar(x + i * width, y_vals, width, label=m, color=color, alpha=0.9)

    ax.set_ylabel("Hierarchy Spearman $\\rho$")
    ax.set_title("Cross-Subject Robustness of Hierarchical Alignment")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in subjects])
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(frameon=True, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
