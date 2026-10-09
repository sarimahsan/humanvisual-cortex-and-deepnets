"""
End-to-End Brain Encoding Pipeline Orchestrator.

One single command reproduces:
1. Feature extraction across models and controls.
2. SVD / Kernel Ridge regression encoding models.
3. Out-of-sample held-out test evaluation.
4. Hierarchy analysis (Spearman rho & bootstrap CIs).
5. Generation of all publication-quality static figures.
"""

import argparse
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from features.extract import extract_model_features
from encoding.pipeline import EncodingPipeline
from analysis.hierarchy import compute_peak_layer_per_roi, evaluate_hierarchy_correlation
from analysis.plotting import (
    plot_layer_roi_heatmap,
    plot_model_comparison_per_roi,
    plot_hierarchy_regression,
)


def run_end_to_end(
    subject_id: str = "subj01",
    data_root: str = "data/raw/algonauts_2023",
    models: list = None,
    device: str = None,
    skip_extraction: bool = False,
    skip_fitting: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if models is None:
        models = ["alexnet", "resnet50", "resnet50_untrained", "gabor_pyramid"]

    print("=" * 70)
    print("🧠 BRAIN ENCODING PIPELINE: END-TO-END EXECUTION")
    print(f"   Subject:    {subject_id}")
    print(f"   Data Root:  {data_root}")
    print(f"   Device:     {device.upper()}")
    print(f"   Models:     {', '.join(models)}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Feature Extraction
    # -------------------------------------------------------------
    if not skip_extraction:
        print("\n[Step 1/4] Running Feature Extraction across models...")
        for m in models:
            print(f"\n--> Extracting features for: {m}")
            extract_model_features(
                model_key=m,
                subject_id=subject_id,
                data_root=data_root,
                device=device,
            )
    else:
        print("\n[Step 1/4] Skipping Feature Extraction (using existing cache)...")

    # -------------------------------------------------------------
    # Step 2: Fit Ridge Regression Encoding Models
    # -------------------------------------------------------------
    print("\n[Step 2/4] Fitting SVD Ridge Regression Encoding Models...")
    pipeline = EncodingPipeline(subject_id=subject_id, data_root=data_root)

    npz_path = os.path.join("results", f"vertex_correlations_{subject_id}.npz")
    summary_path = os.path.join("results", f"summary_{subject_id}.json")

    if skip_fitting and os.path.exists(npz_path) and os.path.exists(summary_path):
        print(f"Loading existing vertex correlations from {npz_path} (skipping re-fitting)...")
        results_summary = pipeline.recalculate_roi_summary_from_npz(npz_path, summary_path)
    else:
        results_summary = pipeline.run_full_subject(selected_models=models)

    # -------------------------------------------------------------
    # Step 3: Hierarchy Analysis (Spearman rho)
    # -------------------------------------------------------------
    print("\n[Step 3/4] Evaluating Cortical Hierarchy Alignment...")
    hierarchy_results = {}
    print("\n-------------------------------------------------------------")
    print(f"{'Model':<25} | {'Hierarchy ρ':<12} | {'p-value':<10} | {'95% CI'}")
    print("-------------------------------------------------------------")

    for m in models:
        peaks = compute_peak_layer_per_roi(results_summary, m)
        if len(peaks) >= 3:
            h_eval = evaluate_hierarchy_correlation(peaks)
            hierarchy_results[m] = h_eval
            rho = h_eval["spearman_rho"]
            pval = h_eval["p_value"]
            ci = h_eval["ci_95"]
            print(f"{m:<25} | {rho:+.3f}        | {pval:<10.4f} | [{ci[0]:.2f}, {ci[1]:.2f}]")
        else:
            print(f"{m:<25} | {'< 3 ROIs':<12} | {'N/A':<10} | N/A")
    print("-------------------------------------------------------------")

    # -------------------------------------------------------------
    # Step 4: Generate All Publication Figures
    # -------------------------------------------------------------
    print("\n[Step 4/4] Generating Publication Figures in figures/...")
    os.makedirs("figures", exist_ok=True)

    # Figure 1: Heatmap
    heatmap_model = "resnet50" if "resnet50" in models else models[0]
    plot_layer_roi_heatmap(
        results_summary, heatmap_model, output_path="figures/fig1_layer_roi_heatmap.png"
    )

    # Figure 2: Model Comparison
    best_scores = {}
    for m in models:
        peaks = compute_peak_layer_per_roi(results_summary, m)
        if peaks:
            best_scores[m] = {r: peaks[r]["max_r"] for r in peaks if not np.isnan(peaks[r]["max_r"])}
    if best_scores:
        plot_model_comparison_per_roi(
            best_scores, output_path="figures/fig2_model_comparison.png"
        )
    else:
        print("Notice: Insufficient ROI peak scores for Figure 2. Skipping.")

    # Figure 3: Hierarchy Regression
    if hierarchy_results:
        plot_hierarchy_regression(
            hierarchy_results, output_path="figures/fig3_hierarchy_alignment.png"
        )
    else:
        print("Notice: Insufficient hierarchy results for Figure 3. Skipping.")

    print("\n" + "=" * 70)
    print("✅ END-TO-END PIPELINE COMPLETE!")
    print(f"   Summary JSON:    results/summary_{subject_id}.json")
    print(f"   Vertex Arrays:   results/vertex_correlations_{subject_id}.npz")
    print(f"   Figures:         figures/fig1_layer_roi_heatmap.png")
    print(f"                    figures/fig2_model_comparison.png")
    print(f"                    figures/fig3_hierarchy_alignment.png")
    print("   To launch UI:    python app/app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Brain Encoding Runner")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject ID")
    parser.add_argument("--data_root", type=str, default="data/raw/algonauts_2023", help="Root data folder")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--models", nargs="*", default=None, help="List of models to run")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip_fitting", action="store_true", help="Skip fitting if results already exist")
    args = parser.parse_args()

    run_end_to_end(
        subject_id=args.subject,
        data_root=args.data_root,
        models=args.models,
        device=args.device,
        skip_extraction=args.skip_extraction,
        skip_fitting=args.skip_fitting,
    )
