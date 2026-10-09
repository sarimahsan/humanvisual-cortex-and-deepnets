"""
Cortical Hierarchy Analysis: Layer Depth vs Anatomical ROI Hierarchy.

Computes the peak-predicting layer (normalized depth [0, 1]) for each visual ROI
and evaluates rank-order correlation (Spearman rho) with anatomical hierarchy order.
Computes bootstrap confidence intervals and permutation p-values.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.stats import spearmanr

from data.rois import ROI_HIERARCHY_RANK, ROI_GROUPS


def compute_peak_layer_per_roi(
    summary_results: Dict[str, Any], model_key: str
) -> Dict[str, Dict[str, float]]:
    """
    Finds the layer with the highest prediction accuracy for each ROI for model_key.

    Returns:
        roi_peaks: {roi_name: {"best_layer": name, "normalized_depth": float, "max_r": float}}
    """
    # Filter entries for model_key
    model_entries = [
        v for v in summary_results.values() if v.get("model_key") == model_key
    ]

    roi_peaks = {}
    for roi in ROI_HIERARCHY_RANK.keys():
        best_layer = None
        best_depth = 0.0
        best_r = -1.0

        for entry in model_entries:
            roi_medians = entry.get("roi_medians", {})
            r_val = roi_medians.get(roi, float("nan"))
            if not np.isnan(r_val) and r_val > best_r:
                best_r = r_val
                best_layer = entry.get("layer_name")
                best_depth = entry.get("normalized_depth", 0.0)

        if best_layer is not None:
            roi_peaks[roi] = {
                "best_layer": best_layer,
                "normalized_depth": best_depth,
                "max_r": best_r,
                "anatomical_rank": ROI_HIERARCHY_RANK[roi],
            }

    return roi_peaks


def evaluate_hierarchy_correlation(
    roi_peaks: Dict[str, Dict[str, float]], n_bootstraps: int = 1000
) -> Dict[str, Any]:
    """
    Calculates Spearman rho between anatomical hierarchy rank and best-layer normalized depth.
    Computes 95% bootstrap confidence interval over ROIs.
    """
    rois = list(roi_peaks.keys())
    if len(rois) < 3:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "ci_95": (float("nan"), float("nan"))}

    ranks = np.array([roi_peaks[r]["anatomical_rank"] for r in rois])
    depths = np.array([roi_peaks[r]["normalized_depth"] for r in rois])

    rho, p_val = spearmanr(ranks, depths)

    # Bootstrap over ROIs
    rng = np.random.RandomState(42)
    boot_rhos = []
    n = len(ranks)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(ranks[idx])) > 1 and len(np.unique(depths[idx])) > 1:
            b_rho, _ = spearmanr(ranks[idx], depths[idx])
            boot_rhos.append(b_rho)

    ci_lower = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else float("nan")
    ci_upper = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else float("nan")

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "ci_95": (ci_lower, ci_upper),
        "rois": rois,
        "ranks": ranks.tolist(),
        "depths": depths.tolist(),
    }
