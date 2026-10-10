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


def compute_center_of_mass_depth_per_roi(
    summary_results: Dict[str, Any], model_key: str, baseline_subtracted: bool = True
) -> Dict[str, float]:
    """
    Computes continuous center-of-mass depth per ROI:
    d_bar = sum(d_i * w_i) / sum(w_i)

    When baseline_subtracted=True (recommended), w_i = max(0, r_i - min_l(r_l)),
    which removes the baseline correlation pedestal present across all layers
    and expands the dynamic range across the cortical hierarchy.
    When baseline_subtracted=False, w_i = max(0, r_i).
    """
    model_entries = [
        v for v in summary_results.values() if v.get("model_key") == model_key
    ]
    com_depths = {}
    for roi in ROI_HIERARCHY_RANK.keys():
        rs = [e.get("roi_medians", {}).get(roi, 0.0) for e in model_entries]
        ds = [e.get("normalized_depth", 0.0) for e in model_entries]

        if not rs:
            com_depths[roi] = 0.5
            continue

        r_arr = np.array(rs, dtype=float)
        d_arr = np.array(ds, dtype=float)

        if baseline_subtracted:
            min_r = np.min(r_arr)
            weights = np.maximum(0.0, r_arr - min_r)
        else:
            weights = np.maximum(0.0, r_arr)

        if np.sum(weights) > 0:
            com_depths[roi] = float(np.sum(d_arr * weights) / np.sum(weights))
        else:
            com_depths[roi] = float(np.mean(d_arr))
    return com_depths


def evaluate_hierarchy_correlation(
    roi_depths: Dict[str, Any], n_bootstraps: int = 1000, n_permutations: int = 10000
) -> Dict[str, Any]:
    """
    Calculates Spearman rho between anatomical hierarchy rank and normalized depth.
    Accepts either:
      - roi_peaks dict: {roi: {"anatomical_rank": int, "normalized_depth": float, ...}}
      - continuous com_depths dict: {roi: float}

    Computes:
      - Asymptotic p-value (Student's t / scipy)
      - Non-parametric Monte Carlo permutation p-value (shuffling ROI ranks N times)
      - 95% bootstrap confidence interval over ROIs
    """
    rois = list(roi_depths.keys())
    if len(rois) < 3:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "permutation_p_value": float("nan"),
            "permutation_p_formatted": "N/A",
            "ci_95": (float("nan"), float("nan")),
        }

    ranks = []
    depths = []
    for r in rois:
        val = roi_depths[r]
        if isinstance(val, dict):
            ranks.append(val.get("anatomical_rank", ROI_HIERARCHY_RANK.get(r, 0)))
            depths.append(val.get("normalized_depth", 0.0))
        else:
            ranks.append(ROI_HIERARCHY_RANK.get(r, 0))
            depths.append(float(val))

    ranks = np.array(ranks)
    depths = np.array(depths)

    rho, p_val = spearmanr(ranks, depths)

    # Fast Vectorized Monte Carlo permutation test
    from scipy.stats import rankdata
    ranks_ranked = rankdata(ranks)
    depths_ranked = rankdata(depths)
    r_c = ranks_ranked - np.mean(ranks_ranked)
    d_c = depths_ranked - np.mean(depths_ranked)
    denom = np.sqrt(np.sum(r_c**2) * np.sum(d_c**2))

    rng = np.random.RandomState(42)
    perm_matrix = np.array([rng.permutation(r_c) for _ in range(n_permutations)])
    perm_rhos = np.dot(perm_matrix, d_c) / denom if denom > 0 else np.zeros(n_permutations)
    
    # Empirical count of permutations exceeding or matching true rho
    extreme_count = np.sum(np.abs(perm_rhos) >= np.abs(rho))
    perm_p = float(extreme_count / n_permutations)
    perm_p_formatted = f"< {1/n_permutations:.0e}" if extreme_count == 0 else f"{perm_p:.4f}"

    # Bootstrap over ROIs
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
        "permutation_p_value": perm_p,
        "permutation_p_formatted": perm_p_formatted,
        "ci_95": (ci_lower, ci_upper),
        "rois": rois,
        "ranks": ranks.tolist(),
        "depths": depths.tolist(),
    }

