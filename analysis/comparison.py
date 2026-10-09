"""
Statistical Model Comparisons via Paired Bootstrapping.

Performs rigorous hypothesis testing between model representations:
1. DeiT-S vs. DINO ViT-S/16 (Supervised vs. Self-Supervised objective, matched architecture)
2. Trained ResNet-50 vs. Untrained ResNet-50 (Learned features vs. architectural inductive bias)
3. Gabor Wavelet Pyramid vs. Early DNN layers (Classical spatial statistics vs. learned filters)
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def fast_pearson_vector(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes Pearson r across images (axis 0) for each vertex (axis 1)."""
    yt_c = y_true - y_true.mean(axis=0, keepdims=True)
    yp_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    cov = (yt_c * yp_c).sum(axis=0)
    std_t = np.sqrt((yt_c**2).sum(axis=0) + 1e-8)
    std_p = np.sqrt((yp_c**2).sum(axis=0) + 1e-8)
    return np.clip(cov / (std_t * std_p), -1.0, 1.0)


def paired_image_bootstrap_test(
    Y_test_true: np.ndarray,
    Y_pred_A: np.ndarray,
    Y_pred_B: np.ndarray,
    roi_indices: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Performs paired bootstrap over test images (axis 0) to evaluate whether
    Model A significantly outperforms Model B within an ROI.

    Returns:
        {
            "delta_r_mean": float,
            "ci_95_lower": float,
            "ci_95_upper": float,
            "p_value_two_sided": float,
            "p_value_one_sided": float
        }
    """
    n_test = Y_test_true.shape[0]
    rng = np.random.RandomState(seed)

    boot_diffs = np.zeros(n_bootstraps, dtype=np.float32)

    for b in range(n_bootstraps):
        boot_idx = rng.randint(0, n_test, size=n_test)
        yt_b = Y_test_true[boot_idx][:, roi_indices]
        yp_A_b = Y_pred_A[boot_idx][:, roi_indices]
        yp_B_b = Y_pred_B[boot_idx][:, roi_indices]

        r_A = fast_pearson_vector(yt_b, yp_A_b)
        r_B = fast_pearson_vector(yt_b, yp_B_b)

        # ROI summary: median across vertices
        diff = np.median(r_A) - np.median(r_B)
        boot_diffs[b] = diff

    delta_mean = float(np.mean(boot_diffs))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Empirical p-values
    p_one_sided = float(np.mean(boot_diffs <= 0.0) if delta_mean > 0 else np.mean(boot_diffs >= 0.0))
    p_two_sided = min(1.0, 2.0 * p_one_sided)

    return {
        "delta_r_mean": delta_mean,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
    }
