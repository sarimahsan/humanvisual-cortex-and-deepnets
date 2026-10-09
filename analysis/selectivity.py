"""
Category Selectivity Analysis in Ventral Visual Cortex.

Tests whether deep neural network predictions in category-selective regions
(FFA, PPA, EBA) reproduce biological category preferences (faces, scenes, bodies)
relative to control object categories (animals, food, vehicles).
"""

from typing import Dict, List, Any
import numpy as np
from data.coco_labels import COCOCategoryManager, SUPER_CATEGORIES
from data.rois import ROIManager


def compute_category_tuning_profile(
    Y_test_true: np.ndarray,
    Y_test_pred: np.ndarray,
    test_image_names: List[str],
    category_mgr: COCOCategoryManager,
    roi_mgr: ROIManager,
    target_rois: List[str] = ("FFA", "PPA", "EBA"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Computes average measured vs. predicted response per semantic category for target ROIs.

    Returns:
        profiles: {
            roi_name: {
                category_name: {
                    "measured_mean": float,
                    "predicted_mean": float,
                    "measured_sem": float,
                    "predicted_sem": float,
                    "n_images": int
                }
            }
        }
    """
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    categories = list(SUPER_CATEGORIES.keys())

    for roi in target_rois:
        if roi not in roi_mgr.roi_dict:
            continue

        roi_idx = roi_mgr.get_roi_indices(roi)
        if len(roi_idx) == 0:
            continue

        # ROI mean response per test image: (N_test,)
        true_roi_mean = np.mean(Y_test_true[:, roi_idx], axis=1)
        pred_roi_mean = np.mean(Y_test_pred[:, roi_idx], axis=1)

        profiles[roi] = {}

        for cat in categories:
            cat_mask = category_mgr.get_category_mask(test_image_names, cat)
            n_images = int(np.sum(cat_mask))

            if n_images > 0:
                m_mean = float(np.mean(true_roi_mean[cat_mask]))
                m_sem = float(np.std(true_roi_mean[cat_mask]) / np.sqrt(n_images))
                p_mean = float(np.mean(pred_roi_mean[cat_mask]))
                p_sem = float(np.std(pred_roi_mean[cat_mask]) / np.sqrt(n_images))
            else:
                m_mean, m_sem, p_mean, p_sem = 0.0, 0.0, 0.0, 0.0

            profiles[roi][cat] = {
                "measured_mean": m_mean,
                "measured_sem": m_sem,
                "predicted_mean": p_mean,
                "predicted_sem": p_sem,
                "n_images": n_images,
            }

    return profiles


def evaluate_selectivity_correlation(
    profiles: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Computes Pearson correlation between the vector of measured category means
    and predicted category means for each ROI.
    """
    roi_profile_corrs = {}
    for roi, cats_data in profiles.items():
        m_vec = [cats_data[c]["measured_mean"] for c in cats_data]
        p_vec = [cats_data[c]["predicted_mean"] for c in cats_data]

        if len(m_vec) > 2 and np.std(m_vec) > 1e-6 and np.std(p_vec) > 1e-6:
            r = float(np.corrcoef(m_vec, p_vec)[0, 1])
        else:
            r = 0.0
        roi_profile_corrs[roi] = r

    return roi_profile_corrs
