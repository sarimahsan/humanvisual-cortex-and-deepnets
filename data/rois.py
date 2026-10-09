"""
ROI Definitions and Hierarchy Mappings for Algonauts 2023 / NSD Dataset.

Manages ROI indices, groupings (Early, Intermediate, High/Category),
and anatomical hierarchy ranks for rank-order correlation analysis.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np

# Canonical anatomical hierarchy rank order
ROI_HIERARCHY_RANK: Dict[str, float] = {
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

# Groupings
ROI_GROUPS: Dict[str, List[str]] = {
    "Early": ["V1", "V2", "V3"],
    "Intermediate": ["hV4"],
    "Body": ["EBA", "FBA"],
    "Face": ["FFA", "OFA"],
    "Place": ["PPA", "RSC", "OPA"],
    "Category_Selective": ["FFA", "PPA", "EBA", "OFA", "OPA", "RSC"]
}

# Color palette for consistent plotting
ROI_PALETTE: Dict[str, str] = {
    "V1": "#2b5c8f",
    "V2": "#417bb8",
    "V3": "#5c9ee0",
    "hV4": "#38a169",
    "OFA": "#d69e2e",
    "FFA": "#dd6b20",
    "OPA": "#9f7aea",
    "PPA": "#805ad5",
    "EBA": "#e53e3e",
    "RSC": "#b83280",
}


class ROIManager:
    """Manages vertex-to-ROI mappings across hemispheres."""

    def __init__(self, roi_dict: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            roi_dict: Dictionary mapping ROI name -> boolean mask or index array
                      of vertices belonging to that ROI.
        """
        self.roi_dict: Dict[str, np.ndarray] = roi_dict or {}

    def get_roi_indices(self, roi_name: str) -> np.ndarray:
        """Returns integer vertex indices for a given ROI."""
        if roi_name not in self.roi_dict:
            raise KeyError(f"ROI '{roi_name}' not found. Available: {list(self.roi_dict.keys())}")
        mask_or_idx = self.roi_dict[roi_name]
        if mask_or_idx.dtype == bool:
            return np.where(mask_or_idx)[0]
        return mask_or_idx.astype(int)

    def get_rois_in_group(self, group_name: str) -> List[str]:
        """Returns list of ROIs existing in the current dataset belonging to group_name."""
        candidates = ROI_GROUPS.get(group_name, [])
        return [r for r in candidates if r in self.roi_dict]

    def summarize_roi_metric(
        self, vertex_scores: np.ndarray, roi_name: str, method: str = "median"
    ) -> float:
        """
        Summarizes vertex prediction accuracy (e.g. Pearson r) for an ROI.
        Uses median by default to be robust against outliers and non-responsive vertices.
        """
        indices = self.get_roi_indices(roi_name)
        if len(indices) == 0:
            return float("nan")
        scores = vertex_scores[indices]
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) == 0:
            return float("nan")

        if method == "median":
            return float(np.median(valid_scores))
        elif method == "mean":
            return float(np.mean(valid_scores))
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    @classmethod
    def from_algonauts_masks(
        cls, lh_roi_masks: Dict[str, np.ndarray], rh_roi_masks: Dict[str, np.ndarray]
    ) -> "ROIManager":
        """
        Constructs unified ROIManager by concatenating left and right hemisphere masks.
        """
        all_rois: Set[str] = set(lh_roi_masks.keys()).union(set(rh_roi_masks.keys()))
        combined: Dict[str, np.ndarray] = {}

        for roi in all_rois:
            lh = lh_roi_masks.get(roi)
            rh = rh_roi_masks.get(roi)
            if lh is not None and rh is not None:
                mask = np.concatenate([lh.astype(bool), rh.astype(bool)])
            elif lh is not None:
                mask = lh.astype(bool)
            else:
                mask = rh.astype(bool)
            combined[roi] = mask

        return cls(combined)
