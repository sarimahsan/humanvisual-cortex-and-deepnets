"""
Algonauts 2023 / NSD Dataset Loader and Preprocessing.

Loads natural scene images, left & right hemisphere fMRI vertex responses,
and ROI masks for Subjects 1-8. Supports memory-mapped reading and
deterministic held-out splitting.
"""

from typing import Dict, List, Optional, Tuple
import glob
import os
import numpy as np
from PIL import Image

from data.rois import ROIManager
from data.splits import SplitManager


class AlgonautsDataset:
    """Loader for an individual subject's Algonauts 2023 data."""

    def __init__(self, data_root: str, subject_id: str = "subj01"):
        """
        Args:
            data_root: Root directory of Algonauts 2023 data.
            subject_id: Subject identifier (e.g. 'subj01').
        """
        self.data_root = data_root
        self.subject_id = subject_id
        self.subj_dir = os.path.join(data_root, subject_id)

        self.images_dir = os.path.join(self.subj_dir, "training_split", "training_images")
        self.fmri_dir = os.path.join(self.subj_dir, "training_split", "training_fmri")
        self.roi_dir = os.path.join(self.subj_dir, "roi_masks")

    def exists(self) -> bool:
        """Returns True if the subject directory and core files exist on disk."""
        return os.path.exists(self.images_dir) and os.path.exists(self.fmri_dir)

    def get_image_paths(self) -> List[str]:
        """Returns sorted list of image file paths for this subject."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        image_paths: List[str] = []
        for p in patterns:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, p)))
        image_paths.sort()
        return image_paths

    def load_fmri_responses(
        self, mmap_mode: Optional[str] = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Loads and horizontally concatenates LH and RH fMRI responses.
        Returns:
            concatenated_responses: shape (n_images, n_total_vertices)
            (n_lh_vertices, n_rh_vertices)
        """
        lh_path = os.path.join(self.fmri_dir, "lh_training_fmri.npy")
        rh_path = os.path.join(self.fmri_dir, "rh_training_fmri.npy")

        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            raise FileNotFoundError(f"fMRI arrays not found in {self.fmri_dir}")

        lh_fmri = np.load(lh_path, mmap_mode=mmap_mode)
        rh_fmri = np.load(rh_path, mmap_mode=mmap_mode)

        n_lh = lh_fmri.shape[1]
        n_rh = rh_fmri.shape[1]

        # Combine hemispheres along vertex axis
        combined = np.concatenate([lh_fmri, rh_fmri], axis=1)
        return combined, (n_lh, n_rh)

    def load_roi_manager(
        self, n_lh_vertices: int, n_rh_vertices: int
    ) -> ROIManager:
        """
        Parses ROI mask files from roi_masks/ directory.
        Handles both individual boolean masks (lh.V1v.npy) and challenge challenge mapping files.
        """
        lh_masks: Dict[str, np.ndarray] = {}
        rh_masks: Dict[str, np.ndarray] = {}

        if not os.path.exists(self.roi_dir):
            return ROIManager({})

        for hemi, n_v, target_dict in [("lh", n_lh_vertices, lh_masks), ("rh", n_rh_vertices, rh_masks)]:
            # 1. PRF visual ROIs (V1, V2, V3, hV4)
            prf_path = os.path.join(self.roi_dir, f"{hemi}.prf-visualrois.npy")
            if os.path.exists(prf_path):
                try:
                    arr = np.load(prf_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        target_dict["V1"] = (arr == 1) | (arr == 2)
                        target_dict["V2"] = (arr == 3) | (arr == 4)
                        target_dict["V3"] = (arr == 5) | (arr == 6)
                        target_dict["hV4"] = (arr == 7)
                except Exception:
                    pass

            # 2. FLOC Faces (OFA, FFA)
            faces_path = os.path.join(self.roi_dir, f"{hemi}.floc-faces.npy")
            if os.path.exists(faces_path):
                try:
                    arr = np.load(faces_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        target_dict["OFA"] = (arr == 1)
                        target_dict["FFA"] = (arr == 2) | (arr == 3)
                except Exception:
                    pass

            # 3. FLOC Bodies (EBA, FBA)
            bodies_path = os.path.join(self.roi_dir, f"{hemi}.floc-bodies.npy")
            if os.path.exists(bodies_path):
                try:
                    arr = np.load(bodies_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        target_dict["EBA"] = (arr == 1)
                except Exception:
                    pass

            # 4. FLOC Places (OPA, PPA, RSC)
            places_path = os.path.join(self.roi_dir, f"{hemi}.floc-places.npy")
            if os.path.exists(places_path):
                try:
                    arr = np.load(places_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        target_dict["OPA"] = (arr == 1)
                        target_dict["PPA"] = (arr == 2)
                        target_dict["RSC"] = (arr == 3)
                except Exception:
                    pass

            # 5. Check individual boolean ROI mask files (e.g. lh.V1.npy, lh.FFA.npy)
            for r_name in ["V1", "V2", "V3", "hV4", "OFA", "FFA", "OPA", "PPA", "EBA", "RSC"]:
                if r_name in target_dict:
                    continue
                cand_files = [
                    os.path.join(self.roi_dir, f"{hemi}.{r_name}.npy"),
                    os.path.join(self.roi_dir, f"{hemi}_{r_name}.npy"),
                ]
                for cf in cand_files:
                    if os.path.exists(cf):
                        try:
                            m_arr = np.load(cf, allow_pickle=True)
                            if isinstance(m_arr, np.ndarray) and m_arr.size == n_v:
                                target_dict[r_name] = m_arr.astype(bool)
                                break
                        except Exception:
                            pass

        return ROIManager.from_algonauts_masks(lh_masks, rh_masks)
