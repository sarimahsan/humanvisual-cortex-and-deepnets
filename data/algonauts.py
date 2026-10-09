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

        roi_files = glob.glob(os.path.join(self.roi_dir, "*.npy"))

        # 1. Parse individual mask files (e.g. lh.V1.npy, lh.FFA.npy)
        for rf in roi_files:
            fname = os.path.basename(rf)
            name_parts = fname.replace(".npy", "").split(".")
            if len(name_parts) >= 2:
                hemi = name_parts[0]
                roi_name = ".".join(name_parts[1:])
                # Clean up sub-divisions (e.g. V1v, V1d -> V1)
                simplified_name = roi_name
                for base_roi in ["V1", "V2", "V3"]:
                    if roi_name.startswith(base_roi):
                        simplified_name = base_roi
                        break

                mask_data = np.load(rf).astype(bool)
                target_dict = lh_masks if hemi.lower().startswith("lh") else rh_masks
                if simplified_name in target_dict:
                    target_dict[simplified_name] = target_dict[simplified_name] | mask_data
                else:
                    target_dict[simplified_name] = mask_data

        # 2. Check for challenge floc / prf mapping files if masks are empty
        mapping_files = [f for f in roi_files if "mapping_" in os.path.basename(f)]
        for mf in mapping_files:
            fname = os.path.basename(mf)
            hemi = "lh" if "lh." in fname or fname.startswith("lh_") else "rh"
            n_v = n_lh_vertices if hemi == "lh" else n_rh_vertices
            map_arr = np.load(mf)
            # Check challenge integer mappings
            # PRF visual ROIs: 1: V1v, 2: V1d, 3: V2v, 4: V2d, 5: V3v, 6: V3d, 7: hV4
            if "prf-visualrois" in fname:
                target = lh_masks if hemi == "lh" else rh_masks
                target["V1"] = (map_arr == 1) | (map_arr == 2)
                target["V2"] = (map_arr == 3) | (map_arr == 4)
                target["V3"] = (map_arr == 5) | (map_arr == 6)
                target["hV4"] = (map_arr == 7)
            # floc-faces: 1: OFA, 2: FFA-1, 3: FFA-2
            elif "floc-faces" in fname:
                target = lh_masks if hemi == "lh" else rh_masks
                target["OFA"] = (map_arr == 1)
                target["FFA"] = (map_arr == 2) | (map_arr == 3)
            # floc-bodies: 1: EBA, 2: FBA-1, 3: FBA-2
            elif "floc-bodies" in fname:
                target = lh_masks if hemi == "lh" else rh_masks
                target["EBA"] = (map_arr == 1)
            # floc-places: 1: OPA, 2: PPA, 3: RSC
            elif "floc-places" in fname:
                target = lh_masks if hemi == "lh" else rh_masks
                target["OPA"] = (map_arr == 1)
                target["PPA"] = (map_arr == 2)
                target["RSC"] = (map_arr == 3)

        return ROIManager.from_algonauts_masks(lh_masks, rh_masks)
