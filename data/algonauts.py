"""
Algonauts 2023 / NSD Dataset Loader and Preprocessing.

Loads natural scene images, left & right hemisphere fMRI vertex responses,
and ROI masks for Subjects 1-8. Supports memory-mapped reading and
deterministic held-out splitting.
"""

from typing import Dict, List, Optional, Set, Tuple
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
        Handles challenge_space files (*.prf-visualrois_challenge_space.npy),
        mapping dictionaries, and individual boolean ROI masks.
        """
        lh_masks: Dict[str, np.ndarray] = {}
        rh_masks: Dict[str, np.ndarray] = {}

        roi_search_dirs = [
            self.roi_dir,
            os.path.join(self.subj_dir, "ROIs"),
            os.path.join(self.data_root, "roi_masks"),
            os.path.join(self.data_root, "ROIs"),
            os.path.join(self.data_root, self.subject_id, "roi_masks"),
        ]
        active_roi_dir = next((d for d in roi_search_dirs if os.path.isdir(d)), self.roi_dir)

        def _find_file(dir_path: str, prefixes: List[str]) -> Optional[str]:
            for p in prefixes:
                cand = os.path.join(dir_path, p)
                if os.path.exists(cand):
                    return cand
            return None

        def _load_mapping_dict(dir_path: str, category: str) -> Optional[Dict]:
            prefixes = [
                f"mapping_{category}.npy",
                f"mapping-{category}.npy",
                f"{category}_mapping.npy",
            ]
            path = _find_file(dir_path, prefixes)
            if path:
                try:
                    data = np.load(path, allow_pickle=True)
                    if hasattr(data, "item"):
                        return data.item()
                    elif isinstance(data, dict):
                        return data
                except Exception:
                    pass
            return None

        def _get_mapping_ids(mapping: Optional[Dict], roi_names: List[str]) -> Set[int]:
            found_ids: Set[int] = set()
            if not mapping:
                return found_ids
            for k, v in mapping.items():
                for name in roi_names:
                    if isinstance(k, str) and name.lower() == k.lower():
                        if isinstance(v, (int, np.integer)):
                            found_ids.add(int(v))
                        elif isinstance(v, (list, tuple, np.ndarray)):
                            found_ids.update(int(x) for x in v)
                    elif isinstance(v, str) and name.lower() == v.lower():
                        if isinstance(k, (int, np.integer)):
                            found_ids.add(int(k))
                        elif isinstance(k, (list, tuple, np.ndarray)):
                            found_ids.update(int(x) for x in k)
            return found_ids

        prf_map = _load_mapping_dict(active_roi_dir, "prf-visualrois")
        faces_map = _load_mapping_dict(active_roi_dir, "floc-faces")
        bodies_map = _load_mapping_dict(active_roi_dir, "floc-bodies")
        places_map = _load_mapping_dict(active_roi_dir, "floc-places")

        for hemi, n_v, target_dict in [("lh", n_lh_vertices, lh_masks), ("rh", n_rh_vertices, rh_masks)]:
            # 1. PRF visual ROIs (V1, V2, V3, hV4)
            prf_path = _find_file(active_roi_dir, [
                f"{hemi}.prf-visualrois_challenge_space.npy",
                f"{hemi}.prf-visualrois.npy",
                f"{hemi}_prf-visualrois_challenge_space.npy",
                f"{hemi}_prf-visualrois.npy",
            ])
            if prf_path:
                try:
                    arr = np.load(prf_path, allow_pickle=True).squeeze()
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        v1_ids = _get_mapping_ids(prf_map, ["V1v", "V1d", "V1"]) or {1, 2}
                        v2_ids = _get_mapping_ids(prf_map, ["V2v", "V2d", "V2"]) or {3, 4}
                        v3_ids = _get_mapping_ids(prf_map, ["V3v", "V3d", "V3"]) or {5, 6}
                        hv4_ids = _get_mapping_ids(prf_map, ["hV4", "V4"]) or {7}

                        target_dict["V1"] = np.isin(arr, list(v1_ids))
                        target_dict["V2"] = np.isin(arr, list(v2_ids))
                        target_dict["V3"] = np.isin(arr, list(v3_ids))
                        target_dict["hV4"] = np.isin(arr, list(hv4_ids))
                except Exception as e:
                    print(f"Warning parsing PRF ROIs from {prf_path}: {e}")

            # 2. FLOC Faces (OFA, FFA)
            faces_path = _find_file(active_roi_dir, [
                f"{hemi}.floc-faces_challenge_space.npy",
                f"{hemi}.floc-faces.npy",
                f"{hemi}_floc-faces_challenge_space.npy",
                f"{hemi}_floc-faces.npy",
            ])
            if faces_path:
                try:
                    arr = np.load(faces_path, allow_pickle=True).squeeze()
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        ofa_ids = _get_mapping_ids(faces_map, ["OFA"]) or {1}
                        ffa_ids = _get_mapping_ids(faces_map, ["FFA", "FFA-1", "FFA-2"]) or {2, 3}

                        target_dict["OFA"] = np.isin(arr, list(ofa_ids))
                        target_dict["FFA"] = np.isin(arr, list(ffa_ids))
                except Exception as e:
                    print(f"Warning parsing Face ROIs from {faces_path}: {e}")

            # 3. FLOC Bodies (EBA, FBA)
            bodies_path = _find_file(active_roi_dir, [
                f"{hemi}.floc-bodies_challenge_space.npy",
                f"{hemi}.floc-bodies.npy",
                f"{hemi}_floc-bodies_challenge_space.npy",
                f"{hemi}_floc-bodies.npy",
            ])
            if bodies_path:
                try:
                    arr = np.load(bodies_path, allow_pickle=True).squeeze()
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        eba_ids = _get_mapping_ids(bodies_map, ["EBA"]) or {1}
                        target_dict["EBA"] = np.isin(arr, list(eba_ids))
                except Exception as e:
                    print(f"Warning parsing Body ROIs from {bodies_path}: {e}")

            # 4. FLOC Places (OPA, PPA, RSC)
            places_path = _find_file(active_roi_dir, [
                f"{hemi}.floc-places_challenge_space.npy",
                f"{hemi}.floc-places.npy",
                f"{hemi}_floc-places_challenge_space.npy",
                f"{hemi}_floc-places.npy",
            ])
            if places_path:
                try:
                    arr = np.load(places_path, allow_pickle=True).squeeze()
                    if isinstance(arr, np.ndarray) and arr.size == n_v:
                        opa_ids = _get_mapping_ids(places_map, ["OPA"]) or {1}
                        ppa_ids = _get_mapping_ids(places_map, ["PPA"]) or {2}
                        rsc_ids = _get_mapping_ids(places_map, ["RSC"]) or {3}

                        target_dict["OPA"] = np.isin(arr, list(opa_ids))
                        target_dict["PPA"] = np.isin(arr, list(ppa_ids))
                        target_dict["RSC"] = np.isin(arr, list(rsc_ids))
                except Exception as e:
                    print(f"Warning parsing Place ROIs from {places_path}: {e}")

            # 5. Check individual boolean ROI mask files (e.g. lh.V1.npy, lh.FFA.npy)
            for r_name in ["V1", "V2", "V3", "hV4", "OFA", "FFA", "OPA", "PPA", "EBA", "RSC"]:
                if r_name in target_dict:
                    continue
                cand_file = _find_file(active_roi_dir, [
                    f"{hemi}.{r_name}_challenge_space.npy",
                    f"{hemi}.{r_name}.npy",
                    f"{hemi}_{r_name}_challenge_space.npy",
                    f"{hemi}_{r_name}.npy",
                ])
                if cand_file:
                    try:
                        m_arr = np.load(cand_file, allow_pickle=True).squeeze()
                        if isinstance(m_arr, np.ndarray):
                            if m_arr.size == n_v:
                                target_dict[r_name] = m_arr.astype(bool)
                            elif np.issubdtype(m_arr.dtype, np.integer) and len(m_arr) > 0 and np.max(m_arr) < n_v:
                                mask = np.zeros(n_v, dtype=bool)
                                mask[m_arr] = True
                                target_dict[r_name] = mask
                    except Exception:
                        pass

        roi_mgr = ROIManager.from_algonauts_masks(lh_masks, rh_masks)
        n_found = len(roi_mgr.roi_dict)
        if n_found > 0:
            print(f"Loaded {n_found} visual ROIs: {list(roi_mgr.roi_dict.keys())}")
        else:
            print(f"Notice: No predefined ROI masks found in {active_roi_dir}.")
        return roi_mgr
