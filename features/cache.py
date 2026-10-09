"""
Feature Cache Manager.

Stores and retrieves intermediate neural representations as float16 numpy arrays
with metadata to ensure feature extraction runs only once and saves disk space.
"""

from typing import Any, Dict, Optional, Tuple
import json
import os
import numpy as np


class FeatureCache:
    """Manages disk caching of extracted representations."""

    def __init__(self, cache_dir: str = "features/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_paths(self, model_key: str, layer_name: str, subject_id: str) -> Tuple[str, str]:
        safe_layer = layer_name.replace(".", "_").replace("/", "_")
        base = f"{subject_id}_{model_key}_{safe_layer}"
        array_path = os.path.join(self.cache_dir, f"{base}.npy")
        meta_path = os.path.join(self.cache_dir, f"{base}.json")
        return array_path, meta_path

    def exists(self, model_key: str, layer_name: str, subject_id: str) -> bool:
        array_path, meta_path = self._get_paths(model_key, layer_name, subject_id)
        return os.path.exists(array_path) and os.path.exists(meta_path)

    def save(
        self,
        features: np.ndarray,
        model_key: str,
        layer_name: str,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves array as float16 .npy along with metadata JSON."""
        array_path, meta_path = self._get_paths(model_key, layer_name, subject_id)
        # Ensure float16 for storage efficiency
        if features.dtype != np.float16:
            features = features.astype(np.float16)

        np.save(array_path, features)

        meta = {
            "model_key": model_key,
            "layer_name": layer_name,
            "subject_id": subject_id,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(
        self,
        model_key: str,
        layer_name: str,
        subject_id: str,
        mmap_mode: Optional[str] = "r",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads cached float16 feature array and metadata."""
        array_path, meta_path = self._get_paths(model_key, layer_name, subject_id)
        if not self.exists(model_key, layer_name, subject_id):
            raise FileNotFoundError(f"Cache miss for {model_key} - {layer_name} ({subject_id})")

        arr = np.load(array_path, mmap_mode=mmap_mode)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return arr, meta
