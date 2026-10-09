"""
End-to-End Encoding Model Pipeline.

Coordinates:
1. Deterministic train / held-out test partitioning.
2. Train-only PCA dimensionality reduction & standardization.
3. Chunked Ridge regression with inner CV alpha optimization.
4. Out-of-sample evaluation on held-out test stimuli.
5. Shuffled null permutation check.
6. ROI-level summarization and metrics persistence.
"""

from typing import Dict, List, Optional, Any
import argparse
import os
import json
import yaml
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from data.algonauts import AlgonautsDataset
from data.splits import SplitManager
from data.rois import ROIManager
from features.cache import FeatureCache
from encoding.pca import FeatureReducer
from encoding.ridge import FastRidgeCV


class EncodingPipeline:
    """Manages model fitting and evaluation for a subject."""

    def __init__(
        self,
        subject_id: str = "subj01",
        data_root: str = "data/raw/algonauts_2023",
        models_cfg_path: str = "configs/models.yaml",
        exp_cfg_path: str = "configs/experiment.yaml",
        output_dir: str = "results",
    ):
        self.subject_id = subject_id
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(models_cfg_path, "r", encoding="utf-8") as f:
            self.models_cfg = yaml.safe_load(f)

        with open(exp_cfg_path, "r", encoding="utf-8") as f:
            self.exp_cfg = yaml.safe_load(f)

        self.cache = FeatureCache(cache_dir=self.exp_cfg["features"]["cache_dir"])

    def run_layer_encoding(
        self,
        model_key: str,
        layer_name: str,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        split_mgr: SplitManager,
        roi_mgr: ROIManager,
        run_shuffled_null: bool = False,
    ) -> Dict[str, Any]:
        """Runs train-only PCA, chunked ridge regression, and test evaluation for one layer."""
        train_idx = split_mgr.train_indices
        test_idx = split_mgr.test_indices

        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]

        Y_train = Y_all[train_idx]
        Y_test = Y_all[test_idx]

        # 1. Train-only PCA & scaling
        pca_dim = self.exp_cfg["features"].get("pca_components", 1024)
        reducer = FeatureReducer(n_components=pca_dim, random_state=self.exp_cfg["experiment"]["seed"])
        X_train_pca = reducer.fit_transform_train(X_train_raw)
        X_test_pca = reducer.transform_test(X_test_raw)

        # 2. Chunked Ridge Regression
        chunk_size = self.exp_cfg["encoding"].get("vertex_chunk_size", 2048)
        solver = FastRidgeCV(
            alphas=self.exp_cfg["encoding"]["alpha_grid"],
            cv_folds=self.exp_cfg["encoding"]["cv_folds"],
            device=self.exp_cfg["encoding"]["device"],
        )

        n_vertices = Y_all.shape[1]
        all_r = np.zeros(n_vertices, dtype=np.float32)
        inner_cv_folds = split_mgr.get_cv_folds(n_splits=self.exp_cfg["encoding"]["cv_folds"])
        # Map relative fold indices to training array indices
        train_index_map = {idx: i for i, idx in enumerate(train_idx)}
        mapped_folds = [
            (np.array([train_index_map[i] for i in tr]),
             np.array([train_index_map[i] for i in val]))
            for tr, val in inner_cv_folds
        ]

        for start_idx in range(0, n_vertices, chunk_size):
            end_idx = min(start_idx + chunk_size, n_vertices)
            Y_tr_chunk = Y_train[:, start_idx:end_idx]
            Y_te_chunk = Y_test[:, start_idx:end_idx]

            chunk_r, _, _ = solver.fit_predict_chunk(
                X_train_pca, Y_tr_chunk, X_test_pca, Y_te_chunk, inner_folds=mapped_folds
            )
            all_r[start_idx:end_idx] = chunk_r

        # 3. Optional Shuffled Null Check
        null_roi_medians = {}
        if run_shuffled_null:
            rng = np.random.RandomState(42)
            perm_train = rng.permutation(len(train_idx))
            Y_train_perm = Y_train[perm_train]

            # Fit on first 500 vertices to verify chance level
            null_r, _, _ = solver.fit_predict_chunk(
                X_train_pca, Y_train_perm[:, :500], X_test_pca, Y_test[:, :500]
            )
            null_mean = float(np.mean(null_r))
        else:
            null_mean = 0.0

        # 4. Summarize per ROI
        roi_medians: Dict[str, float] = {}
        for roi in roi_mgr.roi_dict.keys():
            roi_medians[roi] = roi_mgr.summarize_roi_metric(all_r, roi, method="median")

        return {
            "model_key": model_key,
            "layer_name": layer_name,
            "overall_median_r": float(np.nanmedian(all_r)),
            "roi_medians": roi_medians,
            "null_sample_r": null_mean,
            "vertex_r": all_r,
        }

    def run_full_subject(
        self, selected_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs the entire encoding experiment for this subject across all layers and models."""
        dataset = AlgonautsDataset(self.data_root, self.subject_id)
        if not dataset.exists():
            raise FileNotFoundError(f"Subject data not found at {dataset.subj_dir}")

        print(f"Loading fMRI responses for {self.subject_id}...")
        Y_all, (n_lh, n_rh) = dataset.load_fmri_responses()
        roi_mgr = dataset.load_roi_manager(n_lh, n_rh)
        n_images, n_vertices = Y_all.shape
        print(f"Loaded {n_images} stimuli images, {n_vertices} total vertices across both hemispheres.")

        split_mgr = SplitManager(
            n_samples=n_images,
            test_ratio=self.exp_cfg["data"]["test_split_ratio"],
            seed=self.exp_cfg["experiment"]["seed"],
        )
        split_mgr.verify_no_overlap()

        available_models = list(self.models_cfg.get("models", {}).keys()) + ["gabor_pyramid"]
        if selected_models:
            models_to_run = [m for m in selected_models if m in available_models]
        else:
            models_to_run = available_models

        all_results = {}

        for m_key in models_to_run:
            print(f"\n==========================================")
            print(f"Fitting Encoding Models for: {m_key}")
            print(f"==========================================")

            if m_key == "gabor_pyramid":
                X_feat, _ = self.cache.load(m_key, "multiscale_energy", self.subject_id)
                res = self.run_layer_encoding(
                    m_key, "multiscale_energy", X_feat, Y_all, split_mgr, roi_mgr, run_shuffled_null=True
                )
                res["normalized_depth"] = 0.05
                all_results[f"{m_key}_multiscale_energy"] = res
                continue

            m_cfg = self.models_cfg["models"][m_key]
            for l_info in m_cfg["layers"]:
                l_name = l_info["name"]
                if not self.cache.exists(m_key, l_name, self.subject_id):
                    print(f"Features for {m_key} - {l_name} missing from cache. Run extractor first!")
                    continue

                X_feat, _ = self.cache.load(m_key, l_name, self.subject_id)
                res = self.run_layer_encoding(
                    m_key, l_name, X_feat, Y_all, split_mgr, roi_mgr
                )
                res["normalized_depth"] = l_info.get("normalized_depth", 0.5)
                res["layer_desc"] = l_info.get("desc", "")
                all_results[f"{m_key}_{l_name}"] = res

        # Save summary JSON (without massive vertex arrays)
        summary_out = {}
        vertex_save_dict = {}
        for k, v in all_results.items():
            summary_out[k] = {
                "model_key": v["model_key"],
                "layer_name": v["layer_name"],
                "normalized_depth": v.get("normalized_depth", 0.5),
                "layer_desc": v.get("layer_desc", ""),
                "overall_median_r": v["overall_median_r"],
                "roi_medians": v["roi_medians"],
                "null_sample_r": v["null_sample_r"],
            }
            vertex_save_dict[k] = v["vertex_r"]

        summary_file = os.path.join(self.output_dir, f"summary_{self.subject_id}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2)

        npz_file = os.path.join(self.output_dir, f"vertex_correlations_{self.subject_id}.npz")
        np.savez_compressed(npz_file, **vertex_save_dict)

        print(f"\nPipeline complete for {self.subject_id}!")
        print(f"Summary metrics saved to: {summary_file}")
        print(f"Vertex correlations saved to: {npz_file}")
        return summary_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ridge Encoding Pipeline")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject ID")
    parser.add_argument("--data_root", type=str, default="data/raw/algonauts_2023", help="Root data folder")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to run")
    args = parser.parse_args()

    pipeline = EncodingPipeline(subject_id=args.subject, data_root=args.data_root)
    pipeline.run_full_subject(selected_models=args.models)
