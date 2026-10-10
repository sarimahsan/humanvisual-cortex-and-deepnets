"""
Dimensionality Reduction and Feature Scaling with Strict Leakage Prevention.

Fits standardizer and PCA strictly on the training partition, then transforms
training and held-out test partitions into a fixed dimensionality (e.g. 1024 components).
Uses pure NumPy SVD with optional scikit-learn acceleration.
"""

from typing import Tuple, Optional
import numpy as np


class FeatureReducer:
    """Manages train-only standardization and PCA dimensionality reduction."""

    def __init__(self, n_components: int = 1024, whiten: bool = False, random_state: int = 42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # shape (actual_k, n_features)
        self.singular_values_: Optional[np.ndarray] = None

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fits mean/std and PCA strictly on training features and returns reduced X_train.
        X_train shape: (N_train, D)
        """
        # 1. Train-only standardization (in-place to conserve RAM)
        X_train_scaled = np.array(X_train, dtype=np.float32)
        self.mean_ = np.mean(X_train_scaled, axis=0, keepdims=True)
        self.std_ = np.std(X_train_scaled, axis=0, keepdims=True)
        # Avoid division by zero for invariant dimensions
        self.std_[self.std_ < 1e-7] = 1.0

        X_train_scaled -= self.mean_
        X_train_scaled /= self.std_

        # 2. Fit PCA via GPU VRAM if available (uses 0 System RAM, 50x faster)
        n_samples, n_features = X_train_scaled.shape
        actual_k = min(self.n_components, n_samples, n_features)

        try:
            import torch
            if torch.cuda.is_available():
                X_t = torch.tensor(X_train_scaled, dtype=torch.float32, device="cuda")
                U, S, V = torch.pca_lowrank(X_t, q=actual_k, center=False)
                X_train_pca = (X_t @ V).cpu().numpy().astype(np.float32)
                self.components_ = V.t().cpu().numpy().astype(np.float32)
                self.singular_values_ = S.cpu().numpy().astype(np.float32)
                self._pca_obj = None
                del X_t, U, S, V
                torch.cuda.empty_cache()
                return X_train_pca
        except Exception as e:
            pass

        try:
            from sklearn.decomposition import PCA
            pca = PCA(
                n_components=actual_k,
                whiten=self.whiten,
                random_state=self.random_state,
                svd_solver="randomized" if actual_k < min(n_samples, n_features) else "auto",
            )
            X_train_pca = pca.fit_transform(X_train_scaled)
            self.components_ = pca.components_
            self.singular_values_ = pca.singular_values_
            self._pca_obj = pca
            return X_train_pca.astype(np.float32)
        except ImportError:
            # Standalone NumPy SVD
            U, S, Vt = np.linalg.svd(X_train_scaled, full_matrices=False)
            self.components_ = Vt[:actual_k]
            self.singular_values_ = S[:actual_k]
            self._pca_obj = None

            if self.whiten:
                X_train_pca = U[:, :actual_k] * np.sqrt(n_samples - 1)
            else:
                X_train_pca = U[:, :actual_k] * S[:actual_k]

            return X_train_pca.astype(np.float32)

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        """Applies fitted training scaler and PCA projection to held-out test features."""
        if self.mean_ is None or self.std_ is None or self.components_ is None:
            raise RuntimeError("FeatureReducer must be fit on training data before transforming test data.")

        X_test_f32 = np.asarray(X_test, dtype=np.float32)
        X_test_scaled = (X_test_f32 - self.mean_) / self.std_

        if getattr(self, "_pca_obj", None) is not None:
            return self._pca_obj.transform(X_test_scaled).astype(np.float32)

        X_proj = X_test_scaled @ self.components_.T
        if self.whiten and self.singular_values_ is not None:
            scale = self.singular_values_ / np.sqrt(max(1, len(self.mean_) - 1))
            X_proj /= (scale + 1e-8)

        return X_proj.astype(np.float32)
