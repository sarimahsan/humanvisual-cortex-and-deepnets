"""
Split Management for Strict Leakage Prevention.

Ensures deterministic, matched train/test partitioning across all models,
layers, and subjects. The test set is isolated prior to PCA fitting, scaling,
and Ridge alpha hyperparameter optimization.
"""

from typing import List, Tuple
import numpy as np


class SplitManager:
    """Manages cross-validation and held-out test splits."""

    def __init__(self, n_samples: int, test_ratio: float = 0.15, seed: int = 42):
        """
        Args:
            n_samples: Total number of stimuli images.
            test_ratio: Fraction of images held out for evaluation (default 15%).
            seed: Fixed random seed for complete reproducibility.
        """
        self.n_samples = n_samples
        self.test_ratio = test_ratio
        self.seed = seed

        rng = np.random.RandomState(seed)
        all_indices = np.arange(n_samples)
        rng.shuffle(all_indices)

        n_test = int(np.round(n_samples * test_ratio))
        self.test_indices = np.sort(all_indices[:n_test])
        self.train_indices = np.sort(all_indices[n_test:])

    def get_cv_folds(self, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates K-fold train/validation indices within the training set.
        Returned indices are relative to the original full array.
        """
        rng = np.random.RandomState(self.seed)
        shuffled = self.train_indices.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)

        folds = []
        for i in range(n_splits):
            val_idx = np.sort(chunks[i])
            train_idx = np.sort(np.concatenate([chunks[j] for j in range(n_splits) if j != i]))
            folds.append((train_idx, val_idx))
        return folds

    def verify_no_overlap(self) -> bool:
        """Verifies that train and test indices are strictly disjoint."""
        train_set = set(self.train_indices)
        test_set = set(self.test_indices)
        assert len(train_set.intersection(test_set)) == 0, "Data leakage detected: train and test overlap!"
        assert len(train_set) + len(test_set) == self.n_samples, "Index count mismatch!"
        return True
