"""
High-Performance Closed-Form Ridge Regression Solver (GPU / CPU).

Uses SVD / Eigendecomposition to solve regularized linear encoding models
across thousands of vertices simultaneously across a log-grid of alphas.
Performs inner K-fold cross-validation to select the optimal alpha per vertex chunk.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np


def fast_pearson_r(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes Pearson correlation along dimension 0 (images) for each vertex (dimension 1).
    Inputs: shape (N, V)
    Returns: shape (V,)
    """
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)

    cov = (y_true_centered * y_pred_centered).sum(dim=0)
    std_true = torch.sqrt((y_true_centered**2).sum(dim=0) + eps)
    std_pred = torch.sqrt((y_pred_centered**2).sum(dim=0) + eps)

    r = cov / (std_true * std_pred)
    return torch.clamp(r, -1.0, 1.0)


class FastRidgeCV:
    """
    Closed-form Ridge regression with inner K-fold Cross-Validation.
    Runs efficiently on CUDA or CPU with chunked vertex blocks.
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv_folds: int = 5,
        device: str = "cuda",
    ):
        if alphas is None:
            self.alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
        else:
            self.alphas = list(alphas)

        self.cv_folds = cv_folds
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

    def _solve_ridge_svd(
        self,
        V: torch.Tensor,
        S_sq: torch.Tensor,
        XtY: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Solves W = (X^T X + alpha I)^(-1) X^T Y in O(P) via eigendecomposition.
        V: (P, P), S_sq: (P,), XtY: (P, V)
        Returns: W of shape (P, V)
        """
        # (S^2 + alpha)^(-1)
        inv_diag = 1.0 / (S_sq + alpha)  # (P,)
        # V * diag(inv_diag) * V^T * XtY
        vt_xty = V.t() @ XtY  # (P, V)
        scaled = inv_diag.unsqueeze(1) * vt_xty  # (P, V)
        W = V @ scaled  # (P, V)
        return W

    def fit_predict_chunk(
        self,
        X_train_np: np.ndarray,
        Y_train_np: np.ndarray,
        X_test_np: np.ndarray,
        Y_test_np: np.ndarray,
        inner_folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits Ridge regression on train data with inner-CV alpha selection,
        then evaluates on held-out test data.

        Returns:
            test_correlations: shape (V,) - Pearson r per vertex on test set
            best_alphas: shape (V,) - Selected alpha per vertex
            Y_test_pred: shape (N_test, V) - Predicted test responses
        """
        X_tr = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        Y_tr = torch.tensor(Y_train_np, dtype=torch.float32, device=self.device)
        X_te = torch.tensor(X_test_np, dtype=torch.float32, device=self.device)
        Y_te = torch.tensor(Y_test_np, dtype=torch.float32, device=self.device)

        n_train, n_features = X_tr.shape
        n_vertices = Y_tr.shape[1]

        # 1. Inner Cross-Validation to select optimal alpha per vertex
        if inner_folds is None:
            # Deterministic K-fold split
            perm = torch.randperm(n_train)
            fold_size = n_train // self.cv_folds
            inner_folds_t = []
            for f in range(self.cv_folds):
                val_idx = perm[f * fold_size : (f + 1) * fold_size]
                mask = torch.ones(n_train, dtype=torch.bool)
                mask[val_idx] = False
                train_idx = torch.where(mask)[0]
                inner_folds_t.append((train_idx, val_idx))
        else:
            inner_folds_t = [
                (torch.tensor(tr, dtype=torch.long, device=self.device),
                 torch.tensor(val, dtype=torch.long, device=self.device))
                for tr, val in inner_folds
            ]

        # Accumulate validation performance: (n_alphas, n_vertices)
        cv_scores = torch.zeros((len(self.alphas), n_vertices), device=self.device)

        for tr_idx, val_idx in inner_folds_t:
            X_fold_tr = X_tr[tr_idx]
            Y_fold_tr = Y_tr[tr_idx]
            X_fold_val = X_tr[val_idx]
            Y_fold_val = Y_tr[val_idx]

            # Precompute XtX eigendecomposition on fold training set
            XtX = X_fold_tr.t() @ X_fold_tr  # (P, P)
            S_sq, V = torch.linalg.eigh(XtX)
            XtY = X_fold_tr.t() @ Y_fold_tr  # (P, V)

            for a_idx, alpha in enumerate(self.alphas):
                W = self._solve_ridge_svd(V, S_sq, XtY, alpha)
                Y_pred_val = X_fold_val @ W
                r_val = fast_pearson_r(Y_fold_val, Y_pred_val)
                cv_scores[a_idx] += r_val

        # Find best alpha index per vertex
        best_alpha_idx = torch.argmax(cv_scores, dim=0)  # (V,)
        best_alphas = torch.tensor(
            [self.alphas[i] for i in best_alpha_idx.cpu().numpy()], device=self.device
        )

        # 2. Refit full model using the best alpha for each vertex
        # Precompute full XtX eigendecomposition
        XtX_full = X_tr.t() @ X_tr
        S_sq_full, V_full = torch.linalg.eigh(XtX_full)
        XtY_full = X_tr.t() @ Y_tr

        # Assemble final weights W_full: (P, V)
        W_full = torch.zeros((n_features, n_vertices), device=self.device)
        for a_idx, alpha in enumerate(self.alphas):
            vertex_mask = best_alpha_idx == a_idx
            if vertex_mask.any():
                XtY_sub = XtY_full[:, vertex_mask]
                W_sub = self._solve_ridge_svd(V_full, S_sq_full, XtY_sub, alpha)
                W_full[:, vertex_mask] = W_sub

        # 3. Predict on held-out test set
        Y_test_pred = X_te @ W_full
        test_correlations = fast_pearson_r(Y_te, Y_test_pred)

        return (
            test_correlations.detach().cpu().numpy(),
            best_alphas.detach().cpu().numpy(),
            Y_test_pred.detach().cpu().numpy(),
        )
