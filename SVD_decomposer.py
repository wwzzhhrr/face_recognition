import numpy as np


class SVDDecomposer:
    def __init__(self):
        self.U = None      # Left singular vectors (n_samples × n_samples)
        self.S = None      # Singular values (min(n_samples, n_features))
        self.VT = None     # Right singular vectors (n_features × n_features)

    def fit(self, X: np.ndarray):
        """
        Perform Singular Value Decomposition on centered data matrix X.
        X should be centered (mean zero).
        """
        # Compute the full SVD
        self.U, self.S, self.VT = np.linalg.svd(X, full_matrices=False)

    def transform(self, k: int) -> np.ndarray:
        """
        Return the top-k right singular vectors (V^T[:k, :]), i.e., the principal directions.
        """
        if self.VT is None:
            raise RuntimeError("SVD not computed. Call fit() first.")
        return self.VT[:k, :]
