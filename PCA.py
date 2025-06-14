from SVD_decomposer import SVDDecomposer
import numpy as np


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.components = None  # Top-k eigenvectors (principal directions)
        self.svd = SVDDecomposer()

    def fit(self, X: np.ndarray):
        """
        Fit PCA by performing SVD on the centered data.
        """
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Decompose with SVD
        self.svd.fit(X_centered)

        # Step 3: Take top-k right singular vectors as principal components
        self.components = self.svd.transform(self.n_components).T  # shape: (n_features, k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new data into the principal component space.
        """
        if self.components is None:
            raise RuntimeError("PCA model is not fitted yet.")

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)
