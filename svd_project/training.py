# scripts/training.py
import sys
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Add project root to sys.path so svd_project can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svd_project.data_loading import load_movielens_100k
from svd_project.svd_utils import svd_via_eigendecomposition

def reconstruction_error(A, U, S, Vt):
    """Compute Frobenius norm reconstruction error."""
    return np.linalg.norm(A - U @ np.diag(S) @ Vt, ord='fro')

def main():
    # 1. Load ML-100k dataset
    data_path = "data/ml-100k/u.data"
    A = load_movielens_100k(data_path)
    print(f"Data loaded: shape={A.shape}")

    # 2. Compute SVD using custom implementation
    U, S, Vt = svd_via_eigendecomposition(A)
    error_custom = reconstruction_error(A, U, S, Vt)
    print(f"Custom SVD reconstruction error: {error_custom:.4f}")

    # 3. Compute SVD using scikit-learn
    k = min(A.shape)
    svd_sk = TruncatedSVD(n_components=k)
    A_sk = svd_sk.fit_transform(A)
    V_sk = svd_sk.components_
    A_recon_sk = A_sk @ V_sk
    error_sklearn = np.linalg.norm(A - A_recon_sk, ord='fro')
    print(f"scikit-learn TruncatedSVD reconstruction error: {error_sklearn:.4f}")

    # 4. Top singular values
    print("Top 10 singular values (custom SVD):", S[:10])
    print("Top 10 singular values (sklearn SVD):", svd_sk.singular_values_[:10])

    # 5. Save results for plotting
    os.makedirs("results", exist_ok=True)
    np.save("results/errors.npy", np.array([error_custom, error_sklearn]))
    np.save("results/S_custom.npy", S)
    np.save("results/S_sklearn.npy", svd_sk.singular_values_)

if __name__ == "__main__":
    main()