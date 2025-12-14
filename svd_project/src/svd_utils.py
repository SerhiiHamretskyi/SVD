

import numpy as np

def svd_via_eigendecomposition(A: np.ndarray):


    ATA = A.T @ A


    eigvals, V = np.linalg.eigh(ATA)


    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]


    singular_vals = np.sqrt(np.maximum(eigvals, 0))


    U = (A @ V) / singular_vals

    return U, singular_vals, V.T