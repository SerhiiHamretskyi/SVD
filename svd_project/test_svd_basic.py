import numpy as np
from svd_project.svd_utils import svd_via_eigendecomposition


def test_svd_reconstruction_shape():
    # Check if reconstruction has the same shape as original
    A = np.random.rand(10, 8)
    U, S, Vt = svd_via_eigendecomposition(A)
    A_reconstructed = U @ np.diag(S) @ Vt
    assert A_reconstructed.shape == A.shape


def test_svd_reconstruction_error_small():
    # Check if reconstruction error is very small
    A = np.random.rand(6, 6)
    U, S, Vt = svd_via_eigendecomposition(A)
    A_reconstructed = U @ np.diag(S) @ Vt
    error = np.linalg.norm(A - A_reconstructed)
    assert error < 1e-6


def test_singular_values_nonnegative():
    # Singular values should be non-negative
    A = np.random.rand(5, 7)
    _, S, _ = svd_via_eigendecomposition(A)
    assert np.all(S >= 0)


def test_svd_square_matrix_identity():
    # SVD of identity matrix should return original singular values
    I = np.eye(4)
    U, S, Vt = svd_via_eigendecomposition(I)
    assert np.allclose(S, np.ones(4))
    assert np.allclose(U @ np.diag(S) @ Vt, I)


def test_svd_tall_matrix_reconstruction():
    # Check reconstruction for tall matrix (more rows than columns)
    A = np.random.rand(8, 5)
    U, S, Vt = svd_via_eigendecomposition(A)
    A_reconstructed = U @ np.diag(S) @ Vt
    assert np.allclose(A, A_reconstructed, atol=1e-6)


def test_svd_wide_matrix_reconstruction():
    # Check reconstruction for wide matrix (more columns than rows)
    A = np.random.rand(5, 8)
    U, S, Vt = svd_via_eigendecomposition(A)
    A_reconstructed = U @ np.diag(S) @ Vt
    assert np.allclose(A, A_reconstructed, atol=1e-6)
