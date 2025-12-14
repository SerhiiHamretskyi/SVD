# plots.py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # --- Paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")

    # --- Check results exist ---
    errors_path = os.path.join(results_dir, "errors.npy")
    if not os.path.exists(errors_path):
        raise FileNotFoundError(f"{errors_path} not found! Run training.py first.")

    # --- Load results ---
    errors = np.load(os.path.join(results_dir, "errors.npy"))
    S_custom = np.load(os.path.join(results_dir, "S_custom.npy"))
    S_sklearn = np.load(os.path.join(results_dir, "S_sklearn.npy"))

    # --- Replace NaN or Inf in custom SVD to make it plottable ---
    S_custom = np.nan_to_num(S_custom, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Ensure results folder exists ---
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot reconstruction errors ---
    plt.figure(figsize=(6, 4))
    plt.bar(["Custom SVD", "scikit-learn SVD"], errors, color=["skyblue", "salmon"])
    plt.ylabel("Frobenius Norm Error")
    plt.title("Reconstruction Error Comparison")
    plt.savefig(os.path.join(results_dir, "reconstruction_error.png"))
    plt.show()

    # --- Plot top singular values ---
    plt.figure(figsize=(8, 5))
    plt.plot(S_custom[:20], "o-", label="Custom SVD")
    plt.plot(S_sklearn[:20], "s-", label="scikit-learn SVD")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Top 20 Singular Values")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "singular_values.png"))
    plt.show()

if __name__ == "__main__":
    main()
