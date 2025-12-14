# SVD Project

This project implements **Singular Value Decomposition (SVD)** from scratch and compares it with **scikit-learn's TruncatedSVD**. It uses the **MovieLens 100k dataset** for testing and evaluation.

## Project Structure

- `python_package/data_loading.py` → Load MovieLens dataset.
- `python_package/svd_utils.py` → Custom SVD implementation via eigen decomposition.
- `scripts/training.py` → Compute SVD (custom + scikit-learn) and save results.
- `scripts/plots.py` → Generate comparison plots (reconstruction errors and singular values).
- `data/ml-100k/` → MovieLens dataset files.
- `results/` → Saved errors and singular values, generated plots.

## Requirements

- Python 3.12+
- NumPy
- scikit-learn
- matplotlib

## How to Run

1. Activate your virtual environment:

```bash
source .venv/bin/activate        # activate your virtual environment
python training.py               # runs SVD, computes errors, saves results
python plots.py                  # plots errors and singular values

```bash


