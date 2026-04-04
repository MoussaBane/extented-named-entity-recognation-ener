"""Class-vector utilities: mean vectors and Common Vector Approach (CVA)."""

from typing import Dict, Iterable, Sequence

import numpy as np


def _to_2d_array(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert class vectors to a 2D numpy array with shape (n_samples, hidden_size)."""
    arr = np.asarray(vectors, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("Each class entry must be 1D or 2D array-like embeddings.")

    if arr.shape[0] == 0:
        raise ValueError("Each class must contain at least one embedding vector.")

    return arr


def compute_class_mean_vectors(
    embeddings_by_class: Dict[str, Sequence[Sequence[float]] | np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Compute mean embedding vector for each class.

    Args:
        embeddings_by_class: Mapping {class_label: embeddings}, where embeddings
            are shaped (n_samples, hidden_size) or a single vector (hidden_size,).

    Returns:
        Dictionary {class_label: mean_vector} with vectors of shape (hidden_size,).
    """
    means: Dict[str, np.ndarray] = {}

    for class_label, class_vectors in embeddings_by_class.items():
        x = _to_2d_array(class_vectors)
        means[class_label] = x.mean(axis=0)

    return means


def compute_cva_common_vectors(
    embeddings_by_class: Dict[str, Sequence[Sequence[float]] | np.ndarray],
    svd_tol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Compute Common Vector Approach (CVA) class vectors using SVD.

    For each class:
    1) Build matrix X (n_samples, hidden_size)
    2) Compute class mean m
    3) Remove within-class variance using centered matrix A = X - m
    4) Obtain within-class subspace basis via SVD(A)
    5) Common vector is m projected onto the orthogonal complement of this subspace

    Returns:
        Dictionary {class_label: common_vector} with vectors of shape (hidden_size,).
    """
    common_vectors: Dict[str, np.ndarray] = {}

    for class_label, class_vectors in embeddings_by_class.items():
        x = _to_2d_array(class_vectors)
        mean_vec = x.mean(axis=0)

        centered = x - mean_vec

        # If no within-class variation exists, the common vector is the mean.
        if x.shape[0] == 1 or np.allclose(centered, 0.0):
            common_vectors[class_label] = mean_vec
            continue

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(np.sum(singular_values > svd_tol))

        if rank == 0:
            common_vectors[class_label] = mean_vec
            continue

        basis = vt[:rank].T
        projected_within = basis @ (basis.T @ mean_vec)
        common_vec = mean_vec - projected_within

        common_vectors[class_label] = common_vec

    return common_vectors
