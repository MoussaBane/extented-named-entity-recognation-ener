"""Class-vector utilities: mean vectors and Common Vector Approach (CVA)."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .embeddings import get_last_hidden_tokens_and_embeddings


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


def classify_embedding_by_cosine_similarity(
    embedding: Sequence[float] | np.ndarray,
    class_vectors: Dict[str, Sequence[float] | np.ndarray],
) -> str:
    """
    Classify a word embedding by cosine similarity against class vectors.

    Args:
        embedding: A single embedding vector with shape (hidden_size,).
        class_vectors: Mapping {class_label: vector} where each vector has shape
            (hidden_size,) or is array-like.

    Returns:
        The class label with the highest cosine similarity.
    """
    query = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if query.size == 0:
        raise ValueError("The input embedding must not be empty.")

    query_norm = np.linalg.norm(query)
    if np.isclose(query_norm, 0.0):
        raise ValueError("The input embedding must have non-zero norm.")

    best_label = None
    best_score = -np.inf

    for class_label, vector in class_vectors.items():
        candidate = np.asarray(vector, dtype=np.float64).reshape(-1)
        if candidate.size != query.size:
            raise ValueError(
                f"Dimension mismatch for class '{class_label}': "
                f"expected {query.size}, got {candidate.size}."
            )

        candidate_norm = np.linalg.norm(candidate)
        if np.isclose(candidate_norm, 0.0):
            continue

        score = float(np.dot(query, candidate) / (query_norm * candidate_norm))
        if score > best_score:
            best_score = score
            best_label = class_label

    if best_label is None:
        raise ValueError("No valid class vectors with non-zero norm were provided.")

    return best_label


def classify_sentence_tokens_by_cosine_similarity(
    sentence: str,
    class_vectors: Dict[str, Sequence[float] | np.ndarray],
    model_name: str = "dbmdz/bert-base-turkish-cased",
) -> List[Tuple[str, str]]:
    """
    Predict a class label for each token in a sentence using cosine similarity.

    Returns:
        A list of (token, predicted_label) pairs in token order.
    """
    tokens, token_embeddings = get_last_hidden_tokens_and_embeddings(
        sentence,
        model_name=model_name,
    )

    predictions: List[Tuple[str, str]] = []
    for token, token_embedding in zip(tokens, token_embeddings):
        predicted_label = classify_embedding_by_cosine_similarity(
            token_embedding.detach().cpu().numpy(),
            class_vectors,
        )
        predictions.append((token, predicted_label))

    return predictions
