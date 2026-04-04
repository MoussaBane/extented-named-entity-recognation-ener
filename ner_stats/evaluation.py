"""Evaluation utilities for token classification labels."""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _normalize_label_sequences(
    labels: Sequence[Sequence[str]] | Sequence[str],
) -> List[List[str]]:
    """Normalize flat or nested labels into a list of label sequences."""
    if len(labels) == 0:
        return []

    first_item = labels[0]
    if isinstance(first_item, str):
        return [list(labels)]  # type: ignore[list-item]

    return [list(sequence) for sequence in labels]  # type: ignore[list-item]


def evaluate_token_classification(
    true_labels: Sequence[Sequence[str]] | Sequence[str],
    predicted_labels: Sequence[Sequence[str]] | Sequence[str],
    label_order: Sequence[str] | None = None,
    ignore_labels: Sequence[str] | None = None,
) -> Dict[str, object]:
    """
    Compare predicted labels with true labels and compute token-level metrics.

    Args:
        true_labels: Ground-truth labels, either flat or sentence-grouped.
        predicted_labels: Predicted labels with the same structure as true_labels.
        label_order: Optional explicit class order for the confusion matrix.
        ignore_labels: Optional labels to exclude from evaluation.

    Returns:
        Dictionary with confusion_matrix, labels, precision, recall, and f1.
    """
    true_sequences = _normalize_label_sequences(true_labels)
    predicted_sequences = _normalize_label_sequences(predicted_labels)

    if len(true_sequences) != len(predicted_sequences):
        raise ValueError("true_labels and predicted_labels must contain the same number of sequences.")

    ignore_set = set(ignore_labels or [])

    flattened_true: List[str] = []
    flattened_pred: List[str] = []

    for true_sequence, predicted_sequence in zip(true_sequences, predicted_sequences):
        if len(true_sequence) != len(predicted_sequence):
            raise ValueError("Each true/predicted sequence pair must have the same length.")

        for true_label, predicted_label in zip(true_sequence, predicted_sequence):
            if true_label in ignore_set or predicted_label in ignore_set:
                continue

            flattened_true.append(true_label)
            flattened_pred.append(predicted_label)

    if label_order is None:
        labels = sorted(set(flattened_true) | set(flattened_pred))
    else:
        labels = list(label_order)

    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for true_label, predicted_label in zip(flattened_true, flattened_pred):
        if true_label not in label_to_index or predicted_label not in label_to_index:
            continue
        confusion_matrix[label_to_index[true_label], label_to_index[predicted_label]] += 1

    true_positive = int(np.trace(confusion_matrix))
    false_positive = int(confusion_matrix.sum(axis=0).sum() - true_positive)
    false_negative = int(confusion_matrix.sum(axis=1).sum() - true_positive)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
    recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(flattened_true),
    }