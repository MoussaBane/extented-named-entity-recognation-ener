"""Shared data and reproducibility utilities for token classification."""

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_conll_bio(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read a CoNLL file into token and BIO label sequences."""
    all_tokens: List[List[str]] = []
    all_labels: List[List[str]] = []

    sent_tokens: List[str] = []
    sent_labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if sent_tokens:
                    all_tokens.append(sent_tokens)
                    all_labels.append(sent_labels)
                    sent_tokens = []
                    sent_labels = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            sent_tokens.append(parts[0])
            sent_labels.append(parts[-1])

    if sent_tokens:
        all_tokens.append(sent_tokens)
        all_labels.append(sent_labels)

    return all_tokens, all_labels


def validate_bio_sequence(labels: Sequence[str]) -> List[str]:
    """
    Validate a single BIO sequence and return warnings for irregular transitions.

    This is a permissive validator: it reports issues but does not modify labels.
    """
    warnings: List[str] = []
    previous_type = ""
    inside_entity = False

    for i, label in enumerate(labels):
        if label == "O":
            inside_entity = False
            previous_type = ""
            continue

        if "-" not in label:
            warnings.append(f"Non-BIO tag '{label}' at position {i}.")
            inside_entity = False
            previous_type = ""
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix not in {"B", "I"} or not entity_type:
            warnings.append(f"Malformed BIO tag '{label}' at position {i}.")
            inside_entity = False
            previous_type = ""
            continue

        if prefix == "B":
            inside_entity = True
            previous_type = entity_type
            continue

        if not inside_entity:
            warnings.append(
                f"I-tag '{label}' at position {i} starts without a preceding B-tag."
            )
        elif previous_type != entity_type:
            warnings.append(
                f"I-tag '{label}' at position {i} switches type from '{previous_type}'."
            )

        inside_entity = True
        previous_type = entity_type

    return warnings


def validate_bio_labels(label_sequences: Iterable[Sequence[str]]) -> List[str]:
    """Validate all BIO sequences and return aggregated warning strings."""
    warnings: List[str] = []
    for sent_idx, sequence in enumerate(label_sequences):
        seq_warnings = validate_bio_sequence(sequence)
        for warning in seq_warnings:
            warnings.append(f"sentence={sent_idx}: {warning}")
    return warnings


def normalize_bio_sequence(labels: Sequence[str]) -> List[str]:
    """
    Normalize a BIO sequence by converting invalid I-tags into B-tags.

    This preserves the entity type while making the sequence valid for training
    and evaluation with standard BIO tooling.
    """
    normalized: List[str] = []
    previous_type = ""
    inside_entity = False

    for label in labels:
        if label == "O":
            normalized.append(label)
            inside_entity = False
            previous_type = ""
            continue

        if "-" not in label:
            normalized.append(label)
            inside_entity = False
            previous_type = ""
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            normalized.append(label)
            inside_entity = True
            previous_type = entity_type
            continue

        if prefix == "I" and inside_entity and previous_type == entity_type:
            normalized.append(label)
        else:
            normalized.append(f"B-{entity_type}")
            inside_entity = True
        previous_type = entity_type

    return normalized


def normalize_bio_labels(label_sequences: Iterable[Sequence[str]]) -> List[List[str]]:
    """Normalize all BIO sequences using normalize_bio_sequence."""
    return [normalize_bio_sequence(sequence) for sequence in label_sequences]


def build_label_maps(label_sequences: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create stable label2id/id2label mappings from BIO labels."""
    labels = sorted({label for seq in label_sequences for label in seq})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label
