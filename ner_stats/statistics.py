"""
Corpus-level statistics and plotting utilities for Turkish Extended NER.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .conll_reader import read_conll_file, Sentence


@dataclass
class CorpusStats:
    total_document_folders: int
    annotated_documents: int
    unannotated_documents: int
    total_sentences: int
    sentences_with_entity: int
    entity_sentence_ratio: float
    total_tokens: int
    num_entity_labels_bio: int
    num_entity_types: int
    average_sentence_length: float


def iter_document_dirs(annotation_root: str) -> List[str]:
    """
    Return a list of document directories under the given annotation root.
    Each directory is expected to contain INITIAL_CAS.conll and optionally admin.conll.
    """
    dirs = []
    for name in os.listdir(annotation_root):
        full = os.path.join(annotation_root, name)
        if os.path.isdir(full):
            dirs.append(full)
    return sorted(dirs)


def compute_corpus_statistics(annotation_root: str) -> Tuple[CorpusStats, Counter, Counter]:
    """
    Compute global statistics over the annotated corpus.

    - Counts documents with admin.conll as "annotated"
    - Uses only admin.conll for NER statistics
    """
    doc_dirs = iter_document_dirs(annotation_root)

    total_docs = len(doc_dirs)
    annotated_docs = 0
    unannotated_docs = 0

    total_sentences = 0
    sentences_with_entity = 0
    total_tokens = 0

    label_counter: Counter = Counter()  # full BIO labels
    type_counter: Counter = Counter()   # label types without BIO prefix
    sentence_lengths: List[int] = []

    for d in doc_dirs:
        admin_path = os.path.join(d, "admin.conll")
        initial_path = os.path.join(d, "INITIAL_CAS.conll")

        has_admin = os.path.exists(admin_path)
        has_initial = os.path.exists(initial_path)

        if has_admin:
            annotated_docs += 1
        elif has_initial:
            unannotated_docs += 1

        if not has_admin:
            # We only use annotated files for statistics
            continue

        sentences: List[Sentence] = read_conll_file(admin_path)
        total_sentences += len(sentences)

        for sent in sentences:
            sent_len = len(sent)
            if sent_len == 0:
                continue

            sentence_lengths.append(sent_len)
            has_entity = False

            for token, label in sent:
                total_tokens += 1
                if label != "O":
                    has_entity = True
                    label_counter[label] += 1

                    etype = label
                    if etype.startswith("B-") or etype.startswith("I-"):
                        etype = etype[2:]
                    type_counter[etype] += 1

            if has_entity:
                sentences_with_entity += 1

    entity_sentence_ratio = (
        sentences_with_entity / total_sentences if total_sentences > 0 else 0.0
    )
    avg_sentence_len = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )

    stats = CorpusStats(
        total_document_folders=total_docs,
        annotated_documents=annotated_docs,
        unannotated_documents=unannotated_docs,
        total_sentences=total_sentences,
        sentences_with_entity=sentences_with_entity,
        entity_sentence_ratio=entity_sentence_ratio,
        total_tokens=total_tokens,
        num_entity_labels_bio=len(label_counter),
        num_entity_types=len(type_counter),
        average_sentence_length=avg_sentence_len,
    )

    return stats, label_counter, type_counter


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_stats_json(stats: CorpusStats, out_path: str) -> None:
    """Save corpus statistics as JSON."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)


def save_counter_csv(counter: Counter, out_path: str, column_name: str) -> None:
    """Save a Counter as a CSV file."""
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=[column_name, "count"])
    df.to_csv(out_path, index=False, encoding="utf-8")


def plot_top_entity_types(type_counter: Counter, out_path: str, top_n: int = 20) -> None:
    """
    Save a bar plot of the top N entity types (without BIO prefixes).
    """
    top = type_counter.most_common(top_n)
    if not top:
        return

    labels, counts = zip(*top)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Entity Types")
    plt.xlabel("Entity Type")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_entity_sentence_ratio(stats: CorpusStats, out_path: str) -> None:
    """
    Plot a simple bar chart for sentence-level annotation coverage:
    - sentences with at least one entity
    - sentences without entities
    """
    with_entity = stats.sentences_with_entity
    without_entity = stats.total_sentences - stats.sentences_with_entity

    labels = ["With Entity", "Without Entity"]
    counts = [with_entity, without_entity]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("Sentence-Level Entity Coverage")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sentence_length_histogram(sentence_lengths: List[int], out_path: str) -> None:
    """
    Plot a histogram of sentence lengths (in tokens).
    """
    if not sentence_lengths:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(sentence_lengths, bins=30)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Sentence Length (tokens)")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def collect_sentence_lengths(annotation_root: str) -> List[int]:
    sentence_lengths: List[int] = []
    doc_dirs = iter_document_dirs(annotation_root)

    for d in doc_dirs:
        admin_path = os.path.join(d, "admin.conll")
        if not os.path.exists(admin_path):
            continue

        sentences: List[Sentence] = read_conll_file(admin_path)
        for sent in sentences:
            if sent:
                sentence_lengths.append(len(sent))

    return sentence_lengths

