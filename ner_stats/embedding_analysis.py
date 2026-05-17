"""Pipeline for extracting and aggregating entity embeddings.

Functions to extract token/word-aligned embeddings from CoNLL data,
aggregate by label and export results for visualization and evaluation.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os
import json
import time

import numpy as np
from .embeddings import TransformerEmbedder, get_word_aligned_embeddings
from .cva import compute_class_mean_vectors, compute_cva_common_vectors


def _bio_to_type(label: str) -> str:
    if label == "O":
        return "O"
    if "-" in label:
        return label.split("-", 1)[1]
    return label


def extract_token_embeddings_from_dataset(
    token_seqs: Sequence[Sequence[str]],
    label_seqs: Sequence[Sequence[str]],
    embedder: TransformerEmbedder,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
) -> List[Dict]:
    """Extract word-aligned embeddings for each token and attach label type.

    Returns list of records: {sent_idx, token_idx, token, label, label_type, embedding}
    Embeddings are numpy arrays on CPU.
    """
    records: List[Dict] = []
    total = 0
    start = time.time()

    for sent_idx, (tokens, labels) in enumerate(zip(token_seqs, label_seqs)):
        if max_samples is not None and total >= max_samples:
            break

        words, embeddings = embedder.encode_words(tokens)
        # embeddings is torch tensor on device; convert to cpu numpy
        if len(words) == 0:
            continue

        emb_np = embeddings.detach().cpu().numpy()

        # words align with tokens after first-subtoken alignment; keep indices
        for i, (tok, lbl) in enumerate(zip(words, labels)):
            rec = {
                "sent_idx": sent_idx,
                "token_idx": i,
                "token": tok,
                "label": lbl,
                "label_type": _bio_to_type(lbl),
                "embedding": emb_np[i].tolist(),
            }
            records.append(rec)
            total += 1
            if max_samples is not None and total >= max_samples:
                break

    duration = time.time() - start
    meta = {"extracted": total, "duration_s": duration}
    return records, meta


def aggregate_embeddings_by_label(records: Iterable[Dict]) -> Dict[str, np.ndarray]:
    buckets: Dict[str, List[np.ndarray]] = {}
    for rec in records:
        lbl = rec.get("label_type", "O")
        vec = np.asarray(rec["embedding"], dtype=np.float64)
        buckets.setdefault(lbl, []).append(vec)

    return {k: np.vstack(v) if len(v) > 0 else np.zeros((0, 0)) for k, v in buckets.items()}


def export_embeddings(records: Iterable[Dict], out_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    csv_path = out_prefix + ".csv"
    npy_path = out_prefix + ".npy"

    # save CSV with metadata and flattened vector
    with open(csv_path, "w", encoding="utf-8") as f:
        # header
        f.write("sent_idx,token_idx,token,label,label_type,embedding\n")
        for rec in records:
            emb = " ".join(map(str, rec["embedding"]))
            line = f"{rec['sent_idx']},{rec['token_idx']},{rec['token']},{rec['label']},{rec['label_type']},\"{emb}\"\n"
            f.write(line)

    # save numpy array of embeddings only
    embs = [np.asarray(rec["embedding"], dtype=np.float64) for rec in records]
    if embs:
        stacked = np.vstack(embs)
        np.save(npy_path, stacked)
    else:
        np.save(npy_path, np.zeros((0, 0)))


def build_class_vectors_from_records(records: Iterable[Dict], use_cva: bool = False) -> Dict[str, np.ndarray]:
    buckets = {}
    for rec in records:
        lbl = rec.get("label_type", "O")
        vec = np.asarray(rec["embedding"], dtype=np.float64)
        buckets.setdefault(lbl, []).append(vec)

    if use_cva:
        return compute_cva_common_vectors(buckets)
    return compute_class_mean_vectors(buckets)
