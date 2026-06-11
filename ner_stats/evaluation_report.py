"""Produce classification reports, confusion matrices and save results.
"""
from typing import Dict, List, Sequence
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import evaluate_token_classification


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _to_serializable(x):
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: _to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_serializable(v) for v in x]
        return x

    obj_ser = _to_serializable(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj_ser, f, ensure_ascii=False, indent=2)


def save_metrics_summary(metrics: Dict, path: str):
    # metrics expected to contain accuracy, precision, recall, f1, per_class
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv

    header = ["metric", "value"]
    rows = [
        ("accuracy", metrics.get("accuracy", 0.0)),
        ("macro_f1", metrics.get("macro_f1", metrics.get("f1", 0.0))),
        ("macro_precision", metrics.get("macro_precision", metrics.get("precision", 0.0))),
        ("macro_recall", metrics.get("macro_recall", metrics.get("recall", 0.0))),
        ("support", metrics.get("support", 0)),
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def save_confusion_matrix_plot(confusion: np.ndarray, labels: Sequence[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion.shape[1]), yticks=np.arange(confusion.shape[0]), xticklabels=labels, yticklabels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], "d"), ha="center", va="center", color="white" if confusion[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_top_n_confusion_matrix(metrics: Dict, top_n: int, out_dir: str) -> None:
    """Save confusion matrix and plot restricted to the top-N most frequent labels (by support)."""
    per_class = metrics.get("per_class", {})
    all_labels = metrics.get("labels", [])
    full_cm = np.asarray(metrics["confusion_matrix"])

    # rank labels by support descending, filtering to those present in label list
    label_support = [(lbl, per_class.get(lbl, {}).get("support", 0)) for lbl in all_labels]
    label_support.sort(key=lambda x: -x[1])
    top_labels = [lbl for lbl, _ in label_support[:top_n]]

    # extract sub-matrix indices
    label_index = {lbl: i for i, lbl in enumerate(all_labels)}
    indices = [label_index[lbl] for lbl in top_labels if lbl in label_index]
    if not indices:
        return

    sub_cm = full_cm[np.ix_(indices, indices)]

    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, f"confusion_matrix_top{top_n}.csv"),
        sub_cm,
        fmt="%d",
        delimiter=",",
        header=",".join(top_labels),
        comments="",
    )
    save_confusion_matrix_plot(
        sub_cm, top_labels,
        os.path.join(out_dir, f"confusion_matrix_top{top_n}.png"),
    )


def generate_reports(true_seqs: Sequence[Sequence[str]], pred_seqs: Sequence[Sequence[str]], out_dir: str, top_n: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    metrics = evaluate_token_classification(true_seqs, pred_seqs)
    save_json(metrics, os.path.join(out_dir, "classification_report.json"))
    # save full confusion matrix CSV
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), metrics["confusion_matrix"], fmt="%d", delimiter=",")
    # save human-friendly per-class CSV
    import csv
    per_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(per_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for lbl, vals in metrics["per_class"].items():
            writer.writerow([lbl, vals["precision"], vals["recall"], vals["f1"], vals["support"]])

    save_metrics_summary(metrics, os.path.join(out_dir, "metrics_summary.csv"))
    save_confusion_matrix_plot(metrics["confusion_matrix"], metrics["labels"], os.path.join(out_dir, "confusion_matrix.png"))
    # save top-N frequent labels confusion matrix (Req C)
    if top_n and len(metrics.get("labels", [])) > top_n:
        save_top_n_confusion_matrix(metrics, top_n, out_dir)
    return metrics
