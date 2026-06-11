"""CRF baseline with k-fold evaluation using the same splits as BERT CV.

Trains a linear-chain CRF on token-level features and evaluates with
the same token-level evaluation utilities. Saves per-fold metrics and
confusion matrices under `--output-dir`.
"""
import argparse
import json
import os
from typing import List

import numpy as np
from sklearn_crfsuite import CRF

from ner_stats.data_utils import read_conll_bio, set_global_seed
from ner_stats.evaluation import evaluate_token_classification
from ner_stats import evaluation_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/full_train.conll")
    p.add_argument("--output-dir", default="results/crf_baseline")
    p.add_argument("--num-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def word2features(sent: List[str], i: int) -> dict:
    word = sent[i]
    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        prev = sent[i - 1]
        features.update({"-1:word.lower()": prev.lower(), "-1:istitle()": prev.istitle()})
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        nxt = sent[i + 1]
        features.update({"+1:word.lower()": nxt.lower(), "+1:istitle()": nxt.istitle()})
    else:
        features["EOS"] = True

    return features


def sent2features(sent: List[str]) -> List[dict]:
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(labels: List[str]) -> List[str]:
    return labels


def make_fold_indices(n_samples: int, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, n_folds)
    return folds


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj):
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def main():
    args = parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)

    tokens, labels = read_conll_bio(args.data_file)
    n = len(tokens)
    if n == 0:
        raise SystemExit("No sentences found in dataset")

    folds = make_fold_indices(n, args.num_folds, args.seed)
    fold_metrics = []

    for i in range(args.num_folds):
        val_idx = folds[i]
        train_idx = [idx for j in range(args.num_folds) for idx in folds[j] if j != i]

        X_train = [sent2features(tokens[k]) for k in train_idx]
        y_train = [sent2labels(labels[k]) for k in train_idx]
        X_val = [sent2features(tokens[k]) for k in val_idx]
        y_val = [sent2labels(labels[k]) for k in val_idx]

        crf = CRF(algorithm="lbfgs", max_iterations=100)
        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_val)

        # evaluation expects sequences of equal-length label lists
        metrics = evaluate_token_classification(true_labels=y_val, predicted_labels=y_pred, include_o_label=False)

        fold_dir = os.path.join(args.output_dir, f"fold_{i}")
        ensure_dir(fold_dir)

        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metrics), f, indent=2, ensure_ascii=False)

        # use evaluation_report to save confusion matrix plot & CSV
        evaluation_report.generate_reports(y_val, y_pred, fold_dir)

        fold_metrics.append(metrics)

    # aggregate — build rich folds+aggregates structure (mirrors run_cross_validation.py)
    folds_data = []
    for fold_idx, m in enumerate(fold_metrics):
        folds_data.append({
            "fold": fold_idx,
            "accuracy": float(m.get("accuracy", 0.0)),
            "macro_f1": float(m.get("macro_f1", 0.0)),
            "macro_precision": float(m.get("macro_precision", 0.0)),
            "macro_recall": float(m.get("macro_recall", 0.0)),
            "support": float(m.get("support", 0)),
        })

    def _agg(key):
        vals = [f[key] for f in folds_data]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}

    summary = {
        "num_folds_found": len(folds_data),
        "folds": folds_data,
        "aggregates": {
            "accuracy": _agg("accuracy"),
            "macro_f1": _agg("macro_f1"),
            "macro_precision": _agg("macro_precision"),
            "macro_recall": _agg("macro_recall"),
            "support": _agg("support"),
        },
        # flat aliases for backward compatibility
        "num_folds": args.num_folds,
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds_data])),
        "macro_f1_std": float(np.std([f["macro_f1"] for f in folds_data], ddof=0)),
    }

    with open(os.path.join(args.output_dir, "crf_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"CRF cross-validation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
