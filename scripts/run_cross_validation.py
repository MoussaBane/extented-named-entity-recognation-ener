"""Run k-fold cross-validation for the BERT NER pipeline.

This script reuses the `NERDataset`, `trainer_predictions_to_labels` and
`get_word_level_alignment` helpers from `scripts/train_ner.py` to perform
4-fold cross validation and save per-fold metrics plus aggregated mean/std.

Usage example:
    python scripts/run_cross_validation.py --data-file data/full_train.conll --model-name dbmdz/bert-base-turkish-cased --output-dir results/cross_validation
"""
import argparse
import json
import os
from typing import List

import numpy as np

from ner_stats.data_utils import read_conll_bio, build_label_maps, set_global_seed
from ner_stats.evaluation import evaluate_token_classification
from ner_stats import evaluation_report

# import helpers from train_ner to avoid duplicating alignment/prediction logic
from scripts.train_ner import (
    NERDataset,
    trainer_predictions_to_labels,
    get_word_level_alignment,
)
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/full_train.conll")
    p.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir", default="results/cross_validation")
    p.add_argument("--num-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--skip-bert-train", action="store_true")
    return p.parse_args()


def make_fold_indices(n_samples: int, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, n_folds)
    return folds


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    set_global_seed(args.seed)

    ensure_dir(args.output_dir)

    tokens, labels = read_conll_bio(args.data_file)
    n = len(tokens)
    if n == 0:
        raise SystemExit(f"No sentences found in {args.data_file}")

    label2id, id2label = build_label_maps(labels)
    label_order = [id2label[i] for i in range(len(id2label))]

    folds = make_fold_indices(n, args.num_folds, args.seed)

    fold_metrics = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for i in range(args.num_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(args.num_folds) if j != i])

        train_tokens = [tokens[k] for k in train_idx]
        train_labels = [labels[k] for k in train_idx]
        val_tokens = [tokens[k] for k in val_idx]
        val_labels = [labels[k] for k in val_idx]

        fold_dir = os.path.join(args.output_dir, f"fold_{i}")
        ensure_dir(fold_dir)

        train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
        val_dataset = NERDataset(val_tokens, val_labels, tokenizer, label2id, args.max_length)

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(fold_dir, "model"),
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.eval_batch_size,
                num_train_epochs=args.num_train_epochs,
                weight_decay=0.01,
                logging_steps=50,
                save_strategy="epoch",
                eval_strategy="epoch",
                seed=args.seed,
                report_to="none",
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        if not args.skip_bert_train:
            trainer.train()
            trainer.save_model(os.path.join(fold_dir, "model"))

        # predictions
        preds = trainer_predictions_to_labels(trainer, val_dataset, id2label)

        # align gold with tokenizer first-subtoken positions
        val_gold_aligned = []
        for words, labs in zip(val_tokens, val_labels):
            aligned_words, aligned_labels = get_word_level_alignment(tokenizer, words, labs, args.max_length)
            val_gold_aligned.append(aligned_labels)

        # evaluate using existing utilities and save metrics & confusion
        metrics = evaluate_token_classification(true_labels=val_gold_aligned, predicted_labels=preds, label_order=label_order, include_o_label=False)

        # save per-fold metrics and confusion matrix
        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

        # save confusion CSV and per-class CSV via evaluation_report
        evaluation_report.generate_reports(val_gold_aligned, preds, fold_dir)

        fold_metrics.append(metrics)

    # aggregate
    import math

    def safe_get(m: dict, key: str) -> float:
        return float(m.get(key, 0.0))

    macro_f1s = [safe_get(m, "macro_f1") for m in fold_metrics]
    macro_precision = [safe_get(m, "macro_precision") for m in fold_metrics]
    macro_recall = [safe_get(m, "macro_recall") for m in fold_metrics]

    summary = {
        "num_folds": args.num_folds,
        "macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "macro_f1_std": float(np.std(macro_f1s, ddof=0)) if macro_f1s else 0.0,
        "macro_precision_mean": float(np.mean(macro_precision)) if macro_precision else 0.0,
        "macro_precision_std": float(np.std(macro_precision, ddof=0)) if macro_precision else 0.0,
        "macro_recall_mean": float(np.mean(macro_recall)) if macro_recall else 0.0,
        "macro_recall_std": float(np.std(macro_recall, ddof=0)) if macro_recall else 0.0,
    }

    with open(os.path.join(args.output_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Cross-validation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
