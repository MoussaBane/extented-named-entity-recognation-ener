"""
Train a Hugging Face Transformers NER model from CoNLL BIO-formatted data.

Expected file format per line:
TOKEN ... LABEL

Sentences are separated by blank lines and LABEL is assumed to be in the last column.
"""

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


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


class NERDataset(Dataset):
    """Torch dataset for token classification with aligned labels."""

    def __init__(
        self,
        tokens: Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        words = list(self.tokens[idx])
        word_labels = list(self.labels[idx])

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()
        aligned_labels: List[int] = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label2id[word_labels[word_idx]])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

        return item


def build_label_maps(label_sequences: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create stable label2id/id2label mappings from BIO labels."""
    labels = sorted({label for seq in label_sequences for label in seq})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def build_token_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """
    Build token-level confusion matrix, excluding ignored labels (-100).

    Rows correspond to true labels and columns to predicted labels.
    """
    matrix = np.zeros((num_labels, num_labels), dtype=np.int64)

    for pred_row, label_row in zip(predictions, labels):
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            matrix[int(label_id), int(pred_id)] += 1

    return matrix


def print_confusion_matrix(confusion: np.ndarray, id2label: Dict[int, str]) -> None:
    """Display confusion matrix in a compact text table."""
    ordered_labels = [id2label[i] for i in range(len(id2label))]

    cell_width = max(7, max(len(label) for label in ordered_labels) + 2)
    row_label_width = max(10, max(len(label) for label in ordered_labels) + 2)

    header = " " * row_label_width + "".join(
        f"{label:>{cell_width}}" for label in ordered_labels
    )
    print("\nToken-level confusion matrix (rows=true, cols=pred):")
    print(header)

    for row_idx, true_label in enumerate(ordered_labels):
        row_values = "".join(
            f"{int(confusion[row_idx, col_idx]):>{cell_width}}"
            for col_idx in range(len(ordered_labels))
        )
        print(f"{true_label:>{row_label_width}}{row_values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BERT NER model with Hugging Face Trainer.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train CoNLL file")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to eval CoNLL file")
    parser.add_argument("--model-name", type=str, default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--output-dir", type=str, default="outputs/bert-ner")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_tokens, train_labels = read_conll_bio(args.train_file)
    eval_tokens, eval_labels = read_conll_bio(args.eval_file)

    label2id, id2label = build_label_maps([*train_labels, *eval_labels])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = NERDataset(
        tokens=train_tokens,
        labels=train_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
    )
    eval_dataset = NERDataset(
        tokens=eval_tokens,
        labels=eval_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval_metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []

        for pred_row, label_row in zip(predictions, labels):
            sentence_preds: List[str] = []
            sentence_labels: List[str] = []

            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                sentence_preds.append(id2label[int(pred_id)])
                sentence_labels.append(id2label[int(label_id)])

            true_predictions.append(sentence_preds)
            true_labels.append(sentence_labels)

        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        if not results:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    prediction_output = trainer.predict(eval_dataset)

    raw_predictions = prediction_output.predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]

    pred_ids = np.argmax(raw_predictions, axis=-1)
    label_ids = prediction_output.label_ids
    if label_ids is None:
        raise ValueError("No label IDs found in evaluation predictions.")
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    label_ids = np.asarray(label_ids)
    pred_ids = np.asarray(pred_ids)

    confusion = build_token_confusion_matrix(
        predictions=pred_ids,
        labels=label_ids,
        num_labels=len(label2id),
    )

    print("Evaluation metrics:")
    print(f"precision: {eval_metrics.get('eval_precision', 0.0):.4f}")
    print(f"recall:    {eval_metrics.get('eval_recall', 0.0):.4f}")
    print(f"f1:        {eval_metrics.get('eval_f1', 0.0):.4f}")
    print_confusion_matrix(confusion, id2label)


if __name__ == "__main__":
    main()
