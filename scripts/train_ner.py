"""
End-to-end Turkish ENER pipeline:
1) Train/evaluate BERT token classifier.
2) Build CVA class vectors from token embeddings.
3) Evaluate BERT and CVA with shared metrics and confusion matrices.
4) Compare inference speed.
"""

import argparse
import json
import os
import sys
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.cva import classify_embedding_by_cosine_similarity, compute_cva_common_vectors
from ner_stats.data_utils import (
    build_label_maps,
    read_conll_bio,
    normalize_bio_labels,
    set_global_seed,
    validate_bio_labels,
)
from ner_stats.embeddings import TransformerEmbedder
from ner_stats.evaluation import evaluate_token_classification


ClassVectors = Mapping[str, Sequence[float] | np.ndarray]


def is_transformers_model_dir(path: str) -> bool:
    """Return True when path contains a Hugging Face config with model_type."""
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return isinstance(config, dict) and "model_type" in config


class NERDataset(Dataset):
    """Torch dataset for token classification with first-subtoken label alignment."""

    def __init__(
        self,
        tokens: Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.tokens = list(tokens)
        self.labels = list(labels)
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

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare BERT vs CVA for Turkish ENER.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train CoNLL file")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to eval CoNLL file")
    parser.add_argument("--model-name", type=str, default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--output-dir", type=str, default="outputs/bert-ner")
    parser.add_argument("--results-dir", type=str, default="results/model_comparison")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-sentences", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-bert-train",
        action="store_true",
        help="Skip training and only run prediction/evaluation with an existing model in --output-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding/model extraction (e.g., 'cpu' or 'cuda')",
    )
    return parser.parse_args()


def get_word_level_alignment(
    tokenizer,
    words: Sequence[str],
    labels: Sequence[str],
    max_length: int,
) -> Tuple[List[str], List[str]]:
    """Apply tokenizer truncation and keep only first-subtoken word positions."""
    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = cast(List[int | None], encoding.word_ids())

    aligned_words: List[str] = []
    aligned_labels: List[str] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None or not isinstance(word_id, int) or word_id == previous_word_id:
            previous_word_id = word_id
            continue

        aligned_words.append(cast(str, words[word_id]))
        aligned_labels.append(cast(str, labels[word_id]))
        previous_word_id = word_id

    return aligned_words, aligned_labels


def trainer_predictions_to_labels(
    trainer: Trainer,
    eval_dataset: Dataset,
    id2label: Dict[int, str],
) -> List[List[str]]:
    """Convert trainer predictions to word-aligned label sequences."""
    prediction_output = trainer.predict(eval_dataset)
    raw_predictions = prediction_output.predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]

    label_ids = prediction_output.label_ids
    if label_ids is None:
        raise ValueError("No label IDs found in evaluation predictions.")
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    pred_ids = np.argmax(np.asarray(raw_predictions), axis=-1)
    label_ids = np.asarray(label_ids)

    predicted_labels: List[List[str]] = []
    for pred_row, label_row in zip(pred_ids, label_ids):
        sentence_pred: List[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            sentence_pred.append(id2label[int(pred_id)])
        predicted_labels.append(sentence_pred)

    return predicted_labels


def save_confusion_csv(confusion: np.ndarray, labels: Sequence[str], out_path: str) -> None:
    frame = pd.DataFrame(confusion)
    frame.index = list(labels)
    frame.columns = list(labels)
    frame.to_csv(out_path, encoding="utf-8")


def to_jsonable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, dict):
            output[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        else:
            output[key] = value
    return output


def build_cva_class_vectors(
    train_tokens: Sequence[Sequence[str]],
    train_labels: Sequence[Sequence[str]],
    embedder: TransformerEmbedder,
    max_length: int,
) -> ClassVectors:
    """Build CVA vectors from aligned word embeddings grouped by BIO label."""
    embeddings_by_class: Dict[str, List[np.ndarray]] = {}

    for words, labels in zip(train_tokens, train_labels):
        aligned_words, aligned_labels = get_word_level_alignment(
            tokenizer=embedder.tokenizer,
            words=words,
            labels=labels,
            max_length=max_length,
        )
        _, aligned_embeddings = embedder.encode_words(aligned_words, max_length=max_length)

        for label, embedding in zip(aligned_labels, aligned_embeddings):
            embeddings_by_class.setdefault(label, []).append(
                embedding.detach().cpu().numpy()
            )

    if not embeddings_by_class:
        raise ValueError("No embeddings were extracted for CVA training.")

    stacked_embeddings_by_class: Dict[str, np.ndarray] = {
        label: np.stack(vectors, axis=0) for label, vectors in embeddings_by_class.items()
    }
    return compute_cva_common_vectors(stacked_embeddings_by_class)


def predict_cva_labels(
    tokens: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str]],
    embedder: TransformerEmbedder,
    class_vectors: ClassVectors,
    max_length: int,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """Predict word-level labels with CVA on tokenizer-aligned word positions."""
    eval_words_aligned: List[List[str]] = []
    eval_labels_aligned: List[List[str]] = []
    eval_predictions: List[List[str]] = []

    for words, gold_labels in zip(tokens, labels):
        aligned_words, aligned_gold = get_word_level_alignment(
            tokenizer=embedder.tokenizer,
            words=words,
            labels=gold_labels,
            max_length=max_length,
        )

        _, aligned_embeddings = embedder.encode_words(aligned_words, max_length=max_length)

        pred_labels: List[str] = []
        for embedding in aligned_embeddings:
            pred_labels.append(
                classify_embedding_by_cosine_similarity(
                    embedding.detach().cpu().numpy(),
                    class_vectors,
                )
            )

        eval_words_aligned.append(aligned_words)
        eval_labels_aligned.append(aligned_gold)
        eval_predictions.append(pred_labels)

    return eval_words_aligned, eval_labels_aligned, eval_predictions


def measure_bert_inference_time(
    model,
    tokenizer,
    sentences_words: Sequence[Sequence[str]],
    max_length: int,
    warmup_sentences: int,
) -> Dict[str, float]:
    """Measure sentence-level BERT token classification inference time."""

    def infer(words: Sequence[str]) -> None:
        encoded = tokenizer(
            list(words),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            model(**model_inputs)

    timings: List[float] = []
    warmup = max(0, min(warmup_sentences, len(sentences_words)))

    for words in sentences_words[:warmup]:
        infer(words)

    for words in sentences_words:
        t0 = perf_counter()
        infer(words)
        timings.append(perf_counter() - t0)

    total = float(np.sum(timings))
    avg = float(total / len(timings)) if timings else 0.0
    return {
        "average_seconds": avg,
        "total_seconds": total,
        "num_sentences": len(sentences_words),
    }


def measure_cva_inference_time(
    embedder: TransformerEmbedder,
    class_vectors: ClassVectors,
    sentences_words: Sequence[Sequence[str]],
    max_length: int,
    warmup_sentences: int,
) -> Dict[str, float]:
    """Measure sentence-level CVA inference time (embedding + cosine class lookup)."""

    def infer(words: Sequence[str]) -> None:
        _, embeddings = embedder.encode_words(words, max_length=max_length)
        for embedding in embeddings:
            classify_embedding_by_cosine_similarity(
                embedding.detach().cpu().numpy(),
                class_vectors,
            )

    timings: List[float] = []
    warmup = max(0, min(warmup_sentences, len(sentences_words)))

    for words in sentences_words[:warmup]:
        infer(words)

    for words in sentences_words:
        t0 = perf_counter()
        infer(words)
        timings.append(perf_counter() - t0)

    total = float(np.sum(timings))
    avg = float(total / len(timings)) if timings else 0.0
    return {
        "average_seconds": avg,
        "total_seconds": total,
        "num_sentences": len(sentences_words),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    train_tokens, train_labels = read_conll_bio(args.train_file)
    eval_tokens, eval_labels = read_conll_bio(args.eval_file)

    bio_warnings = validate_bio_labels([*train_labels, *eval_labels])
    train_labels = normalize_bio_labels(train_labels)
    eval_labels = normalize_bio_labels(eval_labels)
    normalized_bio_warnings = validate_bio_labels([*train_labels, *eval_labels])

    bio_warning_path = os.path.join(args.results_dir, "bio_validation_warnings.txt")
    with open(bio_warning_path, "w", encoding="utf-8") as f:
        for warning in bio_warnings:
            f.write(warning + "\n")

    normalized_bio_warning_path = os.path.join(
        args.results_dir,
        "bio_validation_warnings_after_normalization.txt",
    )
    with open(normalized_bio_warning_path, "w", encoding="utf-8") as f:
        for warning in normalized_bio_warnings:
            f.write(warning + "\n")

    label2id, id2label = build_label_maps([*train_labels, *eval_labels])
    label_order = [id2label[i] for i in range(len(id2label))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    eval_dataset = NERDataset(eval_tokens, eval_labels, tokenizer, label2id, args.max_length)

    resolved_model_source = args.model_name
    if args.skip_bert_train and is_transformers_model_dir(args.output_dir):
        resolved_model_source = args.output_dir
    elif args.skip_bert_train:
        print(
            f"[WARN] No loadable model found in {args.output_dir}; "
            f"falling back to --model-name={args.model_name}."
        )

    model = AutoModelForTokenClassification.from_pretrained(
        resolved_model_source,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    seqeval_metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        true_predictions: List[List[str]] = []
        true_labels_local: List[List[str]] = []

        for pred_row, label_row in zip(predictions, labels):
            pred_sentence: List[str] = []
            true_sentence: List[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if int(label_id) == -100:
                    continue
                pred_sentence.append(id2label[int(pred_id)])
                true_sentence.append(id2label[int(label_id)])
            true_predictions.append(pred_sentence)
            true_labels_local.append(true_sentence)

        result = cast(
            Dict[str, float],
            seqeval_metric.compute(predictions=true_predictions, references=true_labels_local) or {},
        )
        return {
            "precision": float(result.get("overall_precision", 0.0)),
            "recall": float(result.get("overall_recall", 0.0)),
            "f1": float(result.get("overall_f1", 0.0)),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
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
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if not args.skip_bert_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    embedding_model_source = args.output_dir if is_transformers_model_dir(args.output_dir) else resolved_model_source

    bert_eval_sequences_pred = trainer_predictions_to_labels(trainer, eval_dataset, id2label)

    # Align gold labels to the same first-subtoken positions used by BERT and CVA.
    eval_words_aligned: List[List[str]] = []
    eval_gold_aligned: List[List[str]] = []
    for words, labels in zip(eval_tokens, eval_labels):
        aligned_words, aligned_labels = get_word_level_alignment(
            tokenizer=tokenizer,
            words=words,
            labels=labels,
            max_length=args.max_length,
        )
        eval_words_aligned.append(aligned_words)
        eval_gold_aligned.append(aligned_labels)

    bert_metrics_with_o = evaluate_token_classification(
        true_labels=eval_gold_aligned,
        predicted_labels=bert_eval_sequences_pred,
        label_order=label_order,
        include_o_label=True,
    )
    bert_metrics_without_o = evaluate_token_classification(
        true_labels=eval_gold_aligned,
        predicted_labels=bert_eval_sequences_pred,
        label_order=label_order,
        include_o_label=False,
    )

    cva_embedder = TransformerEmbedder(model_name=embedding_model_source, device=args.device)
    cva_class_vectors: ClassVectors = build_cva_class_vectors(
        train_tokens=train_tokens,
        train_labels=train_labels,
        embedder=cva_embedder,
        max_length=args.max_length,
    )

    _, cva_gold_aligned, cva_predicted = predict_cva_labels(
        tokens=eval_tokens,
        labels=eval_labels,
        embedder=cva_embedder,
        class_vectors=cva_class_vectors,
        max_length=args.max_length,
    )

    cva_metrics_with_o = evaluate_token_classification(
        true_labels=cva_gold_aligned,
        predicted_labels=cva_predicted,
        label_order=label_order,
        include_o_label=True,
    )
    cva_metrics_without_o = evaluate_token_classification(
        true_labels=cva_gold_aligned,
        predicted_labels=cva_predicted,
        label_order=label_order,
        include_o_label=False,
    )

    bert_timing = measure_bert_inference_time(
        model=trainer.model,
        tokenizer=tokenizer,
        sentences_words=eval_words_aligned,
        max_length=args.max_length,
        warmup_sentences=args.warmup_sentences,
    )
    cva_timing = measure_cva_inference_time(
        embedder=cva_embedder,
        class_vectors=cva_class_vectors,
        sentences_words=eval_words_aligned,
        max_length=args.max_length,
        warmup_sentences=args.warmup_sentences,
    )

    bert_confusion = np.asarray(bert_metrics_with_o["confusion_matrix"], dtype=np.int64)
    bert_labels = [str(label) for label in cast(List[Any], bert_metrics_with_o["labels"])]
    cva_confusion = np.asarray(cva_metrics_with_o["confusion_matrix"], dtype=np.int64)
    cva_labels = [str(label) for label in cast(List[Any], cva_metrics_with_o["labels"])]

    save_confusion_csv(
        bert_confusion,
        bert_labels,
        os.path.join(args.results_dir, "confusion_matrix_bert_with_o.csv"),
    )
    save_confusion_csv(
        cva_confusion,
        cva_labels,
        os.path.join(args.results_dir, "confusion_matrix_cva_with_o.csv"),
    )

    with open(os.path.join(args.results_dir, "metrics_bert_with_o.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(bert_metrics_with_o), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.results_dir, "metrics_bert_without_o.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(bert_metrics_without_o), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.results_dir, "metrics_cva_with_o.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(cva_metrics_with_o), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.results_dir, "metrics_cva_without_o.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(cva_metrics_without_o), f, indent=2, ensure_ascii=False)

    summary = {
        "seed": args.seed,
        "num_train_sentences": len(train_tokens),
        "num_eval_sentences": len(eval_tokens),
        "num_labels": len(label_order),
        "bio_warning_count_raw": len(bio_warnings),
        "bio_warning_count_after_normalization": len(normalized_bio_warnings),
        "bert": {
            "macro_f1_with_o": bert_metrics_with_o["macro_f1"],
            "macro_f1_without_o": bert_metrics_without_o["macro_f1"],
            "accuracy_with_o": bert_metrics_with_o["accuracy"],
            "timing": bert_timing,
        },
        "cva": {
            "macro_f1_with_o": cva_metrics_with_o["macro_f1"],
            "macro_f1_without_o": cva_metrics_without_o["macro_f1"],
            "accuracy_with_o": cva_metrics_with_o["accuracy"],
            "timing": cva_timing,
        },
    }

    with open(os.path.join(args.results_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[INFO] Pipeline completed.")
    print(
        f"[INFO] BIO warnings (raw -> normalized): "
        f"{len(bio_warnings)} -> {len(normalized_bio_warnings)} "
        f"(saved to {bio_warning_path} and {normalized_bio_warning_path})"
    )
    print(
        f"[INFO] BERT macro-F1 (w/o O): {bert_metrics_without_o['macro_f1']:.4f} | "
        f"CVA macro-F1 (w/o O): {cva_metrics_without_o['macro_f1']:.4f}"
    )
    print(
        f"[INFO] Avg sentence inference time (BERT): {bert_timing['average_seconds']:.6f}s | "
        f"(CVA): {cva_timing['average_seconds']:.6f}s"
    )


if __name__ == "__main__":
    main()
