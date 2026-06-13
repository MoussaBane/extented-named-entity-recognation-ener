"""Train and evaluate the Attention NER model with explicit Q/K/V matrices.

Usage
-----
python scripts/run_attention_ner.py \\
    --train-file data/train.conll \\
    --eval-file  data/eval.conll \\
    --output-dir results/attention_ner \\
    --num-train-epochs 5 \\
    --d-head 256

Add --freeze-bert to train only the attention head + classifier (fast mode).

Outputs (in --output-dir)
-------------------------
metrics_with_o.json        — macro P/R/F1, accuracy, per-class breakdown
metrics_without_o.json     — same excluding the O label
confusion_matrix.csv       — full token-level confusion matrix
per_label_report.csv       — precision / recall / F1 / support per BIO label
comparison_summary.json    — concise numbers for thesis table
model/                     — saved PyTorch model + tokenizer
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.attention_ner import AttentionNERModel
from ner_stats.data_utils import (
    build_label_maps,
    normalize_bio_labels,
    read_conll_bio,
    set_global_seed,
)
from ner_stats.evaluation import evaluate_token_classification


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """Token-classification dataset with first-subtoken label alignment."""

    def __init__(
        self,
        tokens: List[List[str]],
        labels: List[List[str]],
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
        words = self.tokens[idx]
        word_labels = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = encoding.word_ids()

        aligned: List[int] = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(self.label2id[word_labels[wid]])
            else:
                aligned.append(-100)
            prev = wid

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        labels[i, :n] = item["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict(
    model: AttentionNERModel,
    loader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
) -> List[List[str]]:
    model.eval()
    all_preds: List[List[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = out.logits.argmax(-1).cpu()

            for pred_row, label_row in zip(pred_ids, labels):
                sentence_preds: List[str] = []
                for pid, lid in zip(pred_row.tolist(), label_row.tolist()):
                    if lid == -100:
                        continue
                    sentence_preds.append(id2label[pid])
                all_preds.append(sentence_preds)
    return all_preds


def align_gold_labels(
    tokens: List[List[str]],
    labels: List[List[str]],
    tokenizer,
    max_length: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    aligned_tokens_out: List[List[str]] = []
    aligned_labels_out: List[List[str]] = []
    for words, word_labels in zip(tokens, labels):
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = enc.word_ids()
        a_tok: List[str] = []
        a_lab: List[str] = []
        prev = None
        for wid in word_ids:
            if wid is None or wid == prev:
                prev = wid
                continue
            a_tok.append(words[wid])
            a_lab.append(word_labels[wid])
            prev = wid
        aligned_tokens_out.append(a_tok)
        aligned_labels_out.append(a_lab)
    return aligned_tokens_out, aligned_labels_out


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_confusion_csv(
    confusion: np.ndarray,
    labels: List[str],
    path: str,
) -> None:
    df = pd.DataFrame(confusion, index=labels, columns=labels)
    df.to_csv(path, encoding="utf-8")


def save_per_label_csv(per_class: Dict[str, object], path: str) -> None:
    rows = []
    for label, metrics in per_class.items():  # type: ignore[union-attr]
        m = metrics  # type: ignore[assignment]
        rows.append({
            "label": label,
            "precision": round(float(m["precision"]), 4),
            "recall": round(float(m["recall"]), 4),
            "f1": round(float(m["f1"]), 4),
            "support": int(m["support"]),
        })
    pd.DataFrame(rows).sort_values("label").to_csv(path, index=False, encoding="utf-8")


def to_jsonable(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}  # type: ignore[union-attr]
    return obj


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AttentionNERModel with learned Q/K/V matrices."
    )
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir", default="results/attention_ner")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--d-head", type=int, default=256,
                   help="Dimension of Q/K/V attention projections")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-train-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--hidden-dropout", type=float, default=0.1)
    p.add_argument("--freeze-bert", action="store_true",
                   help="Freeze BERT weights; train only Q/K/V + classifier")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; load saved model from output-dir/model")
    p.add_argument("--o-weight", type=float, default=1.0,
                   help="Loss weight for the O label (< 1.0 to penalise predicting everything as O)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device(args.device)

    # ---- Data ---------------------------------------------------------------
    train_tokens, train_labels = read_conll_bio(args.train_file)
    eval_tokens, eval_labels = read_conll_bio(args.eval_file)

    train_labels = normalize_bio_labels(train_labels)
    eval_labels = normalize_bio_labels(eval_labels)

    label2id, id2label = build_label_maps([*train_labels, *eval_labels])
    label_order = [id2label[i] for i in range(len(id2label))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    eval_ds = NERDataset(eval_tokens, eval_labels, tokenizer, label2id, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ---- Model --------------------------------------------------------------
    saved_config_path = os.path.join(model_dir, "config.json")
    if args.skip_train and os.path.isfile(saved_config_path):
        with open(saved_config_path, encoding="utf-8") as f:
            saved_cfg = json.load(f)
        d_head = saved_cfg.get("d_head", args.d_head)
        freeze_bert = saved_cfg.get("freeze_bert", args.freeze_bert)
        bert_source = saved_cfg.get("model_name", args.model_name)
    else:
        d_head = args.d_head
        freeze_bert = args.freeze_bert
        bert_source = args.model_name

    model = AttentionNERModel(
        bert_model_name=bert_source,
        num_labels=len(label2id),
        d_head=d_head,
        attn_dropout=args.attn_dropout,
        hidden_dropout=args.hidden_dropout,
        freeze_bert=freeze_bert,
    ).to(device)

    weights_path = os.path.join(model_dir, "attention_ner_weights.pt")
    if args.skip_train and os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[INFO] Loaded weights from {weights_path}")

    # ---- Training -----------------------------------------------------------
    if not args.skip_train:
        total_steps = len(train_loader) * args.num_train_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Build per-class loss weights (lower weight for O to fight class imbalance)
        class_weights = torch.ones(len(label2id), device=device)
        if args.o_weight != 1.0 and "O" in label2id:
            class_weights[label2id["O"]] = args.o_weight

        print(
            f"[INFO] Training AttentionNERModel | "
            f"freeze_bert={args.freeze_bert} | d_head={args.d_head} | "
            f"epochs={args.num_train_epochs} | steps={total_steps} | "
            f"o_weight={args.o_weight}"
        )

        for epoch in range(1, args.num_train_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                logits = out.logits
                labels_gpu = batch["labels"].to(device)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, len(label2id)),
                    labels_gpu.view(-1),
                    weight=class_weights,
                    ignore_index=-100,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch}/{args.num_train_epochs} — avg loss: {avg_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(model_dir, "attention_ner_weights.pt"))
        tokenizer.save_pretrained(model_dir)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({
                "model_name": args.model_name,
                "num_labels": len(label2id),
                "d_head": args.d_head,
                "freeze_bert": args.freeze_bert,
                "label2id": label2id,
            }, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Model saved to {model_dir}")

    # ---- Evaluation ---------------------------------------------------------
    preds = predict(model, eval_loader, id2label, device)

    _, eval_gold_aligned = align_gold_labels(
        eval_tokens, eval_labels, tokenizer, args.max_length
    )

    metrics_with_o = evaluate_token_classification(
        true_labels=eval_gold_aligned,
        predicted_labels=preds,
        label_order=label_order,
        include_o_label=True,
    )
    metrics_without_o = evaluate_token_classification(
        true_labels=eval_gold_aligned,
        predicted_labels=preds,
        label_order=label_order,
        include_o_label=False,
    )

    # Save metrics JSON
    for name, metrics in [
        ("metrics_with_o.json", metrics_with_o),
        ("metrics_without_o.json", metrics_without_o),
    ]:
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metrics), f, indent=2, ensure_ascii=False)

    # Save confusion matrix CSV
    confusion = np.asarray(metrics_with_o["confusion_matrix"], dtype=np.int64)
    labels_list = [str(l) for l in metrics_with_o["labels"]]  # type: ignore[union-attr]
    save_confusion_csv(
        confusion,
        labels_list,
        os.path.join(args.output_dir, "confusion_matrix.csv"),
    )

    # Save per-label CSV
    save_per_label_csv(
        metrics_with_o["per_class"],  # type: ignore[arg-type]
        os.path.join(args.output_dir, "per_label_report.csv"),
    )

    # Compact summary for thesis table
    summary = {
        "model": "AttentionNER",
        "bert_base": args.model_name,
        "d_head": args.d_head,
        "freeze_bert": args.freeze_bert,
        "o_weight": args.o_weight,
        "num_train_sentences": len(train_tokens),
        "num_eval_sentences": len(eval_tokens),
        "num_labels": len(label_order),
        "with_O": {
            "accuracy": metrics_with_o["accuracy"],
            "macro_precision": metrics_with_o["macro_precision"],
            "macro_recall": metrics_with_o["macro_recall"],
            "macro_f1": metrics_with_o["macro_f1"],
        },
        "without_O": {
            "accuracy": metrics_without_o["accuracy"],
            "macro_precision": metrics_without_o["macro_precision"],
            "macro_recall": metrics_without_o["macro_recall"],
            "macro_f1": metrics_without_o["macro_f1"],
        },
    }
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[INFO] AttentionNER macro-F1 (w/o O): {metrics_without_o['macro_f1']:.4f} | "
        f"accuracy: {metrics_with_o['accuracy']:.4f}"
    )
    print(f"[INFO] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
