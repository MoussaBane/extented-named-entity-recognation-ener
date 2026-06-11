"""Character-level + BERT hybrid NER (CharBERT).

Architecture
------------
For each word in a sentence:
  1. Take the first-subtoken hidden state from the BERT encoder.
  2. Run the word's character sequence through a character CNN.
  3. Concatenate the two vectors and classify with a linear head.

This gives the model access to morphological surface cues (useful for
agglutinative Turkish) in addition to contextual BERT representations.

Usage:
    python scripts/run_char_ner.py \\
        --train-file data/full_train.conll \\
        --eval-file  data/full_eval.conll  \\
        --output-dir results/char_ner_full
"""

import argparse
import json
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import TokenClassifierOutput

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.char_features import CharCNNEncoder, build_char_vocab, words_to_char_ids
from ner_stats.data_utils import (
    build_label_maps,
    normalize_bio_labels,
    read_conll_bio,
    set_global_seed,
)
from ner_stats.evaluation import evaluate_token_classification


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT + CharCNN hybrid NER model.")
    p.add_argument("--train-file",  default="data/full_train.conll")
    p.add_argument("--eval-file",   default="data/full_eval.conll")
    p.add_argument("--model-name",  default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir",  default="results/char_ner")
    p.add_argument("--max-length",  type=int,   default=256)
    p.add_argument("--max-chars",   type=int,   default=32,   help="Max characters per word.")
    p.add_argument("--char-emb-dim",type=int,   default=30)
    p.add_argument("--num-filters", type=int,   default=50,   help="CNN filters per kernel size.")
    p.add_argument("--num-epochs",  type=float, default=3.0)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup-steps",type=int,   default=100)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset and collation
# ---------------------------------------------------------------------------

class CharNERDataset(Dataset):
    """Dataset that provides both subword token IDs (for BERT) and
    character IDs (for the char CNN) aligned to the same word positions.
    """

    def __init__(
        self,
        tokens: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        label2id: Dict[str, int],
        char2id: Dict[str, int],
        max_length: int,
        max_chars: int,
    ) -> None:
        self.items: List[Dict] = []
        for words, word_labels in zip(tokens, labels):
            encoding = tokenizer(
                list(words),
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            word_ids = encoding.word_ids()

            # First-subtoken index per word (within the truncated sequence).
            first_subtok: Dict[int, int] = {}
            for tok_idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id not in first_subtok:
                    first_subtok[word_id] = tok_idx

            aligned_wids = sorted(first_subtok.keys())
            first_subtok_indices = [first_subtok[w] for w in aligned_wids]
            aligned_labels       = [label2id[word_labels[w]] for w in aligned_wids]
            aligned_words        = [words[w] for w in aligned_wids]

            char_ids = words_to_char_ids(aligned_words, char2id, max_chars)

            self.items.append({
                "input_ids":           encoding["input_ids"],
                "attention_mask":      encoding["attention_mask"],
                "word_first_subtok":   first_subtok_indices,
                "word_char_ids":       char_ids,
                "word_labels":         aligned_labels,
                "n_words":             len(aligned_wids),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def char_ner_collate(batch: List[Dict], max_chars: int) -> Dict[str, torch.Tensor]:
    """Pad a mixed batch of BERT token sequences and word-level char sequences."""
    max_tok_len  = max(len(item["input_ids"])   for item in batch)
    max_words    = max(item["n_words"]           for item in batch)

    input_ids_list, attn_mask_list = [], []
    first_subtok_list, char_ids_list, word_labels_list = [], [], []

    for item in batch:
        tok_pad  = max_tok_len - len(item["input_ids"])
        word_pad = max_words   - item["n_words"]

        input_ids_list.append(item["input_ids"]      + [0] * tok_pad)
        attn_mask_list.append(item["attention_mask"] + [0] * tok_pad)

        first_subtok_list.append(item["word_first_subtok"] + [-1]     * word_pad)
        word_labels_list.append( item["word_labels"]        + [-100]   * word_pad)
        char_ids_list.append(    item["word_char_ids"]       + [[0] * max_chars] * word_pad)

    return {
        "input_ids":         torch.tensor(input_ids_list,   dtype=torch.long),
        "attention_mask":    torch.tensor(attn_mask_list,   dtype=torch.long),
        "word_first_subtok": torch.tensor(first_subtok_list,dtype=torch.long),
        "word_char_ids":     torch.tensor(char_ids_list,    dtype=torch.long),
        "word_labels":       torch.tensor(word_labels_list, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharBERTForNER(nn.Module):
    """BERT first-subtoken embeddings + CharCNN → linear token classifier.

    Parameters
    ----------
    bert_model_name : str
        Hugging Face model id or local path.
    char_vocab_size : int
        Size of character vocabulary.
    num_labels : int
        Number of NER label classes.
    char_emb_dim, num_filters : int
        Hyper-parameters forwarded to :class:`CharCNNEncoder`.
    dropout : float
        Classifier dropout.  Default: 0.1.
    """

    def __init__(
        self,
        bert_model_name: str,
        char_vocab_size: int,
        num_labels: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        char_emb_dim: int = 30,
        num_filters: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.bert       = AutoModel.from_pretrained(bert_model_name)
        bert_hidden     = self.bert.config.hidden_size
        self.char_cnn   = CharCNNEncoder(
            vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            num_filters=num_filters,
        )
        combined_dim = bert_hidden + self.char_cnn.output_dim
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, num_labels)
        # Store mappings on config for compatibility with evaluation helpers.
        self.config = self.bert.config
        self.config.id2label = id2label
        self.config.label2id = label2id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_first_subtok: torch.Tensor,
        word_char_ids: torch.Tensor,
        word_labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        """
        input_ids, attention_mask : (B, L)  subword token tensors
        word_first_subtok          : (B, W)  first subtoken index per word; -1 = padding
        word_char_ids              : (B, W, C) character IDs
        word_labels                : (B, W)  integer labels; -100 = padding
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden   = bert_out.last_hidden_state   # (B, L, H)
        B, L, H  = hidden.shape
        W        = word_char_ids.shape[1]

        # Extract first-subtoken embeddings with a batched gather.
        idx      = word_first_subtok.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)  # (B, W, H)
        word_emb = torch.gather(hidden, 1, idx)                                     # (B, W, H)
        # Zero out padding word positions.
        valid    = (word_first_subtok >= 0).unsqueeze(-1).float()                   # (B, W, 1)
        word_emb = word_emb * valid

        # Character CNN.
        char_flat  = word_char_ids.view(B * W, -1)          # (B*W, C)
        char_repr  = self.char_cnn(char_flat).view(B, W, -1) # (B, W, char_dim)

        combined = self.dropout(torch.cat([word_emb, char_repr], dim=-1))  # (B, W, H+char_dim)
        logits   = self.classifier(combined)                                # (B, W, num_labels)

        loss = None
        if word_labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                word_labels.view(-1),
                ignore_index=-100,
            )

        return TokenClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: CharBERTForNER,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_first_subtok=batch["word_first_subtok"],
            word_char_ids=batch["word_char_ids"],
            word_labels=batch["word_labels"],
        )
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += out.loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(
    model: CharBERTForNER,
    loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    label_order: List[str],
) -> Dict:
    model.eval()
    all_gold, all_preds = [], []
    for batch in loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        out      = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_first_subtok=batch["word_first_subtok"],
            word_char_ids=batch["word_char_ids"],
        )
        pred_ids   = out.logits.argmax(dim=-1)    # (B, W)
        word_labs  = batch["word_labels"]          # (B, W)

        for b in range(pred_ids.shape[0]):
            preds, golds = [], []
            for w in range(pred_ids.shape[1]):
                if int(word_labs[b, w].item()) != -100:
                    preds.append(id2label[int(pred_ids[b, w].item())])
                    golds.append(id2label[int(word_labs[b, w].item())])
            all_preds.append(preds)
            all_gold.append(golds)

    return evaluate_token_classification(
        true_labels=all_gold,
        predicted_labels=all_preds,
        label_order=label_order,
        include_o_label=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_tokens, train_labels_raw = read_conll_bio(args.train_file)
    eval_tokens,  eval_labels_raw  = read_conll_bio(args.eval_file)

    train_labels = normalize_bio_labels(train_labels_raw)
    eval_labels  = normalize_bio_labels(eval_labels_raw)

    label2id, id2label = build_label_maps([*train_labels, *eval_labels])
    label_order = [id2label[i] for i in range(len(id2label))]

    char2id = build_char_vocab(train_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = CharNERDataset(
        train_tokens, train_labels, tokenizer, label2id, char2id,
        args.max_length, args.max_chars,
    )
    eval_ds = CharNERDataset(
        eval_tokens, eval_labels, tokenizer, label2id, char2id,
        args.max_length, args.max_chars,
    )

    collate_fn = partial(char_ner_collate, max_chars=args.max_chars)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CharBERTForNER(
        bert_model_name=args.model_name,
        char_vocab_size=len(char2id),
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        char_emb_dim=args.char_emb_dim,
        num_filters=args.num_filters,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_epochs     = int(args.num_epochs)
    total_steps    = num_epochs * len(train_loader)
    scheduler      = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_log = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        metrics    = evaluate_model(model, eval_loader, device, id2label, label_order)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
        }
        epoch_log.append(row)
        print(
            f"[INFO] Epoch {epoch}/{num_epochs}  "
            f"loss={train_loss:.4f}  "
            f"macro-F1={metrics['macro_f1']:.4f}  "
            f"acc={metrics['accuracy']:.4f}"
        )

    # Save model artefacts.
    torch.save(
        {
            "state_dict":    model.state_dict(),
            "char2id":       char2id,
            "label2id":      label2id,
            "id2label":      id2label,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "char_ner_model.pt"),
    )

    # Final evaluation.
    final_metrics = evaluate_model(model, eval_loader, device, id2label, label_order)
    summary = {
        "model": "CharBERT",
        "backbone": args.model_name,
        "char_emb_dim": args.char_emb_dim,
        "num_filters": args.num_filters,
        "macro_f1": float(final_metrics["macro_f1"]),
        "accuracy": float(final_metrics["accuracy"]),
        "epoch_log": epoch_log,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n[INFO] CharBERT — final macro-F1 (w/o O): {final_metrics['macro_f1']:.4f}")
    print(f"[INFO] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
