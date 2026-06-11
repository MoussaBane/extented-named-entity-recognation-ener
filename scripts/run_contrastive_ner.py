"""Contrastive NER: combined Supervised Contrastive + Cross-Entropy training.

Training adds a Supervised Contrastive Loss (SupCon) on top of the standard
token-classification cross-entropy loss.  A projection head maps first-subtoken
BERT embeddings to a low-dimensional contrastive space where same-label tokens
are attracted and different-label tokens are repelled.  The classification head
operates directly on the BERT hidden states.

The combined loss is:

    L = (1 − λ) · CE(classification_head(h)) + λ · SupCon(projection_head(h))

where h is the first-subtoken BERT embedding for each valid word and λ is
controlled by ``--contrastive-lambda``.

After training, the script re-evaluates with the same evaluation utilities used
throughout the project and saves a summary JSON that can be compared with the
standard BERT-only result.

Usage:
    python scripts/run_contrastive_ner.py \\
        --train-file data/full_train.conll \\
        --eval-file  data/full_eval.conll  \\
        --output-dir results/contrastive_ner_full
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import TokenClassifierOutput

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.contrastive import ProjectionHead, SupConLoss
from ner_stats.data_utils import (
    build_label_maps,
    normalize_bio_labels,
    read_conll_bio,
    set_global_seed,
)
from ner_stats.evaluation import evaluate_token_classification
from scripts.train_ner import NERDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BERT NER with Supervised Contrastive auxiliary loss."
    )
    p.add_argument("--train-file",          default="data/full_train.conll")
    p.add_argument("--eval-file",           default="data/full_eval.conll")
    p.add_argument("--model-name",          default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir",          default="results/contrastive_ner")
    p.add_argument("--max-length",          type=int,   default=256)
    p.add_argument("--num-epochs",          type=float, default=3.0)
    p.add_argument("--batch-size",          type=int,   default=8)
    p.add_argument("--lr",                  type=float, default=2e-5)
    p.add_argument("--warmup-steps",        type=int,   default=100)
    p.add_argument("--contrastive-lambda",  type=float, default=0.1,
                   help="Weight of SupCon loss; 0 = pure CE, 1 = pure SupCon.")
    p.add_argument("--temperature",         type=float, default=0.07,
                   help="SupCon softmax temperature.")
    p.add_argument("--projection-dim",      type=int,   default=128,
                   help="Contrastive projection head output dimension.")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--device",              type=str,   default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContrastiveNERModel(nn.Module):
    """BERT backbone with a classification head and a contrastive projection head.

    The classification head is used for NER prediction.
    The projection head is used only during training to compute the SupCon loss.

    Parameters
    ----------
    model_name : str
        Hugging Face model id or local checkpoint path.
    num_labels : int
        Number of NER output classes.
    projection_dim : int
        Output dimension of the projection head (contrastive space).
    dropout : float
        Dropout before the classification head.  Default: 0.1.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.bert        = AutoModel.from_pretrained(model_name)
        hidden_size      = self.bert.config.hidden_size
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(hidden_size, num_labels)
        self.projection  = ProjectionHead(hidden_size, output_dim=projection_dim)
        self.config      = self.bert.config
        self.config.id2label  = id2label
        self.config.label2id  = label2id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        """Standard forward; loss uses CE only (for compatibility with eval helpers)."""
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # (B, L, H)
        logits = self.classifier(self.dropout(hidden))   # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=(hidden,))


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def _first_subtoken_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect hidden states and labels at first-subtoken positions (label != -100).

    Returns (embeddings, valid_labels) both of shape (M, H) and (M,).
    """
    mask   = labels.view(-1) != -100          # (B*L,)
    h_flat = hidden.view(-1, hidden.shape[-1]) # (B*L, H)
    l_flat = labels.view(-1)                   # (B*L,)
    return h_flat[mask], l_flat[mask]


def train_epoch(
    model: ContrastiveNERModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    supcon_fn: SupConLoss,
    contrastive_lambda: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return (mean total loss, mean CE loss, mean SupCon loss) for the epoch."""
    model.train()
    total_loss_sum = ce_loss_sum = sc_loss_sum = 0.0

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        labels  = batch["labels"]

        out     = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        ce_loss = out.loss                                     # scalar
        hidden  = out.hidden_states[0]                        # (B, L, H)

        # Supervised contrastive loss on first-subtoken embeddings.
        valid_h, valid_l = _first_subtoken_hidden(hidden, labels)
        proj    = model.projection(valid_h)                   # (M, proj_dim), L2-normalised
        sc_loss = supcon_fn(proj, valid_l)

        total   = (1.0 - contrastive_lambda) * ce_loss + contrastive_lambda * sc_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss_sum += total.item()
        ce_loss_sum    += ce_loss.item()
        sc_loss_sum    += sc_loss.item()

    n = max(len(loader), 1)
    return total_loss_sum / n, ce_loss_sum / n, sc_loss_sum / n


@torch.no_grad()
def evaluate_model(
    model: ContrastiveNERModel,
    loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    label_order: List[str],
) -> Dict:
    model.eval()
    all_gold, all_preds = [], []

    for batch in loader:
        batch     = {k: v.to(device) for k, v in batch.items()}
        out       = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pred_ids  = out.logits.argmax(dim=-1)    # (B, L)
        label_ids = batch["labels"]               # (B, L)

        for b in range(pred_ids.shape[0]):
            preds, golds = [], []
            for l in range(pred_ids.shape[1]):
                if int(label_ids[b, l].item()) != -100:
                    preds.append(id2label[int(pred_ids[b, l].item())])
                    golds.append(id2label[int(label_ids[b, l].item())])
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    eval_ds  = NERDataset(eval_tokens,  eval_labels,  tokenizer, label2id, args.max_length)

    from transformers import DataCollatorForTokenClassification
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collator)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = ContrastiveNERModel(
        model_name=args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        projection_dim=args.projection_dim,
    ).to(device)

    supcon_fn  = SupConLoss(temperature=args.temperature).to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_epochs = int(args.num_epochs)
    total_steps = num_epochs * len(train_loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_log = []
    for epoch in range(1, num_epochs + 1):
        total_l, ce_l, sc_l = train_epoch(
            model, train_loader, optimizer, scheduler,
            supcon_fn, args.contrastive_lambda, device,
        )
        metrics = evaluate_model(model, eval_loader, device, id2label, label_order)
        row = {
            "epoch": epoch,
            "total_loss": total_l,
            "ce_loss":    ce_l,
            "supcon_loss":sc_l,
            "macro_f1":   float(metrics["macro_f1"]),
            "accuracy":   float(metrics["accuracy"]),
        }
        epoch_log.append(row)
        print(
            f"[INFO] Epoch {epoch}/{num_epochs}  "
            f"total={total_l:.4f}  CE={ce_l:.4f}  SupCon={sc_l:.4f}  "
            f"macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}"
        )

    # Save model checkpoint (backbone + heads).
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label2id":   label2id,
            "id2label":   id2label,
            "args":       vars(args),
        },
        os.path.join(args.output_dir, "contrastive_ner_model.pt"),
    )

    # Save BERT backbone separately so it is loadable by transformers.
    bert_ckpt_dir = os.path.join(args.output_dir, "bert_backbone")
    model.bert.save_pretrained(bert_ckpt_dir)
    tokenizer.save_pretrained(bert_ckpt_dir)

    final_metrics = evaluate_model(model, eval_loader, device, id2label, label_order)
    summary = {
        "model":               "ContrastiveBERT",
        "backbone":            args.model_name,
        "contrastive_lambda":  args.contrastive_lambda,
        "temperature":         args.temperature,
        "projection_dim":      args.projection_dim,
        "macro_f1":            float(final_metrics["macro_f1"]),
        "accuracy":            float(final_metrics["accuracy"]),
        "epoch_log":           epoch_log,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n[INFO] ContrastiveBERT — final macro-F1 (w/o O): {final_metrics['macro_f1']:.4f}")
    print(f"[INFO] Results saved to {args.output_dir}")
    print(f"[INFO] BERT backbone saved to {bert_ckpt_dir} (re-usable with --skip-bert-train).")


if __name__ == "__main__":
    main()
