"""
Generate attention heatmaps from a fine-tuned BERT model for entity visualization.

Extracts attention weights from all 12 layers × 12 heads and generates:
- Per-sentence attention heatmaps for sentences containing entities
- Layer-wise average attention patterns
- Head-wise attention statistics
- Entity-to-entity vs entity-to-context attention breakdown

Usage:
    python scripts/generate_attention_heatmaps.py \
        --model-dir outputs/bert-ner-full/checkpoint-390 \
        --eval-conll data/full_eval.conll \
        --output-dir results/attention_heatmaps \
        --num-examples 20
"""

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

from ner_stats.conll_reader import read_conll_file
from ner_stats.data_utils import set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Generate BERT attention heatmaps for entity analysis")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--eval-conll", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-examples", type=int, default=20, help="Number of example sentences to visualize")
    p.add_argument("--layer", type=int, default=None, help="Specific layer to visualize (default: all)")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_attention_weights(model, tokenizer, sentence, max_length, device):
    """
    Extract attention weights for a single sentence.

    Returns:
        tokens: list of subword tokens
        word_ids: list mapping subword idx -> word idx
        attentions: list of 12 tensors, each (12, seq_len, seq_len)
        word_labels: list of word-level labels
    """
    tokens_words = [t for t, _ in sentence]
    labels_words = [l for _, l in sentence]

    encoding = tokenizer(
        tokens_words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    word_ids = encoding.word_ids()
    input_ids = encoding["input_ids"].to(device)
    attn_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )

    # outputs.attentions: tuple of 12 tensors, each (1, 12, seq_len, seq_len)
    attentions = [a[0].cpu().numpy() for a in outputs.attentions]  # list of (12, seq_len, seq_len)

    # Decode subword tokens
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return subword_tokens, word_ids, attentions, labels_words, tokens_words


def plot_attention_heatmap(attention_matrix, tokens, title, save_path,
                           entity_mask=None, figsize=(14, 12)):
    """Plot a single attention head heatmap with optional entity highlighting."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=attention_matrix.max())

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    short_tokens = [t[:10] for t in tokens]
    ax.set_xticklabels(short_tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(short_tokens, fontsize=7)

    if entity_mask is not None:
        for i, is_entity in enumerate(entity_mask):
            if is_entity:
                ax.axhline(y=i - 0.5, color="red", linewidth=0.5, alpha=0.5)
                ax.axvline(x=i - 0.5, color="red", linewidth=0.5, alpha=0.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def compute_attention_statistics(records):
    """
    Compute aggregate statistics over all extracted attention records.

    Returns dict with:
        entity_to_entity: average attention from entity tokens to entity tokens
        entity_to_context: average attention from entity tokens to non-entity tokens
        context_to_entity: average attention from non-entity tokens to entity tokens
        per_layer_entropy: average attention entropy per layer
    """
    stats = defaultdict(list)

    for r in records:
        for layer_idx, attn in enumerate(r["attentions"]):
            # attn: (12, seq_len, seq_len)
            entity_idx = r["entity_subword_indices"]
            all_idx = list(range(attn.shape[1]))
            non_entity_idx = [i for i in all_idx if i not in set(entity_idx)]

            if not entity_idx or not non_entity_idx:
                continue

            # Mean over heads
            mean_attn = attn.mean(axis=0)  # (seq_len, seq_len)

            # Entity→Entity attention
            ee = mean_attn[np.ix_(entity_idx, entity_idx)].mean()
            stats[f"layer_{layer_idx}_entity_to_entity"].append(float(ee))

            # Entity→Context attention
            ec = mean_attn[np.ix_(entity_idx, non_entity_idx)].mean()
            stats[f"layer_{layer_idx}_entity_to_context"].append(float(ec))

            # Entropy of attention distribution (per head, per entity token)
            for entity_i in entity_idx:
                for head in range(attn.shape[0]):
                    dist = attn[head, entity_i, :]
                    dist = dist / (dist.sum() + 1e-8)
                    entropy = -np.sum(dist * np.log(dist + 1e-9))
                    stats[f"layer_{layer_idx}_entropy"].append(float(entropy))

    # Aggregate
    aggregated = {}
    for key, vals in stats.items():
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }

    return aggregated


def plot_layer_attention_summary(stats_dict, save_path):
    """Plot entity-to-entity vs entity-to-context attention per layer."""
    layers = list(range(12))
    ee_means = [stats_dict.get(f"layer_{l}_entity_to_entity", {}).get("mean", 0) for l in layers]
    ec_means = [stats_dict.get(f"layer_{l}_entity_to_context", {}).get("mean", 0) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, ee_means, "o-", label="Entity→Entity", color="red", linewidth=2)
    ax.plot(layers, ec_means, "s--", label="Entity→Context", color="blue", linewidth=2)
    ax.set_xlabel("BERT Layer", fontsize=12)
    ax.set_ylabel("Mean Attention Weight", fontsize=12)
    ax.set_title("Entity Attention Patterns Across Layers", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    sentences = read_conll_file(args.eval_conll)
    # Select sentences with entities
    entity_sentences = [s for s in sentences if any(l != "O" for _, l in s)]
    print(f"Entity sentences: {len(entity_sentences)} / {len(sentences)} total")

    # Limit to num_examples
    examples = entity_sentences[: args.num_examples]

    records = []
    for sent_idx, sentence in enumerate(examples):
        tokens_words = [t for t, _ in sentence]
        labels_words = [l for _, l in sentence]

        subword_tokens, word_ids, attentions, lw, tw = extract_attention_weights(
            model, tokenizer, sentence, args.max_length, device
        )

        # Map entity words to subword indices (first subtoken only)
        entity_subword_indices = []
        seen = set()
        for si, wi in enumerate(word_ids):
            if wi is None or wi in seen:
                continue
            seen.add(wi)
            if wi < len(labels_words) and labels_words[wi] != "O":
                entity_subword_indices.append(si)

        entity_mask = [False] * len(subword_tokens)
        for ei in entity_subword_indices:
            entity_mask[ei] = True

        records.append({
            "sent_idx": sent_idx,
            "subword_tokens": subword_tokens,
            "word_ids": word_ids,
            "attentions": attentions,
            "entity_subword_indices": entity_subword_indices,
            "entity_mask": entity_mask,
            "labels": labels_words,
        })

        # Plot attention for each layer and head for this example
        if sent_idx < 5:  # Only plot first 5 sentences in detail
            sent_dir = os.path.join(args.output_dir, f"sentence_{sent_idx:03d}")
            os.makedirs(sent_dir, exist_ok=True)

            for layer_idx, attn_heads in enumerate(attentions):
                if args.layer is not None and layer_idx != args.layer:
                    continue
                # Mean over heads
                mean_attn = attn_heads.mean(axis=0)
                save_path = os.path.join(sent_dir, f"layer_{layer_idx:02d}_mean.png")
                entity_labels = [f"{labels_words[wi] if wi is not None and wi < len(labels_words) else 'SPE'}"
                                 for wi in word_ids]
                title = f"Sentence {sent_idx} | Layer {layer_idx} (mean over 12 heads)"
                plot_attention_heatmap(
                    mean_attn[:24, :24] if len(subword_tokens) > 24 else mean_attn,
                    subword_tokens[:24] if len(subword_tokens) > 24 else subword_tokens,
                    title, save_path, entity_mask[:24] if len(subword_tokens) > 24 else entity_mask
                )

            # Plot mean over all layers, all heads
            all_layers_mean = np.mean([a.mean(axis=0) for a in attentions], axis=0)
            plot_attention_heatmap(
                all_layers_mean[:24, :24] if len(subword_tokens) > 24 else all_layers_mean,
                subword_tokens[:24] if len(subword_tokens) > 24 else subword_tokens,
                f"Sentence {sent_idx} | Mean all layers and heads",
                os.path.join(sent_dir, "all_layers_mean.png"),
                entity_mask[:24] if len(subword_tokens) > 24 else entity_mask
            )

        print(f"  Processed sentence {sent_idx}: {len(subword_tokens)} subtokens, "
              f"{len(entity_subword_indices)} entity tokens")

    # Compute statistics
    print("Computing attention statistics...")
    stats = compute_attention_statistics(records)

    # Save attention statistics
    import csv
    rows = []
    for key, val in stats.items():
        parts = key.rsplit("_", 2)
        rows.append({
            "stat_key": key,
            "mean": round(val["mean"], 6),
            "std": round(val["std"], 6),
            "n": val["n"],
        })
    with open(os.path.join(args.output_dir, "attention_statistics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stat_key", "mean", "std", "n"])
        w.writeheader()
        w.writerows(rows)
    print("Saved attention_statistics.csv")

    # Layer summary plot
    plot_layer_attention_summary(
        stats,
        os.path.join(args.output_dir, "attention_layer_summary.png")
    )
    print("Saved attention_layer_summary.png")

    # Save metadata
    meta = {
        "model_dir": args.model_dir,
        "eval_conll": args.eval_conll,
        "num_sentences_processed": len(records),
        "num_entity_sentences": len(entity_sentences),
        "bert_layers": 12,
        "bert_heads": 12,
    }
    with open(os.path.join(args.output_dir, "attention_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")
    print(f"Output directory: {args.output_dir}")
    for fname in sorted(os.listdir(args.output_dir)):
        if os.path.isfile(os.path.join(args.output_dir, fname)):
            size = os.path.getsize(os.path.join(args.output_dir, fname))
            print(f"  {fname:50s} {size:>8,} bytes")


if __name__ == "__main__":
    main()
