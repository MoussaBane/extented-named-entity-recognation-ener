"""
Extract Q, K, V projection vectors from BERT attention layers for entity tokens.

For each token in the evaluation set, this script extracts the Query, Key,
and Value projections from a specified BERT attention layer. It then saves
entity-token Q/K/V vectors and generates PCA/t-SNE/UMAP visualizations.

Usage:
    python scripts/extract_qkv_vectors.py \
        --model-dir outputs/bert-ner-full/checkpoint-390 \
        --eval-conll data/full_eval.conll \
        --output-dir results/qkv_analysis \
        --layer 11
"""

import argparse
import os
import json
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from ner_stats.conll_reader import read_conll_file
from ner_stats.data_utils import build_label_maps, set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Extract Q/K/V vectors from BERT attention layers")
    p.add_argument("--model-dir", required=True, help="Path to fine-tuned BERT model directory")
    p.add_argument("--eval-conll", required=True, help="Path to evaluation CoNLL file")
    p.add_argument("--output-dir", required=True, help="Directory to save outputs")
    p.add_argument("--layer", type=int, default=11, help="BERT layer index (0-11) to extract from")
    p.add_argument("--max-sentences", type=int, default=None, help="Limit number of sentences")
    p.add_argument("--max-length", type=int, default=256, help="Max tokenizer length")
    p.add_argument("--top-n-labels", type=int, default=15, help="Top N labels to visualize")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_qkv_projections(model, layer_idx: int):
    """Extract the Q, K, V weight matrices from a BERT attention layer."""
    layer = model.bert.encoder.layer[layer_idx]
    attn = layer.attention.self
    return attn.query, attn.key, attn.value


def extract_entity_qkv(
    model,
    tokenizer,
    sentences,
    label2id,
    layer_idx: int,
    max_length: int,
    device: torch.device,
):
    """
    For each entity token in each sentence, extract the Q/K/V projections
    from the specified BERT attention layer.

    Returns:
        records: list of dicts with keys:
            token, label, hidden_state, q_vec, k_vec, v_vec
    """
    query_proj, key_proj, value_proj = get_qkv_projections(model, layer_idx)
    records = []

    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            tokens_words = [t for t, _ in sentence]
            labels_words = [l for _, l in sentence]

            encoding = tokenizer(
                tokens_words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            word_ids = encoding.word_ids()
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Hidden states: shape (num_layers+1, 1, seq_len, 768)
            hidden_states = outputs.hidden_states  # tuple of tensors

            # We use the output of layer `layer_idx` as input to layer `layer_idx+1`
            # The hidden state entering layer `layer_idx+1` is hidden_states[layer_idx+1]
            # To get Q/K/V at layer `layer_idx`, use hidden_states[layer_idx] as input
            h = hidden_states[layer_idx][0]  # (seq_len, 768)

            # Project to Q, K, V using the layer's linear projections
            q = query_proj(h)  # (seq_len, head_dim * num_heads)
            k = key_proj(h)
            v = value_proj(h)

            # Also grab final hidden state (last layer)
            final_h = hidden_states[-1][0]  # (seq_len, 768)

            # Align to word-level (first subtoken)
            seen_words = set()
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen_words:
                    continue
                if word_idx >= len(labels_words):
                    continue
                seen_words.add(word_idx)

                label = labels_words[word_idx]
                word = tokens_words[word_idx]

                records.append({
                    "token": word,
                    "label": label,
                    "hidden_state": final_h[tok_idx].cpu().numpy(),
                    "q_vec": q[tok_idx].cpu().numpy(),
                    "k_vec": k[tok_idx].cpu().numpy(),
                    "v_vec": v[tok_idx].cpu().numpy(),
                })

    return records


def compute_class_prototypes(records, vec_key: str):
    """Compute per-class mean vectors from records."""
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r[vec_key])
    prototypes = {}
    for label, vecs in by_label.items():
        prototypes[label] = np.mean(vecs, axis=0)
    return prototypes


def plot_pca(embeddings, labels, title, save_path, top_n=15):
    """PCA 2D scatter plot colored by entity label."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Select top_n most frequent labels (excluding O)
    label_counts = dict(zip(unique_labels, counts))
    top_labels = sorted(
        [l for l in label_counts if l != "O"],
        key=lambda x: -label_counts[x]
    )[:top_n]
    top_labels.append("O")
    top_set = set(top_labels)

    mask = np.array([l in top_set for l in labels])
    filtered_embs = embeddings[mask]
    filtered_labels = np.array(labels)[mask]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(filtered_embs)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab20", len(top_labels))
    for i, label in enumerate(top_labels):
        idx = filtered_labels == label
        if idx.sum() == 0:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.6,
                   color=cmap(i), label=label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return pca.explained_variance_ratio_[:2]


def plot_tsne(embeddings, labels, title, save_path, top_n=15):
    """t-SNE 2D scatter plot."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    top_labels = sorted(
        [l for l in label_counts if l != "O"],
        key=lambda x: -label_counts[x]
    )[:top_n]
    top_labels.append("O")
    top_set = set(top_labels)

    mask = np.array([l in top_set for l in labels])
    filtered_embs = embeddings[mask]
    filtered_labels = np.array(labels)[mask]

    # Subsample for speed if large
    if len(filtered_embs) > 3000:
        idx = np.random.choice(len(filtered_embs), 3000, replace=False)
        filtered_embs = filtered_embs[idx]
        filtered_labels = filtered_labels[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(filtered_embs)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab20", len(top_labels))
    for i, label in enumerate(top_labels):
        idx = filtered_labels == label
        if idx.sum() == 0:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.6,
                   color=cmap(i), label=label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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

    print(f"Loading eval data: {args.eval_conll}")
    sentences = read_conll_file(args.eval_conll)
    if args.max_sentences:
        sentences = sentences[: args.max_sentences]
    print(f"  {len(sentences)} sentences loaded")

    label2id = model.config.label2id
    id2label = model.config.id2label

    print(f"Extracting Q/K/V from layer {args.layer}...")
    records = extract_entity_qkv(
        model, tokenizer, sentences, label2id,
        layer_idx=args.layer,
        max_length=args.max_length,
        device=device,
    )
    print(f"  Extracted {len(records)} token records")

    # Separate entity tokens from O tokens
    entity_records = [r for r in records if r["label"] != "O"]
    print(f"  Entity tokens: {len(entity_records)}")

    labels_all = [r["label"] for r in records]
    labels_entity = [r["label"] for r in entity_records]

    # Stack arrays
    hidden_all = np.stack([r["hidden_state"] for r in records])
    q_all = np.stack([r["q_vec"] for r in records])
    k_all = np.stack([r["k_vec"] for r in records])
    v_all = np.stack([r["v_vec"] for r in records])

    hidden_entity = np.stack([r["hidden_state"] for r in entity_records]) if entity_records else np.zeros((0, 768))
    q_entity = np.stack([r["q_vec"] for r in entity_records]) if entity_records else np.zeros((0, q_all.shape[1]))
    k_entity = np.stack([r["k_vec"] for r in entity_records]) if entity_records else np.zeros((0, k_all.shape[1]))
    v_entity = np.stack([r["v_vec"] for r in entity_records]) if entity_records else np.zeros((0, v_all.shape[1]))

    # Save entity NPY files
    np.save(os.path.join(args.output_dir, "entity_embeddings.npy"), hidden_entity)
    np.save(os.path.join(args.output_dir, "entity_q_vectors.npy"), q_entity)
    np.save(os.path.join(args.output_dir, "entity_k_vectors.npy"), k_entity)
    np.save(os.path.join(args.output_dir, "entity_v_vectors.npy"), v_entity)

    # Save all-token arrays too
    np.save(os.path.join(args.output_dir, "all_q_vectors.npy"), q_all)
    np.save(os.path.join(args.output_dir, "all_k_vectors.npy"), k_all)
    np.save(os.path.join(args.output_dir, "all_v_vectors.npy"), v_all)

    print(f"Saved entity_embeddings.npy {hidden_entity.shape}")
    print(f"Saved entity_q_vectors.npy {q_entity.shape}")
    print(f"Saved entity_k_vectors.npy {k_entity.shape}")
    print(f"Saved entity_v_vectors.npy {v_entity.shape}")

    # Compute class prototypes
    proto_h = compute_class_prototypes(entity_records, "hidden_state")
    proto_q = compute_class_prototypes(entity_records, "q_vec")
    proto_k = compute_class_prototypes(entity_records, "k_vec")
    proto_v = compute_class_prototypes(entity_records, "v_vec")

    # Save prototype vectors as npy
    proto_labels = sorted(proto_h.keys())
    proto_matrix = np.stack([proto_h[l] for l in proto_labels])
    np.save(os.path.join(args.output_dir, "prototype_vectors.npy"), proto_matrix)
    with open(os.path.join(args.output_dir, "prototype_labels.json"), "w") as f:
        json.dump(proto_labels, f)
    print(f"Saved prototype_vectors.npy {proto_matrix.shape} ({len(proto_labels)} classes)")

    # Cosine similarity matrix between prototypes
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    sim_matrix = cosine_similarity(proto_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=proto_labels, columns=proto_labels)
    sim_df.to_csv(os.path.join(args.output_dir, "centroid_similarity_matrix.csv"))
    print("Saved centroid_similarity_matrix.csv")

    # Prototype similarity heatmap (top-20 by frequency)
    from collections import Counter
    label_counts = Counter(labels_entity)
    top20 = [l for l, _ in label_counts.most_common(20) if l in proto_h]
    if len(top20) >= 2:
        sub_sim = cosine_similarity(
            np.stack([proto_h[l] for l in top20])
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sub_sim, cmap="RdYlGn", vmin=0.7, vmax=1.0)
        ax.set_xticks(range(len(top20)))
        ax.set_yticks(range(len(top20)))
        ax.set_xticklabels(top20, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(top20, fontsize=9)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("Prototype Centroid Similarity Matrix (Top-20 Entity Types)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "prototype_similarity_heatmap.png"), dpi=150)
        plt.close()
        print("Saved prototype_similarity_heatmap.png")

    # Generate PCA/t-SNE plots for all 4 representation types
    plots = [
        ("hidden_state", hidden_entity, "Entity Hidden States (BERT Layer 12)", "pca_entities.png", "tsne_entities.png"),
        ("q_vec", q_entity, "Entity Q-Projections (BERT Layer {})".format(args.layer), "pca_q_vectors.png", "tsne_q_vectors.png"),
        ("k_vec", k_entity, "Entity K-Projections (BERT Layer {})".format(args.layer), "pca_k_vectors.png", "tsne_k_vectors.png"),
        ("v_vec", v_entity, "Entity V-Projections (BERT Layer {})".format(args.layer), "pca_v_vectors.png", "tsne_v_vectors.png"),
    ]

    stats = {}
    for key, embs, title, pca_name, tsne_name in plots:
        if len(embs) < 5:
            print(f"Skipping {key}: too few embeddings")
            continue
        print(f"Plotting PCA for {key}...")
        ev = plot_pca(
            embs, labels_entity, title,
            os.path.join(args.output_dir, pca_name),
            top_n=args.top_n_labels
        )
        stats[f"{key}_pca_explained"] = ev.tolist()

        print(f"Plotting t-SNE for {key}...")
        plot_tsne(
            embs, labels_entity, title.replace("PCA", "t-SNE"),
            os.path.join(args.output_dir, tsne_name),
            top_n=args.top_n_labels
        )

    # UMAP
    if HAS_UMAP and len(hidden_entity) > 10:
        print("Generating UMAP plots...")
        for key, embs, title, _, _ in plots:
            if len(embs) < 5:
                continue
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embs[:min(len(embs), 3000)])
            sub_labels = labels_entity[:min(len(labels_entity), 3000)]

            unique_labels, counts = np.unique(sub_labels, return_counts=True)
            lc = dict(zip(unique_labels, counts))
            top_labels = sorted([l for l in lc if l != "O"], key=lambda x: -lc[x])[:args.top_n_labels]
            top_labels.append("O")
            top_set = set(top_labels)
            mask = np.array([l in top_set for l in sub_labels])

            fig, ax = plt.subplots(figsize=(12, 8))
            cmap = plt.cm.get_cmap("tab20", len(top_labels))
            for i, label in enumerate(top_labels):
                idx = np.array(sub_labels)[mask] == label
                if idx.sum() == 0:
                    continue
                ax.scatter(coords[mask][idx, 0], coords[mask][idx, 1], s=8, alpha=0.6,
                           color=cmap(i), label=label)
            ax.set_title(f"UMAP: {title}", fontsize=14)
            ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=2)
            plt.tight_layout()
            umap_path = os.path.join(args.output_dir, f"umap_{key.replace('_', '')}.png")
            plt.savefig(umap_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {umap_path}")
    else:
        print("UMAP not available (pip install umap-learn) or too few embeddings.")

    # Save metadata
    meta = {
        "model_dir": args.model_dir,
        "eval_conll": args.eval_conll,
        "bert_layer": args.layer,
        "total_tokens": len(records),
        "entity_tokens": len(entity_records),
        "num_entity_classes": len(proto_labels),
        "hidden_dim": int(hidden_all.shape[1]),
        "qkv_dim": int(q_all.shape[1]),
        "pca_stats": stats,
    }
    with open(os.path.join(args.output_dir, "qkv_extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone! Output files:")
    for fname in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, fname))
        print(f"  {fname:45s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
