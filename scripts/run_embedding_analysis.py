"""Command-line entrypoint to run the embedding analysis pipeline.

Usage: python scripts/run_embedding_analysis.py --help
"""
import argparse
import os
import time
import json

from ner_stats.data_utils import set_global_seed, read_conll_bio
from ner_stats.embeddings import TransformerEmbedder
from ner_stats.embedding_analysis import (
    extract_token_embeddings_from_dataset,
    aggregate_embeddings_by_label,
    export_embeddings,
    build_class_vectors_from_records,
)
from ner_stats import visualization
from ner_stats import evaluation_report
from ner_stats.experiment_logger import ExperimentLogger
from ner_stats.cva import classify_embedding_by_cosine_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-format", default="png")
    parser.add_argument("--use-cva", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--train", default="data/full_train.conll")
    parser.add_argument("--eval", default="data/full_eval.conll")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = ExperimentLogger(args.output_dir)
    set_global_seed(args.seed)

    # load data
    train_tokens, train_labels = read_conll_bio(args.train)
    eval_tokens, eval_labels = read_conll_bio(args.eval)

    logger.log("data_loaded", {"train_sents": len(train_tokens), "eval_sents": len(eval_tokens)})

    embedder = TransformerEmbedder(model_name=args.model_name, device=args.device)
    logger.log("model", {"name": args.model_name})

    # extract embeddings from train
    train_records, train_meta = extract_token_embeddings_from_dataset(train_tokens, train_labels, embedder, max_samples=args.max_samples)
    logger.log("train_extraction", train_meta)
    export_embeddings(train_records, os.path.join(args.output_dir, "train_embeddings"))

    # build class vectors
    class_vectors = build_class_vectors_from_records(train_records, use_cva=args.use_cva)
    # save class vectors
    cv_path = os.path.join(args.output_dir, "class_vectors.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in class_vectors.items()}, f, ensure_ascii=False)
    logger.log("class_vectors_saved", {"n_classes": len(class_vectors)})

    # extract eval embeddings and classify token-by-token using cosine similarity to class vectors
    eval_records, eval_meta = extract_token_embeddings_from_dataset(eval_tokens, eval_labels, embedder, max_samples=args.max_samples)
    logger.log("eval_extraction", eval_meta)
    export_embeddings(eval_records, os.path.join(args.output_dir, "eval_embeddings"))

    # prepare true and predicted sequences for token-level evaluation
    true_seqs = []
    pred_seqs = []
    # group records by sentence
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in eval_records:
        grouped[r["sent_idx"]].append(r)

    for sent_idx in sorted(grouped.keys()):
        recs = grouped[sent_idx]
        true = [r["label"] for r in recs]
        preds = []
        for r in recs:
            emb = r["embedding"]
            pred = classify_embedding_by_cosine_similarity(emb, class_vectors)
            # map back to BIO format: we output only type or O; use B- for predictions at token-level
            if pred == "O":
                preds.append("O")
            else:
                preds.append(f"B-{pred}")

        true_seqs.append(true)
        pred_seqs.append(preds)

    metrics = evaluation_report.generate_reports(true_seqs, pred_seqs, args.output_dir)
    logger.log("metrics", {"accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1")})

    # build embedding matrix and labels for visualization (use eval set)
    import numpy as np

    embs = [np.asarray(r["embedding"], dtype=np.float64) for r in eval_records]
    labels = [r["label_type"] for r in eval_records]
    if embs:
        X = np.vstack(embs)
        # PCA 2D
        X2, pca = visualization.pca_reduce(X, n_components=2)
        visualization.plot_2d_scatter(X2, labels, os.path.join(args.output_dir, f"pca_embeddings.{args.plot_format}"), title="PCA 2D (eval)")
        # overlay prototypes (class vectors) if available
        try:
            visualization.plot_prototypes_2d(X2, labels, class_vectors, os.path.join(args.output_dir, f"pca_embeddings_with_prototypes.{args.plot_format}"), title="PCA 2D with Prototypes (eval)", pca=pca)
        except Exception:
            pass
        # PCA 3D
        X3, pca3 = visualization.pca_reduce(X, n_components=3)
        visualization.plot_3d_scatter(X3, labels, os.path.join(args.output_dir, f"pca_embeddings_3d.{args.plot_format}"), title="PCA 3D (eval)")
        # t-SNE
        X_tsne = visualization.tsne_reduce(X, n_components=2)
        visualization.plot_2d_scatter(X_tsne, labels, os.path.join(args.output_dir, f"tsne_embeddings.{args.plot_format}"), title="t-SNE 2D (eval)")
        # UMAP if available
        try:
            X_umap = visualization.umap_reduce(X, n_components=2)
            visualization.plot_2d_scatter(X_umap, labels, os.path.join(args.output_dir, f"umap_embeddings.{args.plot_format}"), title="UMAP 2D (eval)")
        except Exception:
            pass

    logger.save()
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
