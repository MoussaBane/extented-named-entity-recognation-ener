"""Compare Context-only (mean vectors) vs CVA (common vectors) vs Combined.

Builds class mean vectors and CVA common vectors from training split, then
classifies evaluation tokens and produces evaluation reports for each method.
"""
import argparse
import json
import os
from typing import Dict

import numpy as np

from ner_stats.data_utils import read_conll_bio
from ner_stats.embeddings import TransformerEmbedder
from ner_stats.cva import compute_class_mean_vectors, compute_cva_common_vectors, top_k_cosine_similarities
from ner_stats import evaluation_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="data/full_train.conll")
    p.add_argument("--eval-file", default="data/full_eval.conll")
    p.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir", default="results/context_vs_cva")
    p.add_argument("--device", default=None)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    train_tokens, train_labels = read_conll_bio(args.train_file)
    eval_tokens, eval_labels = read_conll_bio(args.eval_file)

    embedder = TransformerEmbedder(model_name=args.model_name, device=args.device)

    # accumulate embeddings by label from train
    from collections import defaultdict

    buckets: Dict[str, list] = defaultdict(list)
    for words, labs in zip(train_tokens, train_labels):
        aligned_words, aligned_embs = embedder.encode_words(words)
        for lbl, emb in zip(labs, aligned_embs):
            buckets.setdefault(lbl, []).append(emb.detach().cpu().numpy())

    # compute mean vectors and CVA common vectors
    means = {k: np.vstack(v).mean(axis=0) for k, v in buckets.items() if v}
    cva = compute_cva_common_vectors({k: np.vstack(v) for k, v in buckets.items() if v})

    # helper: classify by highest cosine to prototypes
    def classify_by_prototypes(emb, prototypes):
        sims = [(k, float(np.dot(emb, v) / (np.linalg.norm(emb) * np.linalg.norm(v)))) for k, v in prototypes.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[0][0] if sims else "O"

    # run evaluation: produce predicted sequences for each method
    preds_context = []
    preds_cva = []
    preds_combined = []

    for words, labs in zip(eval_tokens, eval_labels):
        aligned_words, aligned_embs = embedder.encode_words(words)
        sent_ctx = []
        sent_cva = []
        sent_comb = []
        for emb in aligned_embs:
            emb_np = emb.detach().cpu().numpy()
            # context-only (means)
            ctx_label = classify_by_prototypes(emb_np, means)
            cva_label = classify_by_prototypes(emb_np, cva)
            # combined: average cosine scores
            # compute scores per class for both
            combined_scores = {}
            for k in set(list(means.keys()) + list(cva.keys())):
                s1 = float(np.dot(emb_np, means[k]) / (np.linalg.norm(emb_np) * np.linalg.norm(means[k]))) if k in means else -1.0
                s2 = float(np.dot(emb_np, cva[k]) / (np.linalg.norm(emb_np) * np.linalg.norm(cva[k]))) if k in cva else -1.0
                combined_scores[k] = 0.5 * (s1 + s2)
            comb_label = max(combined_scores.items(), key=lambda x: x[1])[0] if combined_scores else "O"

            sent_ctx.append(ctx_label)
            sent_cva.append(cva_label)
            sent_comb.append(comb_label)

        preds_context.append(sent_ctx)
        preds_cva.append(sent_cva)
        preds_combined.append(sent_comb)

    # produce reports
    ctx_dir = os.path.join(args.output_dir, "context_only")
    ensure_dir(ctx_dir)
    metrics_ctx = evaluation_report.generate_reports(eval_labels, preds_context, ctx_dir)

    cva_dir = os.path.join(args.output_dir, "cva_only")
    ensure_dir(cva_dir)
    metrics_cva = evaluation_report.generate_reports(eval_labels, preds_cva, cva_dir)

    comb_dir = os.path.join(args.output_dir, "combined")
    ensure_dir(comb_dir)
    metrics_comb = evaluation_report.generate_reports(eval_labels, preds_combined, comb_dir)

    summary = {"context": metrics_ctx, "cva": metrics_cva, "combined": metrics_comb}
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Comparison completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
