"""OOV entity experiment: find entities unseen during training and retrieve top-3 nearest labels.

Saves CSV results under `--output-dir/oov_results.csv` and a JSON summary.
"""
import argparse
import csv
import json
import os
from typing import List

import numpy as np

from ner_stats.data_utils import read_conll_bio
from ner_stats.embeddings import TransformerEmbedder, get_word_aligned_embeddings
from ner_stats.cva import top_k_cosine_similarities
from scripts.train_ner import build_cva_class_vectors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="data/full_train.conll")
    p.add_argument("--eval-file", default="data/full_eval.conll")
    p.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir", default="results/oov_experiment")
    p.add_argument("--device", default=None)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    train_tokens, train_labels = read_conll_bio(args.train_file)
    eval_tokens, eval_labels = read_conll_bio(args.eval_file)

    # build set of seen entity surfaces in train (lowercased)
    seen = set()
    for sent, labs in zip(train_tokens, train_labels):
        for tok, lab in zip(sent, labs):
            if lab != "O":
                seen.add(tok.lower())

    embedder = TransformerEmbedder(model_name=args.model_name, device=args.device)
    class_vectors = build_cva_class_vectors(train_tokens, train_labels, embedder, max_length=256)

    results_path = os.path.join(args.output_dir, "oov_results.csv")
    summary = {"total_oov": 0}

    with open(results_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sent_idx", "token_idx", "token", "gold_label", "top1", "top1_score", "top3"])
        total = 0
        for si, (sent, labs) in enumerate(zip(eval_tokens, eval_labels)):
            # get word-aligned embeddings for the sentence
            words, embeddings = embedder.encode_words(sent)
            # map words back: words list corresponds to first-subtoken aligned words
            for ti, (w, emb, gold) in enumerate(zip(words, embeddings, labs)):
                if gold == "O":
                    continue
                if w.lower() in seen:
                    continue
                total += 1
                top3 = top_k_cosine_similarities(emb.detach().cpu().numpy(), class_vectors, k=3)
                top1, top1_score = top3[0] if top3 else (None, None)
                writer.writerow([si, ti, w, gold, top1, top1_score, json.dumps(top3, ensure_ascii=False)])

    summary["total_oov"] = total
    with open(os.path.join(args.output_dir, "oov_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"OOV experiment completed. Results: {results_path}")


if __name__ == "__main__":
    main()
