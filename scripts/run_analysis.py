"""
Entry point script to compute corpus statistics and generate plots.

Usage:
    python scripts/run_analysis.py --data-root data/annotation
"""

import argparse
import os
import sys

# When this script is executed directly by filename (e.g. "python scripts/run_analysis.py"),
# Python sets sys.path[0] to the script directory (scripts/) rather than the project root.
# Add the project root to sys.path so `from ner_stats import ...` works without requiring
# PYTHONPATH or `python -m scripts.run_analysis`.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.statistics import (
    compute_corpus_statistics,
    save_stats_json,
    save_counter_csv,
    plot_top_entity_types,
    plot_entity_sentence_ratio,
    plot_sentence_length_histogram,
    collect_sentence_lengths,
    ensure_dir,
)

from ner_stats.tagset import load_tagset, group_tagset_by_prefix, compare_tagset_with_corpus



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute statistics for Turkish Extended NER annotations."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/annotation",
        help="Root directory containing document folders (default: data/annotation)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where statistics and plots will be saved (default: results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotation_root = args.data_root
    results_root = args.results_dir
    plots_dir = os.path.join(results_root, "plots")

    ensure_dir(results_root)
    ensure_dir(plots_dir)

    print(f"[INFO] Using annotation root: {annotation_root}")
    print(f"[INFO] Results will be saved under: {results_root}")

    # 1) Compute corpus statistics
    stats, label_counter, type_counter = compute_corpus_statistics(annotation_root)

    # 2) Save numerical outputs
    stats_path = os.path.join(results_root, "stats.json")
    label_csv_path = os.path.join(results_root, "label_counts.csv")
    type_csv_path = os.path.join(results_root, "type_counts.csv")

    save_stats_json(stats, stats_path)
    save_counter_csv(label_counter, label_csv_path, column_name="label")
    save_counter_csv(type_counter, type_csv_path, column_name="entity_type")

    print("[INFO] Saved stats.json, label_counts.csv, and type_counts.csv")

    # 3) Collect sentence lengths for histogram
    sentence_lengths = collect_sentence_lengths(annotation_root)

    # 4) Generate plots
    top_types_plot = os.path.join(plots_dir, "top20_entity_types.png")
    ratio_plot = os.path.join(plots_dir, "entity_sentence_ratio.png")
    sent_len_plot = os.path.join(plots_dir, "sentence_length_hist.png")

    plot_top_entity_types(type_counter, top_types_plot, top_n=20)
    plot_entity_sentence_ratio(stats, ratio_plot)
    plot_sentence_length_histogram(sentence_lengths, sent_len_plot)

    print("[INFO] Plots generated in", plots_dir)

    # 5) Tagset-based quality check (if tagset file is available)
    tagset_path = "data/ENER-tagset.tsv"
    if os.path.exists(tagset_path):
        print(f"\n[INFO] Loading tagset from {tagset_path}")
        tagset = load_tagset(tagset_path)

        # entity types seen in the corpus (same as keys of type_counter)
        corpus_entity_types = set(type_counter.keys())

        qc = compare_tagset_with_corpus(tagset, corpus_entity_types)

        unused_path = os.path.join(results_root, "unused_tags_in_corpus.txt")
        unknown_path = os.path.join(results_root, "unknown_types_in_tagset_comparison.txt")

        with open(unused_path, "w", encoding="utf-8") as f:
            for t in qc["unused_in_corpus"]:
                f.write(t + "\n")

        with open(unknown_path, "w", encoding="utf-8") as f:
            for t in qc["unknown_in_tagset"]:
                f.write(t + "\n")

        print("[INFO] Tagset QC:")
        print(f"  - Tagset size           : {len(tagset)}")
        print(f"  - Types seen in corpus  : {len(corpus_entity_types)}")
        print(f"  - Unused in corpus      : {len(qc['unused_in_corpus'])}")
        print(f"  - Unknown vs. tagset    : {len(qc['unknown_in_tagset'])}")
        print(f"[INFO] Details saved to:\n  {unused_path}\n  {unknown_path}")
    else:
        print("\n[WARN] Tagset file not found, skipping tagset QC.")


    # 6) Print a short summary to console
    print("\n=== Corpus Summary ===")
    print(f"Total document folders      : {stats.total_document_folders}")
    print(f"Annotated documents         : {stats.annotated_documents}")
    print(f"Unannotated documents       : {stats.unannotated_documents}")
    print(f"Total sentences             : {stats.total_sentences}")
    print(f"Sentences with entity       : {stats.sentences_with_entity}")
    print(f"Entity sentence ratio       : {stats.entity_sentence_ratio:.3f}")
    print(f"Total tokens                : {stats.total_tokens}")
    print(f"# BIO labels                : {stats.num_entity_labels_bio}")
    print(f"# entity types              : {stats.num_entity_types}")
    print(f"Average sentence length     : {stats.average_sentence_length:.2f} tokens")


if __name__ == "__main__":
    main()
