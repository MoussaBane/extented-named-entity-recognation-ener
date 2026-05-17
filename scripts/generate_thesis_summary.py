"""Generate a thesis-ready markdown summary from produced results."""
import os
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embed-dir", default="results/embedding_smoke")
parser.add_argument("--model-comparison", default="results/model_comparison_smoke/comparison_summary.json")
parser.add_argument("--out-md", default="results/thesis_results_summary.md")
args = parser.parse_args()

OUT_MD = args.out_md
EMB_OUT = os.path.join(args.embed_dir, "eval_embeddings.npy")
EMB_CSV = os.path.join(args.embed_dir, "eval_embeddings.csv")
CLASS_VECT = os.path.join(args.embed_dir, "class_vectors.json")
COMP_SUM = args.model_comparison


def load_label_counts(csv_path):
    counts = {}
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 6:
                continue
            lbl = parts[4]
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def main():
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    emb = np.load(EMB_OUT)
    with open(CLASS_VECT, encoding="utf-8") as f:
        class_vectors = json.load(f)

    comp = None
    if os.path.exists(COMP_SUM):
        with open(COMP_SUM, encoding="utf-8") as f:
            comp = json.load(f)

    counts = load_label_counts(EMB_CSV)
    top_classes = sorted(counts.items(), key=lambda x: -x[1])[:10]

    from sklearn.decomposition import PCA
    n_comp = min(10, emb.shape[0], emb.shape[1])
    if n_comp < 1:
        n_comp = 1
    pca = PCA(n_components=n_comp)
    pca.fit(emb)
    expl = pca.explained_variance_ratio_[:5]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("## Thesis Results Summary\n\n")
        f.write("**Dataset statistics:**\n\n")
        f.write(f"- Eval embeddings: {emb.shape[0]} tokens, embedding dim={emb.shape[1]}\n")
        f.write("\n**Embedding statistics:**\n\n")
        f.write(f"- Mean vector norm: {np.mean(np.linalg.norm(emb, axis=1)):.4f}\n")
        f.write(f"- Embedding std (per-dim mean): {np.mean(np.std(emb, axis=0)):.4f}\n")
        f.write("\n**PCA explained variance (top 5 PCs):**\n\n")
        for i, r in enumerate(expl, start=1):
            f.write(f"- PC{i}: {r:.4f}\n")
        f.write("\n**Top classes by token count (eval):**\n\n")
        for cls, c in top_classes:
            f.write(f"- {cls}: {c}\n")

        if comp is not None:
            f.write("\n**Classification performance (smoke comparison):**\n\n")
            bert = comp.get("bert", {})
            cva = comp.get("cva", {})
            f.write("| Method | Accuracy | Macro F1 | Avg inference (s) |\n")
            f.write("| --- | ---: | ---: | ---: |\n")
            f.write(f"| Transformer | {bert.get('accuracy_with_o', 'N/A')} | {bert.get('macro_f1_without_o', bert.get('macro_f1_with_o','N/A'))} | {bert.get('timing',{}).get('average_seconds','N/A')} |\n")
            f.write(f"| CVA         | {cva.get('accuracy_with_o', 'N/A')} | {cva.get('macro_f1_without_o', cva.get('macro_f1_with_o','N/A'))} | {cva.get('timing',{}).get('average_seconds','N/A')} |\n")

        f.write("\n**CVA vs Transformer:**\n\n")
        f.write("See `results/model_comparison_smoke/` for full metric JSONs and confusion matrices.\n")

    print("Wrote thesis summary to", OUT_MD)


if __name__ == "__main__":
    main()
