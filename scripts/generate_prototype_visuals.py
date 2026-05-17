"""Generate prototype overlay plots and nearest-neighbor listings for CVA prototypes."""
import os
import json
import numpy as np
from ner_stats import visualization

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="results/embedding_smoke")
args = parser.parse_args()

OUT = args.out_dir
EMB_NPY = os.path.join(OUT, "eval_embeddings.npy")
EMB_CSV = os.path.join(OUT, "eval_embeddings.csv")
CV_PATH = os.path.join(OUT, "class_vectors.json")


def load_records_csv(path):
    import csv

    records = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for parts in reader:
            if len(parts) < 6:
                continue
            sent_idx = int(parts[0])
            token_idx = int(parts[1])
            token = parts[2]
            label = parts[3]
            label_type = parts[4]
            # join any remaining parts in case the CSV was split unexpectedly
            emb_raw = parts[5] if len(parts) == 6 else ",".join(parts[5:])
            # strip surrounding quotes if present
            emb_raw = emb_raw.strip()
            if emb_raw.startswith('"') and emb_raw.endswith('"'):
                emb_raw = emb_raw[1:-1]
            # try parsing; if it fails, attempt to replace commas with spaces then parse
            try:
                emb = np.fromstring(emb_raw, sep=" ")
                if emb.size == 0:
                    raise ValueError("empty parse")
            except Exception:
                try:
                    emb = np.fromstring(emb_raw.replace(",", " "), sep=" ")
                except Exception:
                    emb = np.array([])
            records.append({"sent_idx": sent_idx, "token_idx": token_idx, "token": token, "label": label, "label_type": label_type, "embedding": emb})
    return records


def main():
    os.makedirs(OUT, exist_ok=True)
    embs = np.load(EMB_NPY)
    records = load_records_csv(EMB_CSV)
    # build stacked embeddings from records (skip malformed/empty embeddings)
    filtered_records = [r for r in records if isinstance(r.get("embedding"), np.ndarray) and r.get("embedding").size > 0]
    if len(filtered_records) == 0:
        # fallback to .npy if CSV parsing failed entirely
        stacked = embs
        labels = [r["label_type"] for r in records]
        record_index_map = list(range(len(records)))
    else:
        stacked = np.vstack([r["embedding"] for r in filtered_records])
        labels = [r["label_type"] for r in filtered_records]
        record_index_map = None

    with open(CV_PATH, encoding="utf-8") as f:
        prototypes = json.load(f)

    # PCA projection of embeddings (use stacked from records)
    X2, pca = visualization.pca_reduce(stacked, n_components=2)
    visualization.plot_prototypes_2d(X2, labels, prototypes, os.path.join(OUT, "prototypes_pca.png"), title="Prototypes (PCA)")

    # nearest neighbors per prototype (cosine)
    from numpy.linalg import norm

    # use the same stacked embeddings for neighbor search
    norms = norm(stacked, axis=1)
    neighbors = {}
    for k, v in prototypes.items():
        vec = np.asarray(v, dtype=float)
        sim = (stacked @ vec) / (norms * np.linalg.norm(vec) + 1e-12)
        idx = np.argsort(-sim)[:10]
        # map indices to tokens from filtered_records if used
        if record_index_map is None:
            source_records = filtered_records
        else:
            source_records = records
        neighbors[k] = [{"token": source_records[i]["token"], "label": source_records[i]["label_type"], "similarity": float(sim[i])} for i in idx]

    with open(os.path.join(OUT, "prototype_neighbors.json"), "w", encoding="utf-8") as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)

    print("Prototype plots and neighbor lists saved to", OUT)


if __name__ == "__main__":
    main()
