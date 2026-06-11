"""Multi-seed evaluation with paired bootstrap and paired t-test significance testing.

Runs the BERT + CVA pipeline for multiple random seeds and reports:
  - per-seed macro F1 for both methods
  - mean ± std across seeds
  - paired t-test p-value (BERT vs CVA, across seeds)
  - paired bootstrap p-value (BERT vs CVA, on the last seed's predictions)

If a BERT checkpoint already exists under <output-dir>/seed_<N>/bert_model/,
training is skipped for that seed and only evaluation is rerun.

Usage:
    python scripts/run_multi_seed.py \\
        --train-file data/full_train.conll \\
        --eval-file  data/full_eval.conll  \\
        --seeds 42 123 456 789 1234        \\
        --output-dir results/multi_seed
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ner_stats.data_utils import (
    build_label_maps,
    normalize_bio_labels,
    read_conll_bio,
    set_global_seed,
    validate_bio_labels,
)
from ner_stats.embeddings import TransformerEmbedder
from ner_stats.evaluation import evaluate_token_classification
from scripts.train_ner import (
    NERDataset,
    build_cva_class_vectors,
    get_word_level_alignment,
    is_transformers_model_dir,
    predict_cva_labels,
    to_jsonable_metrics,
    trainer_predictions_to_labels,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate as evaluate_lib


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed BERT + CVA evaluation with bootstrap significance test."
    )
    p.add_argument("--train-file", default="data/full_train.conll")
    p.add_argument("--eval-file", default="data/full_eval.conll")
    p.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--output-dir", default="results/multi_seed")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1234])
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Number of bootstrap resampling iterations for the significance test.",
    )
    p.add_argument(
        "--skip-bert-train", action="store_true",
        help="Skip training and reuse existing checkpoints for all seeds.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _flatten(seqs_a: List[List[str]], seqs_b: List[List[str]]) -> Tuple[List[str], List[str]]:
    a_flat: List[str] = []
    b_flat: List[str] = []
    for sa, sb in zip(seqs_a, seqs_b):
        a_flat.extend(sa)
        b_flat.extend(sb)
    return a_flat, b_flat


def _token_macro_f1(gold: List[str], pred: List[str], entity_labels: List[str]) -> float:
    """Token-level macro F1 excluding the O label, computed from flat lists."""
    from collections import defaultdict
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for g, p in zip(gold, pred):
        if g == "O" and p == "O":
            continue
        if g != "O":
            if g == p:
                tp[g] += 1
            else:
                fn[g] += 1
                if p != "O":
                    fp[p] += 1
        else:
            if p != "O":
                fp[p] += 1

    f1_sum = 0.0
    for label in entity_labels:
        t = tp[label]
        prec = t / (t + fp[label]) if (t + fp[label]) > 0 else 0.0
        rec  = t / (t + fn[label]) if (t + fn[label]) > 0 else 0.0
        f1_sum += 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1_sum / len(entity_labels) if entity_labels else 0.0


def paired_bootstrap_test(
    gold_seqs: List[List[str]],
    pred_a_seqs: List[List[str]],
    pred_b_seqs: List[List[str]],
    entity_labels: List[str],
    n_bootstrap: int = 2000,
    rng_seed: int = 0,
) -> Dict:
    """Paired bootstrap test comparing system A vs system B.

    Resamples sentences (with replacement) *n_bootstrap* times and reports:
      - observed macro F1 for each system
      - observed difference (A − B)
      - bootstrap standard deviation of the difference
      - one-sided p-values

    Parameters
    ----------
    gold_seqs, pred_a_seqs, pred_b_seqs
        Sentence-level label sequences (equal length).
    entity_labels
        All entity labels (excluding O) for macro-F1 computation.
    n_bootstrap
        Number of bootstrap iterations.
    rng_seed
        NumPy RNG seed for reproducibility.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(gold_seqs)

    gold_flat, a_flat = _flatten(gold_seqs, pred_a_seqs)
    _, b_flat = _flatten(gold_seqs, pred_b_seqs)

    f1_a_obs = _token_macro_f1(gold_flat, a_flat, entity_labels)
    f1_b_obs = _token_macro_f1(gold_flat, b_flat, entity_labels)
    obs_diff = f1_a_obs - f1_b_obs

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bg = [gold_seqs[i] for i in idx]
        ba = [pred_a_seqs[i] for i in idx]
        bb = [pred_b_seqs[i] for i in idx]
        gf, af = _flatten(bg, ba)
        _, bf = _flatten(bg, bb)
        diffs.append(_token_macro_f1(gf, af, entity_labels) - _token_macro_f1(gf, bf, entity_labels))

    diffs_arr = np.array(diffs)
    return {
        "f1_a": float(f1_a_obs),
        "f1_b": float(f1_b_obs),
        "observed_diff_a_minus_b": float(obs_diff),
        "bootstrap_std": float(np.std(diffs_arr)),
        "p_value_a_better_than_b": float(np.mean(diffs_arr <= 0)),
        "p_value_b_better_than_a": float(np.mean(diffs_arr >= 0)),
        "n_bootstrap": n_bootstrap,
        "note": "p_value_a_better_than_b < 0.05 means A is significantly better than B at the 5% level.",
    }


def _paired_ttest(a_values: List[float], b_values: List[float]) -> Tuple[float, float]:
    """Paired t-test returning (t-statistic, two-sided p-value).

    Falls back to (NaN, NaN) when scipy is unavailable or n < 2.
    """
    if len(a_values) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import ttest_rel
        t, p = ttest_rel(a_values, b_values)
        return float(t), float(p)
    except ImportError:
        diffs = np.array(a_values) - np.array(b_values)
        n = len(diffs)
        mean_d = np.mean(diffs)
        std_d = np.std(diffs, ddof=1)
        se = std_d / np.sqrt(n)
        t_stat = float(mean_d / se) if se > 0 else 0.0
        return t_stat, float("nan")


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def run_single_seed(
    seed: int,
    train_tokens, train_labels,
    eval_tokens, eval_labels,
    label2id, id2label, label_order,
    tokenizer,
    args,
    seed_dir: str,
) -> Tuple[Dict, List[List[str]], List[List[str]], List[List[str]]]:
    """Train BERT (or reuse checkpoint) and evaluate both BERT and CVA.

    Returns
    -------
    metrics : dict   Scalar metrics for this seed.
    gold_aligned     Sentence-level gold labels (word-level alignment).
    bert_preds       BERT predicted labels (same alignment).
    cva_preds        CVA predicted labels (same alignment).
    """
    set_global_seed(seed)
    os.makedirs(seed_dir, exist_ok=True)
    bert_model_dir = os.path.join(seed_dir, "bert_model")

    train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    eval_dataset  = NERDataset(eval_tokens, eval_labels, tokenizer, label2id, args.max_length)

    seqeval = evaluate_lib.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels_batch = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        true_preds, true_labs = [], []
        for pred_row, lab_row in zip(predictions, labels_batch):
            ps, ls = [], []
            for p, l in zip(pred_row, lab_row):
                if int(l) == -100:
                    continue
                ps.append(id2label[int(p)])
                ls.append(id2label[int(l)])
            true_preds.append(ps)
            true_labs.append(ls)
        result = seqeval.compute(predictions=true_preds, references=true_labs) or {}
        return {
            "precision": float(result.get("overall_precision", 0.0)),
            "recall":    float(result.get("overall_recall", 0.0)),
            "f1":        float(result.get("overall_f1", 0.0)),
        }

    skip_train = args.skip_bert_train or is_transformers_model_dir(bert_model_dir)
    model_src  = bert_model_dir if (skip_train and is_transformers_model_dir(bert_model_dir)) else args.model_name

    model = AutoModelForTokenClassification.from_pretrained(
        model_src,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=bert_model_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=seed,
            data_seed=seed,
            report_to="none",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if not skip_train:
        trainer.train()
        trainer.save_model(bert_model_dir)

    bert_preds = trainer_predictions_to_labels(trainer, eval_dataset, id2label)

    gold_aligned: List[List[str]] = []
    for words, labs in zip(eval_tokens, eval_labels):
        _, aligned_labs = get_word_level_alignment(tokenizer, words, labs, args.max_length)
        gold_aligned.append(aligned_labs)

    bert_metrics = evaluate_token_classification(
        true_labels=gold_aligned,
        predicted_labels=bert_preds,
        label_order=label_order,
        include_o_label=False,
    )

    embedding_src = bert_model_dir if is_transformers_model_dir(bert_model_dir) else args.model_name
    embedder = TransformerEmbedder(model_name=embedding_src, device=args.device)
    class_vectors = build_cva_class_vectors(train_tokens, train_labels, embedder, args.max_length)

    _, cva_gold_aligned, cva_preds = predict_cva_labels(
        tokens=eval_tokens,
        labels=eval_labels,
        embedder=embedder,
        class_vectors=class_vectors,
        max_length=args.max_length,
    )

    cva_metrics = evaluate_token_classification(
        true_labels=cva_gold_aligned,
        predicted_labels=cva_preds,
        label_order=label_order,
        include_o_label=False,
    )

    seed_result = {
        "seed": seed,
        "bert_macro_f1": float(bert_metrics["macro_f1"]),
        "bert_accuracy": float(bert_metrics["accuracy"]),
        "cva_macro_f1": float(cva_metrics["macro_f1"]),
        "cva_accuracy": float(cva_metrics["accuracy"]),
    }

    with open(os.path.join(seed_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(seed_result, fh, indent=2, ensure_ascii=False)

    # Save predictions for later bootstrap analysis.
    preds_path = os.path.join(seed_dir, "predictions.json")
    with open(preds_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"gold": gold_aligned, "bert_preds": bert_preds, "cva_preds": cva_preds},
            fh,
            ensure_ascii=False,
        )

    return seed_result, gold_aligned, bert_preds, cva_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_tokens, train_labels_raw = read_conll_bio(args.train_file)
    eval_tokens,  eval_labels_raw  = read_conll_bio(args.eval_file)

    train_labels = normalize_bio_labels(train_labels_raw)
    eval_labels  = normalize_bio_labels(eval_labels_raw)

    label2id, id2label = build_label_maps([*train_labels, *eval_labels])
    label_order = [id2label[i] for i in range(len(id2label))]
    entity_labels = [l for l in label_order if l != "O"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    per_seed_results: List[Dict] = []
    last_gold: List[List[str]] = []
    last_bert_preds: List[List[str]] = []
    last_cva_preds: List[List[str]] = []

    for seed in args.seeds:
        print(f"\n[INFO] ===== Seed {seed} =====")
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        result, gold, bert_p, cva_p = run_single_seed(
            seed=seed,
            train_tokens=train_tokens,
            train_labels=train_labels,
            eval_tokens=eval_tokens,
            eval_labels=eval_labels,
            label2id=label2id,
            id2label=id2label,
            label_order=label_order,
            tokenizer=tokenizer,
            args=args,
            seed_dir=seed_dir,
        )
        per_seed_results.append(result)
        last_gold, last_bert_preds, last_cva_preds = gold, bert_p, cva_p
        print(
            f"[INFO] Seed {seed}: BERT macro-F1={result['bert_macro_f1']:.4f}, "
            f"CVA macro-F1={result['cva_macro_f1']:.4f}"
        )

    bert_f1s = [r["bert_macro_f1"] for r in per_seed_results]
    cva_f1s  = [r["cva_macro_f1"]  for r in per_seed_results]

    t_stat, t_pvalue = _paired_ttest(bert_f1s, cva_f1s)

    bootstrap = paired_bootstrap_test(
        gold_seqs=last_gold,
        pred_a_seqs=last_bert_preds,
        pred_b_seqs=last_cva_preds,
        entity_labels=entity_labels,
        n_bootstrap=args.n_bootstrap,
        rng_seed=0,
    )

    summary = {
        "seeds": args.seeds,
        "per_seed": per_seed_results,
        "bert": {
            "macro_f1_mean": float(np.mean(bert_f1s)),
            "macro_f1_std":  float(np.std(bert_f1s, ddof=1)) if len(bert_f1s) > 1 else 0.0,
            "macro_f1_values": bert_f1s,
        },
        "cva": {
            "macro_f1_mean": float(np.mean(cva_f1s)),
            "macro_f1_std":  float(np.std(cva_f1s, ddof=1)) if len(cva_f1s) > 1 else 0.0,
            "macro_f1_values": cva_f1s,
        },
        "paired_t_test_across_seeds": {
            "t_statistic": t_stat,
            "p_value_two_sided": t_pvalue,
            "note": "Tests H0: mean BERT F1 == mean CVA F1 across seeds (paired, two-sided).",
        },
        "paired_bootstrap_last_seed": bootstrap,
    }

    summary_path = os.path.join(args.output_dir, "multi_seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to {args.output_dir}")
    print(f"[INFO] BERT macro-F1: {summary['bert']['macro_f1_mean']:.4f} ± {summary['bert']['macro_f1_std']:.4f}")
    print(f"[INFO] CVA  macro-F1: {summary['cva']['macro_f1_mean']:.4f}  ± {summary['cva']['macro_f1_std']:.4f}")
    print(f"[INFO] Paired t-test p (two-sided): {t_pvalue:.4f}")
    print(f"[INFO] Bootstrap p (BERT better than CVA): {1 - bootstrap['p_value_a_better_than_b']:.4f}")


if __name__ == "__main__":
    main()
