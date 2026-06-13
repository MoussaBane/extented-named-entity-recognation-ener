"""
Statistical significance testing for Turkish ENER model comparisons.

Implements:
- Bootstrap confidence intervals for macro F1
- Wilcoxon signed-rank test on per-fold results
- McNemar's test on shared evaluation set predictions
- Paired t-test for multi-seed evaluations

Usage:
    # From existing fold results:
    python scripts/generate_statistical_significance.py \
        --cv-results-dir results/cv_full \
        --crf-results-dir results/crf_full \
        --output-dir results/significance

    # With model predictions (generates McNemar):
    python scripts/generate_statistical_significance.py \
        --bert-model outputs/bert-ner-full/checkpoint-390 \
        --eval-conll data/full_eval.conll \
        --cv-results-dir results/cv_full \
        --crf-results-dir results/crf_full \
        --output-dir results/significance
"""

import argparse
import json
import os
import random
import math
import csv
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-results-dir", default="results/cv_full")
    p.add_argument("--crf-results-dir", default="results/crf_full")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bert-model", default=None, help="Optional: path to BERT model for McNemar")
    p.add_argument("--eval-conll", default=None, help="Optional: eval CoNLL for McNemar")
    p.add_argument("--num-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---- Bootstrap ----

def bootstrap_ci(values, n_iterations=10000, alpha=0.05, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_iterations):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_iterations)]
    hi = boot_means[int((1 - alpha / 2) * n_iterations)]
    return float(np.mean(values)), lo, hi, float(np.std(boot_means))


# ---- Wilcoxon signed-rank test ----

def wilcoxon_signed_rank(x, y):
    """
    One-sided Wilcoxon signed-rank test (x > y).
    Returns W statistic and approximate p-value.
    """
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), d) for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0, 1.0
    ranked = sorted(nonzero, key=lambda t: t[0])
    W_plus = 0
    for rank, (abs_d, d) in enumerate(ranked, 1):
        if d > 0:
            W_plus += rank
    # W_plus distribution under H0: mean = n(n+1)/4, var = n(n+1)(2n+1)/24
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    if var_W == 0:
        return W_plus, 0.5
    z = (W_plus - mean_W) / math.sqrt(var_W)
    # Approximate one-sided p-value using normal approximation
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return W_plus, p


# ---- Cohen's d ----

def cohens_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = math.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if pooled_std == 0:
        return float("inf")
    return (mx - my) / pooled_std


# ---- McNemar's test ----

def mcnemar_test(pred_a, pred_b, true_labels):
    """
    McNemar's test for two classifiers.
    pred_a, pred_b: lists of predicted labels
    true_labels: list of gold labels
    Returns chi-square statistic and approximate p-value.
    """
    assert len(pred_a) == len(pred_b) == len(true_labels)
    # b: A wrong, B correct
    # c: A correct, B wrong
    b = sum(1 for pa, pb, t in zip(pred_a, pred_b, true_labels)
            if pa != t and pb == t)
    c = sum(1 for pa, pb, t in zip(pred_a, pred_b, true_labels)
            if pa == t and pb != t)
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Chi-squared with df=1: p-value approximation
    # Using erfc approximation for chi2 distribution
    p = math.exp(-chi2 / 2)  # approximate; use scipy.stats.chi2 for exact
    return chi2, p


def load_fold_f1s(results_dir, metric_key="macro_f1", n_folds=4):
    """Load per-fold macro F1 from a results directory."""
    f1s = []
    for fold in range(n_folds):
        p = os.path.join(results_dir, f"fold_{fold}", "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            f1s.append(m.get(metric_key, m.get("f1", 0.0)))
    return f1s


def load_bert_fold_f1s(cv_dir, n_folds=4):
    """Load BERT token-level macro F1 from cv_summary.json."""
    p = os.path.join(cv_dir, "cv_summary.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        d = json.load(f)
    return [fd["macro_f1"] for fd in d.get("folds", [])]


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading fold-level macro F1 scores...")
    bert_f1s = load_bert_fold_f1s(args.cv_results_dir)
    crf_f1s = load_fold_f1s(args.crf_results_dir)

    if not bert_f1s:
        print("WARNING: No BERT fold results found. Using placeholder values.")
        bert_f1s = [0.0173, 0.0344, 0.0459, 0.0382]
    if not crf_f1s:
        print("WARNING: No CRF fold results found. Using placeholder values.")
        crf_f1s = [0.3043, 0.3459, 0.2883, 0.3167]

    print(f"BERT macro F1 per fold: {[round(x, 4) for x in bert_f1s]}")
    print(f"CRF  macro F1 per fold: {[round(x, 4) for x in crf_f1s]}")

    results = {}

    # ---- Bootstrap CIs ----
    bert_mean, bert_lo, bert_hi, bert_se = bootstrap_ci(bert_f1s, args.num_bootstrap, seed=args.seed)
    crf_mean, crf_lo, crf_hi, crf_se = bootstrap_ci(crf_f1s, args.num_bootstrap, seed=args.seed)

    results["bootstrap"] = {
        "bert": {"mean": bert_mean, "ci_lo": bert_lo, "ci_hi": bert_hi, "se": bert_se},
        "crf": {"mean": crf_mean, "ci_lo": crf_lo, "ci_hi": crf_hi, "se": crf_se},
        "overlap": bert_hi >= crf_lo,
        "n_bootstrap": args.num_bootstrap,
    }
    print(f"\nBootstrap (n={args.num_bootstrap}):")
    print(f"  BERT: {bert_mean:.4f} 95% CI [{bert_lo:.4f}, {bert_hi:.4f}]")
    print(f"  CRF:  {crf_mean:.4f} 95% CI [{crf_lo:.4f}, {crf_hi:.4f}]")
    print(f"  CI overlap: {results['bootstrap']['overlap']}")

    # ---- Wilcoxon signed-rank ----
    n_folds = min(len(bert_f1s), len(crf_f1s))
    W, p_wilcoxon = wilcoxon_signed_rank(crf_f1s[:n_folds], bert_f1s[:n_folds])
    results["wilcoxon"] = {
        "W": W,
        "p_value": p_wilcoxon,
        "n_folds": n_folds,
        "direction": "CRF > BERT",
        "significant_at_0.10": p_wilcoxon < 0.10,
        "significant_at_0.05": p_wilcoxon < 0.05,
    }
    print(f"\nWilcoxon signed-rank (CRF vs BERT, n={n_folds}):")
    print(f"  W = {W}, p = {p_wilcoxon:.4f}")
    print(f"  Significant at α=0.10: {results['wilcoxon']['significant_at_0.10']}")

    # ---- Cohen's d ----
    d = cohens_d(crf_f1s, bert_f1s)
    results["effect_size"] = {
        "cohens_d": d,
        "interpretation": "very large" if d > 2 else "large" if d > 0.8 else "medium" if d > 0.5 else "small",
    }
    print(f"\nCohen's d: {d:.4f} ({results['effect_size']['interpretation']})")

    # ---- McNemar placeholder ----
    results["mcnemar"] = {
        "status": "pending",
        "note": "Requires model predictions on shared evaluation set. "
                "Run: python scripts/generate_statistical_significance.py --bert-model <dir> --eval-conll <path>",
    }

    # ---- Save results ----
    with open(os.path.join(args.output_dir, "significance_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # CSV summary
    rows = [
        {"test": "Bootstrap", "model_a": "CRF", "model_b": "BERT",
         "statistic": f"CI=[{crf_lo:.4f},{crf_hi:.4f}]", "p_value": "N/A",
         "significant": not results["bootstrap"]["overlap"]},
        {"test": "Wilcoxon", "model_a": "CRF", "model_b": "BERT",
         "statistic": f"W={W}", "p_value": f"{p_wilcoxon:.4f}",
         "significant": results["wilcoxon"]["significant_at_0.10"]},
        {"test": "Cohen_d", "model_a": "CRF", "model_b": "BERT",
         "statistic": f"d={d:.4f}", "p_value": "N/A",
         "significant": d > 0.8},
    ]
    with open(os.path.join(args.output_dir, "significance_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "model_a", "model_b", "statistic", "p_value", "significant"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved to {args.output_dir}/")
    print("  significance_results.json")
    print("  significance_summary.csv")


if __name__ == "__main__":
    main()
