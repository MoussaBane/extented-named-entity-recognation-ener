"""Generate a comprehensive thesis-ready report from all experiment results.

Produces both Markdown and LaTeX versions of all result tables and summaries,
pulling from CV, CRF, comparison, OOV and embedding result directories.

Usage:
    python scripts/generate_thesis_summary.py \
        --cv-dir results/cv_full \
        --crf-dir results/crf_full \
        --compare-dir results/compare_full \
        --bert-cva-dir results/model_comparison_full_cleaned \
        --oov-dir results/oov_full \
        --embed-dir results/embedding_full \
        --out-dir results/thesis_report
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-dir", default="results/cv_full")
    p.add_argument("--crf-dir", default="results/crf_full")
    p.add_argument("--compare-dir", default="results/compare_full")
    p.add_argument("--bert-cva-dir", default="results/model_comparison_full_cleaned")
    p.add_argument("--oov-dir", default="results/oov_full")
    p.add_argument("--embed-dir", default="results/embedding_full")
    p.add_argument("--attention-dir", default="results/attention_ner")
    p.add_argument("--out-dir", default="results/thesis_report")
    return p.parse_args()


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt(val, decimals=4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def fmt_pm(mean, std, decimals=4) -> str:
    if mean is None:
        return "N/A"
    return f"{float(mean):.{decimals}f} ± {float(std):.{decimals}f}"


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers: List[str], rows: List[List[str]]) -> str:
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    data_rows = ["| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_row, sep] + data_rows)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def latex_table(caption: str, label: str, headers: List[str], rows: List[List[str]]) -> str:
    col_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(c) for c in row) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_bert_cv_section(cv_dir: str):
    summary = load_json(os.path.join(cv_dir, "cv_summary.json"))
    if not summary:
        return None, None, "*(BERT CV results not found)*"

    folds = summary.get("folds", [])
    agg = summary.get("aggregates", {})

    md_rows = []
    for f in folds:
        fold_id = f.get("fold", folds.index(f))
        md_rows.append([
            f"Fold {fold_id}",
            fmt(f.get("macro_precision"), 4),
            fmt(f.get("macro_recall"), 4),
            fmt(f.get("macro_f1"), 4),
            fmt(f.get("accuracy"), 4),
            str(int(f.get("support", 0))),
        ])
    md_rows.append([
        "**Mean ± Std**",
        fmt_pm(agg.get("macro_precision", {}).get("mean"), agg.get("macro_precision", {}).get("std")),
        fmt_pm(agg.get("macro_recall", {}).get("mean"), agg.get("macro_recall", {}).get("std")),
        fmt_pm(agg.get("macro_f1", {}).get("mean"), agg.get("macro_f1", {}).get("std")),
        fmt_pm(agg.get("accuracy", {}).get("mean"), agg.get("accuracy", {}).get("std")),
        fmt(agg.get("support", {}).get("mean"), 1),
    ])

    headers = ["Fold", "Precision", "Recall", "Macro F1", "Accuracy", "Support"]
    md = md_table(headers, md_rows)

    latex_rows = []
    for f in folds:
        fold_id = f.get("fold", folds.index(f))
        latex_rows.append([
            f"Fold {fold_id}",
            fmt(f.get("macro_precision"), 4),
            fmt(f.get("macro_recall"), 4),
            fmt(f.get("macro_f1"), 4),
            fmt(f.get("accuracy"), 4),
        ])
    latex_rows.append([
        "Mean $\\pm$ Std",
        fmt_pm(agg.get("macro_precision", {}).get("mean"), agg.get("macro_precision", {}).get("std")),
        fmt_pm(agg.get("macro_recall", {}).get("mean"), agg.get("macro_recall", {}).get("std")),
        fmt_pm(agg.get("macro_f1", {}).get("mean"), agg.get("macro_f1", {}).get("std")),
        fmt_pm(agg.get("accuracy", {}).get("mean"), agg.get("accuracy", {}).get("std")),
    ])
    latex = latex_table(
        "BERT NER 4-Fold Cross-Validation Results",
        "tab:bert_cv",
        ["Fold", "Precision", "Recall", "Macro F1", "Accuracy"],
        latex_rows,
    )
    return md, latex, None


def build_crf_cv_section(crf_dir: str):
    summary = load_json(os.path.join(crf_dir, "crf_cv_summary.json"))
    if not summary:
        return None, None, "*(CRF CV results not found)*"

    # prefer rich folds+aggregates format; fall back to reading per-fold metrics.json
    folds = summary.get("folds", [])
    agg = summary.get("aggregates", {})
    if not folds:
        num = summary.get("num_folds", summary.get("num_folds_found", 4))
        for i in range(num):
            fold_m = load_json(os.path.join(crf_dir, f"fold_{i}", "metrics.json"))
            if fold_m:
                folds.append({
                    "fold": i,
                    "macro_precision": fold_m.get("macro_precision", 0.0),
                    "macro_recall": fold_m.get("macro_recall", 0.0),
                    "macro_f1": fold_m.get("macro_f1", 0.0),
                    "accuracy": fold_m.get("accuracy", 0.0),
                    "support": fold_m.get("support", 0),
                })

    md_rows = []
    for f in folds:
        fold_id = f.get("fold", folds.index(f))
        md_rows.append([
            f"Fold {fold_id}",
            fmt(f.get("macro_precision"), 4),
            fmt(f.get("macro_recall"), 4),
            fmt(f.get("macro_f1"), 4),
            fmt(f.get("accuracy"), 4),
            str(int(f.get("support", 0))),
        ])
    md_rows.append([
        "**Mean ± Std**",
        fmt_pm(agg.get("macro_precision", {}).get("mean"), agg.get("macro_precision", {}).get("std")),
        fmt_pm(agg.get("macro_recall", {}).get("mean"), agg.get("macro_recall", {}).get("std")),
        fmt_pm(agg.get("macro_f1", {}).get("mean"), agg.get("macro_f1", {}).get("std")),
        fmt_pm(agg.get("accuracy", {}).get("mean"), agg.get("accuracy", {}).get("std")),
        fmt(agg.get("support", {}).get("mean"), 1),
    ])

    headers = ["Fold", "Precision", "Recall", "Macro F1", "Accuracy", "Support"]
    md = md_table(headers, md_rows)

    latex_rows = []
    for f in folds:
        fold_id = f.get("fold", folds.index(f))
        latex_rows.append([
            f"Fold {fold_id}",
            fmt(f.get("macro_precision"), 4),
            fmt(f.get("macro_recall"), 4),
            fmt(f.get("macro_f1"), 4),
            fmt(f.get("accuracy"), 4),
        ])
    latex_rows.append([
        "Mean $\\pm$ Std",
        fmt_pm(agg.get("macro_precision", {}).get("mean"), agg.get("macro_precision", {}).get("std")),
        fmt_pm(agg.get("macro_recall", {}).get("mean"), agg.get("macro_recall", {}).get("std")),
        fmt_pm(agg.get("macro_f1", {}).get("mean"), agg.get("macro_f1", {}).get("std")),
        fmt_pm(agg.get("accuracy", {}).get("mean"), agg.get("accuracy", {}).get("std")),
    ])
    latex = latex_table(
        "CRF Baseline 4-Fold Cross-Validation Results",
        "tab:crf_cv",
        ["Fold", "Precision", "Recall", "Macro F1", "Accuracy"],
        latex_rows,
    )
    return md, latex, None


def build_bert_vs_cva_section(bert_cva_dir: str):
    summary = load_json(os.path.join(bert_cva_dir, "comparison_summary.json"))
    if not summary:
        return None, None, "*(BERT vs CVA comparison not found)*"

    bert = summary.get("bert", {})
    cva = summary.get("cva", {})

    rows = [
        ["BERT (Transformer)", fmt(bert.get("macro_f1_with_o"), 4), fmt(bert.get("macro_f1_without_o"), 4), fmt(bert.get("accuracy_with_o"), 4), fmt(bert.get("timing", {}).get("average_seconds"), 5)],
        ["CVA (Common Vector)", fmt(cva.get("macro_f1_with_o"), 4), fmt(cva.get("macro_f1_without_o"), 4), fmt(cva.get("accuracy_with_o"), 4), fmt(cva.get("timing", {}).get("average_seconds"), 5)],
    ]
    headers = ["Method", "Macro F1 (w/ O)", "Macro F1 (w/o O)", "Accuracy", "Avg Infer (s)"]
    md = md_table(headers, rows)
    latex = latex_table(
        "BERT vs. CVA Classification Comparison",
        "tab:bert_vs_cva",
        headers,
        rows,
    )
    return md, latex, None


def build_context_cva_comparison_section(compare_dir: str):
    summary_path = os.path.join(compare_dir, "comparison_summary.json")
    if not os.path.exists(summary_path):
        return None, None, "*(Context vs CVA comparison not found)*"

    # comparison_summary can be large (embeddings); read only top-level keys
    ctx_report = load_json(os.path.join(compare_dir, "context_only", "classification_report.json"))
    cva_report = load_json(os.path.join(compare_dir, "cva_only", "classification_report.json"))
    comb_report = load_json(os.path.join(compare_dir, "combined", "classification_report.json"))

    def get_metrics(report):
        if not report:
            return {"macro_f1": None, "macro_precision": None, "macro_recall": None, "accuracy": None}
        return {
            "macro_f1": report.get("macro_f1"),
            "macro_precision": report.get("macro_precision"),
            "macro_recall": report.get("macro_recall"),
            "accuracy": report.get("accuracy"),
        }

    ctx_m = get_metrics(ctx_report)
    cva_m = get_metrics(cva_report)
    comb_m = get_metrics(comb_report)

    rows = [
        ["Context-only (mean vectors)", fmt(ctx_m["macro_precision"], 4), fmt(ctx_m["macro_recall"], 4), fmt(ctx_m["macro_f1"], 4), fmt(ctx_m["accuracy"], 4)],
        ["CVA-only (common vectors)", fmt(cva_m["macro_precision"], 4), fmt(cva_m["macro_recall"], 4), fmt(cva_m["macro_f1"], 4), fmt(cva_m["accuracy"], 4)],
        ["Combined (CVA + Context)", fmt(comb_m["macro_precision"], 4), fmt(comb_m["macro_recall"], 4), fmt(comb_m["macro_f1"], 4), fmt(comb_m["accuracy"], 4)],
    ]
    headers = ["Method", "Precision", "Recall", "Macro F1", "Accuracy"]
    md = md_table(headers, rows)
    latex = latex_table(
        "Context-only vs CVA vs Combined Classification (Req K, L, M)",
        "tab:context_cva_combined",
        headers,
        rows,
    )
    return md, latex, None


def build_oov_section(oov_dir: str):
    summary = load_json(os.path.join(oov_dir, "oov_summary.json"))
    if not summary:
        return None, None, "*(OOV experiment results not found)*"

    total = summary.get("total_oov", "N/A")

    # try to read per-label accuracy from oov_results.csv if present
    csv_path = os.path.join(oov_dir, "oov_results.csv")
    top1_correct = 0
    top3_correct = 0
    total_rows = 0
    if os.path.exists(csv_path):
        import csv
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1
                    gold = row.get("gold_label", "")
                    top1 = row.get("top1", "")
                    top3_str = row.get("top3", "[]")
                    if top1 == gold:
                        top1_correct += 1
                    try:
                        top3 = json.loads(top3_str)
                        if any(item[0] == gold for item in top3):
                            top3_correct += 1
                    except Exception:
                        pass
        except Exception:
            pass

    rows = [["Total OOV entity tokens", str(total)]]
    if total_rows > 0:
        rows.append(["Top-1 accuracy", f"{top1_correct}/{total_rows} = {top1_correct/total_rows:.4f}"])
        rows.append(["Top-3 accuracy", f"{top3_correct}/{total_rows} = {top3_correct/total_rows:.4f}"])

    headers = ["Metric", "Value"]
    md = md_table(headers, rows)
    latex = latex_table(
        "OOV Entity Experiment Summary (Req I, J)",
        "tab:oov",
        headers,
        rows,
    )
    return md, latex, None


def build_attention_ner_section(attention_dir: str):
    summary = load_json(os.path.join(attention_dir, "comparison_summary.json"))
    if not summary:
        return None, None, "*(AttentionNER results not found — run scripts/run_attention_ner.py)*"

    wo = summary.get("without_O", {})
    wi = summary.get("with_O", {})
    config_rows = [
        ["BERT base", summary.get("bert_base", "N/A")],
        ["Attention head dim (d_head)", str(summary.get("d_head", "N/A"))],
        ["Freeze BERT", str(summary.get("freeze_bert", "N/A"))],
        ["Train sentences", str(summary.get("num_train_sentences", "N/A"))],
        ["Eval sentences", str(summary.get("num_eval_sentences", "N/A"))],
        ["Label count", str(summary.get("num_labels", "N/A"))],
    ]
    metric_rows = [
        ["Macro Precision (w/o O)", fmt(wo.get("macro_precision"), 4)],
        ["Macro Recall (w/o O)", fmt(wo.get("macro_recall"), 4)],
        ["Macro F1 (w/o O)", fmt(wo.get("macro_f1"), 4)],
        ["Accuracy (w/ O)", fmt(wi.get("accuracy"), 4)],
        ["Macro F1 (w/ O)", fmt(wi.get("macro_f1"), 4)],
    ]
    all_rows = config_rows + [["---", "---"]] + metric_rows
    headers = ["Parameter / Metric", "Value"]
    md = md_table(headers, all_rows)

    # per-label table from CSV if available
    per_label_path = os.path.join(attention_dir, "per_label_report.csv")
    if os.path.exists(per_label_path):
        import csv
        label_rows = []
        try:
            with open(per_label_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("label", "O") == "O":
                        continue
                    if int(row.get("support", 0)) == 0:
                        continue
                    label_rows.append([
                        row.get("label", ""),
                        row.get("precision", ""),
                        row.get("recall", ""),
                        row.get("f1", ""),
                        row.get("support", ""),
                    ])
        except Exception:
            pass
        if label_rows:
            label_rows_sorted = sorted(label_rows, key=lambda r: -float(r[4]))[:20]
            md += "\n\n### Top-20 Entity Labels by Support\n\n"
            md += md_table(
                ["Label", "Precision", "Recall", "F1", "Support"],
                label_rows_sorted,
            )

    latex = latex_table(
        "AttentionNER: Learned Q/K/V Self-Attention for Turkish ENER",
        "tab:attention_ner",
        headers,
        config_rows + metric_rows,
    )
    return md, latex, None


def build_model_comparison_summary(bert_cva_dir: str, attention_dir: str, crf_dir: str):
    bert_cva = load_json(os.path.join(bert_cva_dir, "comparison_summary.json"))
    attn = load_json(os.path.join(attention_dir, "comparison_summary.json"))
    crf = load_json(os.path.join(crf_dir, "crf_cv_summary.json"))

    if not any([bert_cva, attn, crf]):
        return None, None

    rows = []
    if crf:
        agg = crf.get("aggregates", {})
        rows.append([
            "CRF Baseline",
            fmt(agg.get("macro_precision", {}).get("mean"), 4),
            fmt(agg.get("macro_recall", {}).get("mean"), 4),
            fmt_pm(agg.get("macro_f1", {}).get("mean"), agg.get("macro_f1", {}).get("std")),
            "4-fold CV",
        ])
    if bert_cva:
        bert = bert_cva.get("bert", {})
        rows.append([
            "BERT Fine-tune",
            fmt(bert.get("macro_f1_with_o"), 4),
            "—",
            fmt(bert.get("macro_f1_without_o"), 4),
            "single run",
        ])
        cva = bert_cva.get("cva", {})
        rows.append([
            "CVA (Common Vector)",
            fmt(cva.get("macro_f1_with_o"), 4),
            "—",
            fmt(cva.get("macro_f1_without_o"), 4),
            "single run",
        ])
    if attn:
        wo = attn.get("without_O", {})
        frozen = "frozen BERT" if attn.get("freeze_bert") else "full fine-tune"
        rows.append([
            f"AttentionNER ({frozen})",
            fmt(wo.get("macro_precision"), 4),
            fmt(wo.get("macro_recall"), 4),
            fmt(wo.get("macro_f1"), 4),
            "single run",
        ])

    headers = ["Model", "Macro Precision", "Macro Recall", "Macro F1 (w/o O)", "Notes"]
    md = md_table(headers, rows)
    latex = latex_table(
        "Overall Model Comparison — Turkish ENER",
        "tab:model_comparison",
        headers,
        rows,
    )
    return md, latex


def build_embedding_section(embed_dir: str):
    experiment = load_json(os.path.join(embed_dir, "experiment.json"))
    cls_report = load_json(os.path.join(embed_dir, "classification_report.json"))

    ev_path = os.path.join(embed_dir, "pca_explained_variance.txt")
    ev_text = None
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            ev_text = f.read().strip()

    rows = []
    if experiment:
        events = experiment.get("events", [])
        for ev in events:
            if ev.get("event") == "train_extraction":
                rows.append(["Train tokens extracted", str(ev.get("data", {}).get("num_records", "N/A"))])
            elif ev.get("event") == "eval_extraction":
                rows.append(["Eval tokens extracted", str(ev.get("data", {}).get("num_records", "N/A"))])
            elif ev.get("event") == "class_vectors_saved":
                rows.append(["Class vectors", str(ev.get("data", {}).get("n_classes", "N/A"))])

    if cls_report:
        rows.append(["CVA embed accuracy", fmt(cls_report.get("accuracy"), 4)])
        rows.append(["CVA embed macro F1", fmt(cls_report.get("macro_f1"), 4)])

    if not rows:
        return None, None, "*(Embedding experiment results not found)*"

    headers = ["Metric", "Value"]
    md = md_table(headers, rows)
    if ev_text:
        md += "\n\n**PCA Explained Variance (2 components):**\n\n```\n" + ev_text + "\n```"

    latex = latex_table(
        "Embedding Extraction and CVA Classification Summary",
        "tab:embedding",
        headers,
        rows,
    )
    return md, latex, None


# ---------------------------------------------------------------------------
# Top-N per-class table across methods
# ---------------------------------------------------------------------------

def build_top_labels_comparison(compare_dir: str, top_n: int = 10):
    ctx_report = load_json(os.path.join(compare_dir, "context_only", "classification_report.json"))
    cva_report = load_json(os.path.join(compare_dir, "cva_only", "classification_report.json"))
    comb_report = load_json(os.path.join(compare_dir, "combined", "classification_report.json"))

    if not ctx_report:
        return None, None

    per_class_ctx = ctx_report.get("per_class", {})
    per_class_cva = cva_report.get("per_class", {}) if cva_report else {}
    per_class_comb = comb_report.get("per_class", {}) if comb_report else {}

    # rank labels by support in ctx report descending
    sorted_labels = sorted(per_class_ctx.items(), key=lambda x: -x[1].get("support", 0))[:top_n]

    rows = []
    for lbl, ctx_vals in sorted_labels:
        cva_vals = per_class_cva.get(lbl, {})
        comb_vals = per_class_comb.get(lbl, {})
        rows.append([
            lbl,
            str(int(ctx_vals.get("support", 0))),
            fmt(ctx_vals.get("f1"), 3),
            fmt(cva_vals.get("f1"), 3),
            fmt(comb_vals.get("f1"), 3),
        ])

    headers = ["Label", "Support", "F1 (Context)", "F1 (CVA)", "F1 (Combined)"]
    md = md_table(headers, rows)
    latex = latex_table(
        f"Per-class F1 for Top-{top_n} Most Frequent Labels",
        "tab:per_class_top",
        headers,
        rows,
    )
    return md, latex


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    md_sections = ["# Thesis Results Report\n\nGenerated from experiment results.\n"]
    latex_sections = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=2cm}",
        r"\begin{document}",
        "",
        r"\section*{Thesis Results Tables}",
        "",
    ]

    def add_section(title: str, anchor: str, md_content, latex_content, note=None):
        md_sections.append(f"\n## {title}\n")
        if note:
            md_sections.append(note + "\n")
        elif md_content:
            md_sections.append(md_content + "\n")
        if latex_content:
            latex_sections.append(f"\n% {title}")
            latex_sections.append(latex_content)

    # BERT 4-fold CV
    md_cv, latex_cv, cv_note = build_bert_cv_section(args.cv_dir)
    add_section("BERT 4-Fold Cross-Validation (Req A, B)", "bert_cv", md_cv, latex_cv, cv_note)

    # CRF 4-fold CV
    md_crf, latex_crf, crf_note = build_crf_cv_section(args.crf_dir)
    add_section("CRF Baseline 4-Fold Cross-Validation (Req E)", "crf_cv", md_crf, latex_crf, crf_note)

    # BERT vs CVA single run
    md_bv, latex_bv, bv_note = build_bert_vs_cva_section(args.bert_cva_dir)
    add_section("BERT vs CVA Single-Run Comparison (Req D, G)", "bert_vs_cva", md_bv, latex_bv, bv_note)

    # Context-only vs CVA vs Combined
    md_cc, latex_cc, cc_note = build_context_cva_comparison_section(args.compare_dir)
    add_section("Context-only vs CVA vs Combined (Req K, L, M)", "context_cva", md_cc, latex_cc, cc_note)

    # Top-N per-class breakdown
    md_top, latex_top = build_top_labels_comparison(args.compare_dir, top_n=10)
    if md_top:
        add_section("Per-class F1: Top-10 Most Frequent Labels (Req C)", "per_class", md_top, latex_top)

    # OOV experiment
    md_oov, latex_oov, oov_note = build_oov_section(args.oov_dir)
    add_section("OOV Entity Experiment — Top-3 Retrieval (Req I, J)", "oov", md_oov, latex_oov, oov_note)

    # Embedding analysis
    md_emb, latex_emb, emb_note = build_embedding_section(args.embed_dir)
    add_section("Embedding Extraction & CVA Analysis (Req F, H, N)", "embedding", md_emb, latex_emb, emb_note)

    # AttentionNER (Q/K/V attention head)
    md_attn, latex_attn, attn_note = build_attention_ner_section(args.attention_dir)
    add_section("AttentionNER — Learned Q/K/V Self-Attention (Req O)", "attention_ner", md_attn, latex_attn, attn_note)

    # Overall model comparison table
    md_comp, latex_comp = build_model_comparison_summary(args.bert_cva_dir, args.attention_dir, args.crf_dir)
    if md_comp:
        add_section("Overall Model Comparison", "model_comparison", md_comp, latex_comp)

    latex_sections.append(r"\end{document}")

    # write markdown
    md_path = os.path.join(args.out_dir, "thesis_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))
    print(f"Markdown report: {md_path}")

    # write LaTeX
    latex_path = os.path.join(args.out_dir, "thesis_report.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_sections))
    print(f"LaTeX report:    {latex_path}")


if __name__ == "__main__":
    main()
