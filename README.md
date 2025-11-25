 # Turkish Extended NER — Annotation Statistics

This repository provides reproducible corpus-level statistics and visualizations for
an extended Named Entity Recognition (NER) annotation collection in Turkish. The
primary intent is dataset quality inspection and preparation of downstream tasks
such as character-level boundary detection and contrastive learning experiments.

Note: The original README has been preserved unchanged as `README.ORIGINAL.md`.
If you prefer the original layout or content, see that file.

Supported features:

- Parsing INCEpTION/CoNLL-style annotations under `data/annotation/`.
- Computing corpus statistics (tokens, sentences, entity counts, label distributions).
- Tagset quality checks comparing `data/ENER-tagset.tsv` with annotations.
- Exporting numerical summaries (`stats.json`, CSVs) and generating plots (`results/plots/`).

Repository layout (relevant files)

- `data/annotation/` — input: document subfolders containing `.conll` annotation files.
- `data/ENER-tagset.tsv` — canonical tagset used for QC.
- `ner_stats/` — library code: `conll_reader.py`, `statistics.py`, `tagset.py`.
- `scripts/run_analysis.py` — CLI entrypoint for the analysis pipeline.
- `results/` — output directory for JSON, CSVs, and generated plots.
- `requirements.txt` — Python dependencies.

Prerequisites

- Python 3.8 or newer (3.10+ recommended).
- Git (optional) to clone the repository.

Installation (recommended)

Windows — PowerShell (copy/paste):

```powershell
# from project root
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Running the analysis

Option 1 — call the venv Python directly (no activation required):

```powershell
& "C:\full\path\to\project\\.venv\Scripts\python.exe" scripts/run_analysis.py --data-root data/annotation --results-dir results
```

Option 2 — module invocation (recommended):

```powershell
$env:PYTHONPATH = "C:\full\path\to\project"
& "C:\full\path\to\project\.venv\Scripts\python.exe" -m scripts.run_analysis --data-root data/annotation --results-dir results
```

Or on macOS / Linux:

```bash
export PYTHONPATH="/full/path/to/project"
/full/path/to/project/.venv/bin/python -m scripts.run_analysis --data-root data/annotation --results-dir results
```

Notes on Windows paths

- PowerShell treats space-separated tokens as separate commands. Use the call operator
  `&` and quote full paths containing spaces.
- If activation is blocked by policy, the `Set-ExecutionPolicy` line above sets a
  temporary process-level bypass.

Expected outputs

- `results/stats.json` — corpus statistics (JSON).
- `results/label_counts.csv` — per-label counts.
- `results/type_counts.csv` — aggregated type counts.
- `results/plots/` — PNG files with common visualizations.
- `results/unused_tags_in_corpus.txt` and `results/unknown_types_in_tagset_comparison.txt` — tagset QC reports.

Example run (what to expect)

After a successful run you should see console messages similar to:

```
[INFO] Results will be saved under: results
[INFO] Saved stats.json, label_counts.csv, and type_counts.csv
[INFO] Plots generated in results\plots
```

Development notes

- `ner_stats/__init__.py` and `scripts/__init__.py` are present to allow module
  invocation with `python -m scripts.run_analysis`.
- If you reorganize the project, either update `PYTHONPATH` or install the package
  in editable mode (`pip install -e .`) with a `pyproject.toml` / `setup.py`.

Troubleshooting

- Module import errors: ensure you run from the project root or set `PYTHONPATH`.
- Activation errors on PowerShell: use the `Set-ExecutionPolicy -Scope Process` command
  shown above to temporarily allow activation.
- If `pip install -r requirements.txt` fails, verify your network and Python version.

Contributing

Contributions are welcome. Suggested tasks:

- Add unit tests for parsing and statistics functions.
- Add CI to validate a minimal analysis run on a small fixture dataset.
- Provide convenience entry scripts: `run_analysis.ps1` and `run_analysis.sh`.

If you open a pull request, include a short description, tests or a reproducible
example, and the platform(s) you tested on.

License & contact

This repository does not include a license file — add one if you intend to publish.
Open an issue for questions or to request features.

---

(Rewritten README: detailed, professional instructions for users and developers.)