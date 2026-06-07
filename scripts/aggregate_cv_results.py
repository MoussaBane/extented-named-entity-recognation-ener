import json
import os
import csv
import argparse
from statistics import mean, stdev


def read_metrics_csv(path):
    metrics = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) >= 2:
                key = row[0]
                try:
                    val = float(row[1])
                except Exception:
                    val = row[1]
                metrics[key] = val
    return metrics


def aggregate(results_dir, num_folds=4):
    folds = []
    for i in range(num_folds):
        fold_dir = os.path.join(results_dir, f"fold_{i}")
        if os.path.isdir(fold_dir):
            metrics_file = os.path.join(fold_dir, 'metrics_summary.csv')
            if os.path.isfile(metrics_file):
                m = read_metrics_csv(metrics_file)
                m['fold'] = i
                folds.append(m)

    if not folds:
        print('No completed folds found.')
        return 1

    # determine numeric keys
    numeric_keys = [k for k,v in folds[0].items() if k != 'fold' and isinstance(v, (int,float))]
    summary = {'num_folds_found': len(folds), 'folds': folds, 'aggregates': {}}

    for k in numeric_keys:
        vals = [f[k] for f in folds if isinstance(f.get(k), (int,float))]
        if vals:
            summary['aggregates'][k] = {'mean': mean(vals), 'std': stdev(vals) if len(vals) > 1 else 0.0}

    out_json = os.path.join(results_dir, 'cv_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Wrote {out_json} (folds found: {len(folds)})')
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='results/cv_full')
    p.add_argument('--num-folds', type=int, default=4)
    args = p.parse_args()
    return aggregate(args.results_dir, args.num_folds)


if __name__ == '__main__':
    raise SystemExit(main())
