import time
import argparse
import os
import subprocess
import sys


def count_completed_folds(results_dir, num_folds):
    found = 0
    for i in range(num_folds):
        m = os.path.join(results_dir, f"fold_{i}", 'metrics_summary.csv')
        if os.path.isfile(m):
            found += 1
    return found


def run_aggregator(results_dir, num_folds):
    cmd = [sys.executable, '-m', 'scripts.aggregate_cv_results', '--results-dir', results_dir, '--num-folds', str(num_folds)]
    subprocess.run(cmd, check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='results/cv_full')
    p.add_argument('--num-folds', type=int, default=4)
    p.add_argument('--interval', type=int, default=600, help='Poll interval seconds')
    p.add_argument('--max-checks', type=int, default=24, help='Max polls before exit')
    args = p.parse_args()

    last_count = -1
    checks = 0
    while checks < args.max_checks:
        found = count_completed_folds(args.results_dir, args.num_folds)
        if found != last_count:
            print(f'Found {found}/{args.num_folds} completed folds; running aggregator...')
            run_aggregator(args.results_dir, args.num_folds)
            last_count = found
        if found >= args.num_folds:
            print('All folds found; exiting watcher.')
            return 0
        checks += 1
        time.sleep(args.interval)

    print('Max checks reached; exiting watcher.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
