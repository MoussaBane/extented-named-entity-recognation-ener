"""Wait for CV and CRF full runs to complete, then run post experiments.

This script polls `results/cv_full` and `results/crf_full` for completed fold outputs
and, once all folds are present, runs `scripts.oov_experiment` and
`scripts.compare_context_cva` on the full dataset.
"""
import time
import os
import subprocess
import argparse


def count_completed_folds(results_dir, num_folds):
    found = 0
    for i in range(num_folds):
        m = os.path.join(results_dir, f"fold_{i}", 'metrics_summary.csv')
        if os.path.isfile(m):
            found += 1
    return found


def run_module(module, args_list):
    cmd = ['python', '-m', module] + args_list
    print('Running:', ' '.join(cmd))
    return subprocess.run(cmd, check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cv-dir', default='results/cv_full')
    p.add_argument('--crf-dir', default='results/crf_full')
    p.add_argument('--num-folds', type=int, default=4)
    p.add_argument('--interval', type=int, default=600)
    p.add_argument('--max-checks', type=int, default=48)
    args = p.parse_args()

    checks = 0
    while checks < args.max_checks:
        cv_found = count_completed_folds(args.cv_dir, args.num_folds)
        crf_found = count_completed_folds(args.crf_dir, args.num_folds)
        print(f'Poll {checks}: CV {cv_found}/{args.num_folds}, CRF {crf_found}/{args.num_folds}')
        if cv_found >= args.num_folds and crf_found >= args.num_folds:
            print('Both CV and CRF complete — starting post experiments')

            # Run OOV experiment
            run_module('scripts.oov_experiment', [
                '--train-file', 'data/full_train.conll',
                '--eval-file', 'data/full_eval.conll',
                '--output-dir', 'results/oov_full'
            ])

            # Run compare_context_cva
            run_module('scripts.compare_context_cva', [
                '--train-file', 'data/full_train.conll',
                '--eval-file', 'data/full_eval.conll',
                '--output-dir', 'results/compare_full'
            ])

            print('Post experiments finished.')
            return 0

        checks += 1
        time.sleep(args.interval)

    print('Max checks reached without both completions.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
