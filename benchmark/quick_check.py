"""
Quick sanity check — runs the benchmark on just 10 datasets to verify
everything works before launching the full 500-dataset run.

Usage:
    python -m benchmark.quick_check
"""
import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmark.ppo_benchmark import run_benchmark, generate_report

if __name__ == '__main__':
    print("Running quick check (10 classification + 5 regression datasets)...\n")

    df_clf = run_benchmark(
        task_type='classification',
        n_datasets=10,
        cv_folds=3,
        output_dir=os.path.join(os.path.dirname(__file__), 'results_quickcheck'),
        resume=False,
        seed=42,
    )

    df_reg = run_benchmark(
        task_type='regression',
        n_datasets=5,
        cv_folds=3,
        output_dir=os.path.join(os.path.dirname(__file__), 'results_quickcheck'),
        resume=False,
        seed=42,
    )

    print("\n[Quick Check] DONE")
    if df_clf is not None and not df_clf.empty:
        print(f"  Classification - {len(df_clf)} datasets processed")
        print(f"    Top-1 accuracy : {df_clf['top1_correct'].mean()*100:.1f}%")
        print(f"    Top-3 accuracy : {df_clf['top3_correct'].mean()*100:.1f}%")
        print(f"    Avg regret     : {df_clf['regret'].mean():.4f}")
    if df_reg is not None and not df_reg.empty:
        print(f"  Regression - {len(df_reg)} datasets processed")
        print(f"    Top-1 accuracy : {df_reg['top1_correct'].mean()*100:.1f}%")
        print(f"    Top-3 accuracy : {df_reg['top3_correct'].mean()*100:.1f}%")
        print(f"    Avg regret     : {df_reg['regret'].mean():.4f}")
