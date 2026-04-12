"""
Download 15 real-world benchmark datasets for PPO model selector testing.

Groups:
  5 Classification  — real-world, medium/large, diverse classes
  5 Regression      — real-world, medium/large, continuous target
  5 Messy           — real-world with issues: missing values, noise, imbalance

None of these datasets are in the PPO training set (TRAINING_IDS checked).

Saves each dataset as:
    datasets/<name>.csv
    datasets/<name>_meta.json

Usage:
    python datasets/download_datasets.py
"""

import os
import sys
import json
import time
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# PPO training set IDs — must not appear in our test datasets
TRAINING_IDS = {
    31, 37, 44, 50, 54, 151, 182, 188, 1462, 1464, 1480, 1494, 1510,
    1461, 1489, 1590, 4534, 1501, 40496, 40668, 40670, 40701, 40975,
    40982, 40983, 40984, 40994, 41027, 23, 29, 38,
    507, 531, 546, 41021, 41980, 42225, 42570, 287, 42571, 42705,
    41187, 422, 505, 41702,
}

# Max rows to keep per dataset (keeps file sizes manageable)
MAX_ROWS = 60_000

# ==============================================================================
# Dataset registry
# ==============================================================================
DATASETS = [

    # ---------- CLASSIFICATION (5) -------------------------------------------
    {
        'name': 'covertype',
        'category': 'classification',
        'source': 'sklearn_covtype',
        'expected_rows': '581K (capped to 60K)', 'features': 54, 'classes': 7,
        'description': 'Forest cover type from cartographic variables (7 classes, UCI)',
    },
    {
        'name': 'har',
        'category': 'classification',
        'source': 'openml',
        'dataset_id': 1478,
        'expected_rows': '10K', 'features': 561, 'classes': 6,
        'description': 'Human Activity Recognition via smartphone sensors — 6 activity classes',
    },
    {
        'name': 'credit_card_fraud',
        'category': 'classification',
        'source': 'openml',
        'dataset_id': 1597,
        'expected_rows': '284K (capped to 60K)', 'features': 30, 'classes': 2,
        'description': 'Credit card fraud detection — real PCA features, 0.17% fraud (severe imbalance)',
    },
    {
        'name': 'mozilla4',
        'category': 'classification',
        'source': 'openml',
        'dataset_id': 1046,
        'expected_rows': '15K', 'features': 5, 'classes': 2,
        'description': 'Mozilla browser performance benchmark results (binary)',
    },
    {
        'name': 'eye_movements',
        'category': 'classification',
        'source': 'openml',
        'dataset_id': 1044,
        'expected_rows': '7K', 'features': 23, 'classes': 12,
        'description': 'Eye movement patterns during reading — 12 class fine-grained classification',
    },

    # ---------- REGRESSION (5) -----------------------------------------------
    {
        'name': 'california_housing',
        'category': 'regression',
        'source': 'sklearn_california',
        'expected_rows': '20K', 'features': 8,
        'description': 'California median house values — standard regression benchmark',
    },
    {
        'name': 'kin8nm',
        'category': 'regression',
        'source': 'openml',
        'dataset_id': 189,
        'expected_rows': '8K', 'features': 8,
        'description': 'Robot arm kinematics — predict end-effector distance (8 DOF)',
    },
    {
        'name': 'cpu_act',
        'category': 'regression',
        'source': 'openml',
        'dataset_id': 573,
        'expected_rows': '8K', 'features': 21,
        'description': 'Computer system CPU activity percentage (regression)',
    },
    {
        'name': 'abalone',
        'category': 'regression',
        'source': 'openml',
        'dataset_id': 183,
        'expected_rows': '4K', 'features': 8,
        'description': 'Abalone age from physical measurements — ring count prediction',
    },
    {
        'name': 'house_16H',
        'category': 'regression',
        'source': 'openml',
        'dataset_id': 574,
        'expected_rows': '22K', 'features': 16,
        'description': 'US house price median prediction (16 demographic features)',
    },

    # ---------- MESSY (5) ----------------------------------------------------
    {
        'name': 'kddcup09_appetency',
        'category': 'messy_classification',
        'source': 'openml',
        'dataset_id': 1111,
        'expected_rows': '50K', 'features': 230,
        'description': (
            'KDD Cup 2009 telecom churn — 230 features, -1 used as missing marker, '
            '~98% negative class (severe class imbalance)'
        ),
    },
    {
        'name': 'madelon',
        'category': 'messy_classification',
        'source': 'openml',
        'dataset_id': 1485,
        'expected_rows': '2.6K', 'features': 500,
        'description': (
            'Madelon adversarial benchmark — 500 features but only 20 are informative, '
            '480 are pure noise. Binary classification. Classic feature-noise stress test.'
        ),
    },
    {
        'name': 'nomao',
        'category': 'messy_classification',
        'source': 'openml',
        'dataset_id': 1486,
        'expected_rows': '34K', 'features': 118,
        'description': (
            'Nomao place deduplication — 118 features (numeric + categorical), '
            'binary, real-world entity matching with mixed data types'
        ),
    },
    {
        'name': 'amazon_employee_access',
        'category': 'messy_classification',
        'source': 'openml',
        'dataset_id': 4135,
        'expected_rows': '32K', 'features': 9,
        'description': (
            'Amazon employee access prediction — 9 purely categorical high-cardinality features, '
            'binary, ~94% positive class (imbalanced)'
        ),
    },
    {
        'name': 'numerai28',
        'category': 'messy_classification',
        'source': 'openml',
        'dataset_id': 23517,
        'expected_rows': '96K (capped to 60K)', 'features': 21,
        'description': (
            'Numerai28.6 stock market prediction — 21 obfuscated anonymous float features, '
            'binary target, balanced classes. Real financial data with no feature names.'
        ),
    },
]


# ==============================================================================
# Downloaders
# ==============================================================================

def _cap(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"     (sampled down to {max_rows:,} rows)")
    return df


def download_sklearn_covtype() -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype
    print("  Fetching from sklearn.datasets.fetch_covtype ...")
    data = fetch_covtype(as_frame=True)
    df = data.frame.rename(columns={'Cover_Type': 'target'})
    return _cap(df, MAX_ROWS)


def download_sklearn_california() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing
    print("  Fetching from sklearn.datasets.fetch_california_housing ...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={'MedHouseVal': 'target'})
    return df


def download_openml(dataset_id: int) -> pd.DataFrame:
    import openml
    assert dataset_id not in TRAINING_IDS, f"Dataset {dataset_id} is in PPO training set!"
    print(f"  Fetching OpenML dataset id={dataset_id} ...")
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    target_col = ds.default_target_attribute
    X, y, _, attribute_names = ds.get_data(
        dataset_format='dataframe',
        target=target_col,
    )
    df = X.copy()
    df['target'] = y
    return _cap(df, MAX_ROWS)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 65)
    print("  PPO Benchmark Dataset Downloader")
    print("  Saving to:", OUT_DIR)
    print("=" * 65)

    results = []
    success, failed = 0, 0

    for i, spec in enumerate(DATASETS, 1):
        name = spec['name']
        cat  = spec['category']
        src  = spec['source']

        print(f"\n[{i:2d}/15] {name}  ({cat})")
        print(f"       {spec['description'][:70]}")

        csv_path  = os.path.join(OUT_DIR, f"{name}.csv")
        meta_path = os.path.join(OUT_DIR, f"{name}_meta.json")

        # Skip if already downloaded
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, nrows=1)
            full = pd.read_csv(csv_path)
            print(f"  Already exists — {len(full):,} rows, {len(full.columns)-1} features. Skipping.")
            results.append({'name': name, 'status': 'skipped', 'rows': len(full)})
            success += 1
            continue

        t0 = time.time()
        try:
            if src == 'sklearn_covtype':
                df = download_sklearn_covtype()
            elif src == 'sklearn_california':
                df = download_sklearn_california()
            elif src == 'openml':
                df = download_openml(spec['dataset_id'])
            else:
                raise ValueError(f"Unknown source: {src}")

            # Ensure 'target' column exists and is last
            if 'target' not in df.columns:
                raise ValueError("No 'target' column found after download")

            cols = [c for c in df.columns if c != 'target'] + ['target']
            df = df[cols]

            # Save CSV
            df.to_csv(csv_path, index=False)

            # Compute quick stats
            target = df['target']
            n_rows, n_cols = df.shape
            n_features = n_cols - 1
            missing_pct = float(df.iloc[:, :-1].isnull().mean().mean() * 100)

            if cat == 'regression':
                target_info = {
                    'type': 'continuous',
                    'mean': float(target.astype(float).mean()),
                    'std': float(target.astype(float).std()),
                    'min': float(target.astype(float).min()),
                    'max': float(target.astype(float).max()),
                }
            else:
                vc = target.value_counts()
                target_info = {
                    'type': 'categorical',
                    'n_classes': int(target.nunique()),
                    'class_balance': {str(k): int(v) for k, v in vc.items()},
                    'majority_pct': float(vc.iloc[0] / len(target) * 100) if len(vc) > 0 else 0.0,
                }

            meta = {
                'name': name,
                'category': cat,
                'description': spec['description'],
                'rows': n_rows,
                'features': n_features,
                'missing_pct': round(missing_pct, 2),
                'target': target_info,
                'openml_dataset_id': spec.get('dataset_id', None),
                'in_ppo_training': False,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            elapsed = time.time() - t0
            print(f"  Saved: {n_rows:,} rows x {n_features} features  |  "
                  f"missing={missing_pct:.1f}%  |  {elapsed:.1f}s")
            results.append({'name': name, 'status': 'ok', 'rows': n_rows})
            success += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            results.append({'name': name, 'status': 'failed', 'error': str(e)})
            failed += 1

    # Summary
    print("\n" + "=" * 65)
    print(f"  Done — {success} succeeded, {failed} failed")
    print("=" * 65)
    print(f"\n{'Name':<30} {'Status':<10} {'Rows':>8}")
    print("-" * 52)
    for r in results:
        rows_str = f"{r.get('rows', 0):>8,}" if r.get('rows') else '       -'
        print(f"{r['name']:<30} {r['status']:<10} {rows_str}")

    # Save summary
    summary_path = os.path.join(OUT_DIR, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print(f"All CSVs in:      {OUT_DIR}")


if __name__ == '__main__':
    main()
