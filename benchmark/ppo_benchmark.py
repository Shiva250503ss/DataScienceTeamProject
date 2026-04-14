"""
PPO Model Selector Benchmark — 500 Unseen OpenML Datasets

Validates the PPO model selector by evaluating it on datasets NOT used
during training, comparing its recommendations against:
  1. Random selection      — uniform random model choice
  2. Always-RandomForest   — fixed baseline (usually strong default)
  3. Always-GradientBoosting — another strong fixed baseline
  4. Oracle                — always picks the true best model (upper bound)

Metrics reported per dataset and as aggregate:
  - Top-1 Accuracy  : PPO's #1 pick == best model on this dataset
  - Top-3 Accuracy  : any of PPO's top-3 picks == best model
  - Avg Regret      : mean(oracle_score - ppo_top1_score)  [lower = better]
  - Avg Rank        : mean rank position of PPO's top-1 pick
  - Win Rate vs RF  : % datasets where PPO's pick >= RF score

Usage:
    python -m benchmark.ppo_benchmark
    python -m benchmark.ppo_benchmark --task classification --n 300
    python -m benchmark.ppo_benchmark --task regression --n 200
    python -m benchmark.ppo_benchmark --task both --n 500 --resume
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Add project root to path ─────────────────────────────────────────────────
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BENCH_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─── Project imports ──────────────────────────────────────────────────────────
from meta_features import extract_meta_features, N_META_FEATURES
from rl_selector.inference import RLModelSelector, CLASSIFICATION_MODELS, REGRESSION_MODELS

# ─── ML model imports ─────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# =============================================================================
# TRAINING DATASET IDs — MUST EXCLUDE THESE FROM BENCHMARK
# =============================================================================

TRAINING_CLASSIFICATION_IDS = {
    31, 37, 44, 50, 54, 151, 182, 188, 1462, 1464, 1480, 1494, 1510,
    1461, 1489, 1590, 4534, 1501, 40496, 40668, 40670, 40701, 40975,
    40982, 40983, 40984, 40994, 41027, 23, 29, 38,
}

TRAINING_REGRESSION_IDS = {
    507, 531, 546, 41021, 41980, 42225, 42570, 287, 42571, 42705,
    41187, 422, 505, 41702,
}

TRAINING_ALL_IDS = TRAINING_CLASSIFICATION_IDS | TRAINING_REGRESSION_IDS


# =============================================================================
# MODEL FACTORIES — fast defaults for benchmark evaluation
# =============================================================================

def get_benchmark_classifiers() -> Dict:
    """Classification models matching PPO's action space — fast CV defaults."""
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced',
        ),
        'GaussianNB': GaussianNB(),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'SVC': SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced'),
        'DecisionTreeClassifier': DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight='balanced',
        ),
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced',
        ),
        'ExtraTreesClassifier': ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced',
        ),
        'GradientBoostingClassifier': GradientBoostingClassifier(
            n_estimators=200, random_state=42,
        ),
    }


def get_benchmark_regressors() -> Dict:
    """Regression models matching PPO's action space — fast CV defaults."""
    return {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=8, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
        'ExtraTreesRegressor': ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators=200, random_state=42,
        ),
    }


# =============================================================================
# OPENML DATASET DISCOVERY
# =============================================================================

def fetch_candidate_datasets(
    task_type: str,
    n_target: int = 500,
    min_samples: int = 100,
    max_samples: int = 100_000,
    min_features: int = 2,
    max_features: int = 500,
    max_classes: int = 10,
) -> List[Dict]:
    """
    Fetch a list of suitable OpenML datasets, excluding training IDs.

    Returns list of dicts with keys: id, name, n_samples, n_features
    """
    try:
        import openml
    except ImportError:
        print("ERROR: openml not installed. Run: pip install openml")
        sys.exit(1)

    print(f"\n[OpenML] Querying {task_type} datasets...")

    exclude_ids = TRAINING_ALL_IDS

    try:
        if task_type == 'classification':
            # Query supervised classification tasks
            tasks = openml.tasks.list_tasks(
                task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
                output_format='dataframe',
            )
            # Filter by reasonable size
            tasks = tasks[
                (tasks['NumberOfInstances'] >= min_samples) &
                (tasks['NumberOfInstances'] <= max_samples) &
                (tasks['NumberOfFeatures'] >= min_features) &
                (tasks['NumberOfFeatures'] <= max_features) &
                (tasks['NumberOfClasses'] >= 2) &
                (tasks['NumberOfClasses'] <= max_classes) &
                (tasks['NumberOfMissingValues'] / (tasks['NumberOfInstances'] * tasks['NumberOfFeatures']) < 0.3)
            ]
            # Get dataset IDs from tasks
            did_col = 'source_data'
            if did_col not in tasks.columns:
                did_col = 'did'
            dataset_ids = tasks[did_col].dropna().astype(int).unique().tolist()

        else:  # regression
            tasks = openml.tasks.list_tasks(
                task_type=openml.tasks.TaskType.SUPERVISED_REGRESSION,
                output_format='dataframe',
            )
            tasks = tasks[
                (tasks['NumberOfInstances'] >= min_samples) &
                (tasks['NumberOfInstances'] <= max_samples) &
                (tasks['NumberOfFeatures'] >= min_features) &
                (tasks['NumberOfFeatures'] <= max_features)
            ]
            did_col = 'source_data'
            if did_col not in tasks.columns:
                did_col = 'did'
            dataset_ids = tasks[did_col].dropna().astype(int).unique().tolist()

    except Exception as e:
        print(f"[OpenML] Task list failed ({e}), falling back to dataset list...")
        try:
            # Fallback: list all datasets and filter
            all_ds = openml.datasets.list_datasets(output_format='dataframe')
            all_ds = all_ds[
                (all_ds['NumberOfInstances'] >= min_samples) &
                (all_ds['NumberOfInstances'] <= max_samples) &
                (all_ds['NumberOfFeatures'] >= min_features) &
                (all_ds['NumberOfFeatures'] <= max_features)
            ]
            dataset_ids = all_ds.index.tolist()
        except Exception as e2:
            print(f"[OpenML] Dataset list also failed ({e2})")
            return []

    # Exclude training IDs
    dataset_ids = [d for d in dataset_ids if d not in exclude_ids]
    random.shuffle(dataset_ids)  # randomise order

    print(f"[OpenML] Found {len(dataset_ids)} candidate {task_type} datasets after exclusions")

    # Return as many as needed
    candidates = []
    for did in dataset_ids[:min(n_target * 2, len(dataset_ids))]:
        candidates.append({'id': did})

    return candidates


# =============================================================================
# DATASET LOADING & PREPROCESSING
# =============================================================================

def load_openml_dataset(dataset_id: int, task_type: str) -> Optional[Tuple]:
    """
    Load and preprocess an OpenML dataset.

    Returns: (X, y, dataset_name) or None on failure.
    """
    try:
        import openml
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_raw, y_raw, _, _ = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute,
        )
        dataset_name = dataset.name

    except Exception as e:
        return None

    if X_raw is None or y_raw is None:
        return None

    # Basic size checks
    if len(X_raw) < 50 or len(X_raw.columns) < 1:
        return None

    # ── Preprocess target ──────────────────────────────────────────────────
    if task_type == 'classification':
        le = LabelEncoder()
        try:
            y = le.fit_transform(y_raw.astype(str))
        except Exception:
            return None

        n_classes = len(np.unique(y))
        if n_classes < 2 or n_classes > 20:
            return None

        # Require minimum samples per class
        min_count = pd.Series(y).value_counts().min()
        if min_count < 5:
            return None

    else:  # regression
        try:
            y = pd.to_numeric(y_raw, errors='coerce').values
        except Exception:
            return None

        if np.isnan(y).mean() > 0.3:
            return None
        y = y[~np.isnan(y)]
        if len(y) < 50:
            return None

    # ── Preprocess features ────────────────────────────────────────────────
    # Encode categoricals
    X = X_raw.copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        try:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        except Exception:
            X[col] = 0

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        return None

    # Handle inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing values
    if X.isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X_arr = imputer.fit_transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns)

    # Align samples if regression removed NaN targets
    if task_type == 'regression':
        y_full = pd.to_numeric(y_raw, errors='coerce').values
        valid_mask = ~np.isnan(y_full)
        X = X[valid_mask].reset_index(drop=True)
        y = y_full[valid_mask]

    if len(X) < 50 or len(X) != len(y):
        return None

    return X, np.array(y), dataset_name


# =============================================================================
# MODEL EVALUATION — ground truth scoring
# =============================================================================

def evaluate_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    cv_folds: int = 3,
    max_samples_for_cv: int = 10_000,
) -> Dict[str, float]:
    """
    Evaluate all candidate models with cross-validation.
    Returns {model_name: cv_score}.
    """
    n = len(X)
    if n > max_samples_for_cv:
        # Subsample for speed
        if task_type == 'classification':
            # Stratified subsample
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples_for_cv, random_state=42)
            try:
                idx, _ = next(sss.split(X, y))
            except Exception:
                idx = np.random.choice(n, max_samples_for_cv, replace=False)
        else:
            idx = np.random.choice(n, max_samples_for_cv, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        n = len(X)

    # Scale features for distance-based models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    if task_type == 'classification':
        models = get_benchmark_classifiers()
        scoring = 'accuracy'
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        models = get_benchmark_regressors()
        scoring = 'r2'
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # For large datasets skip quadratic-time models (SVC/SVR, KNN)
    slow_models = set()
    if n > 5000:
        slow_models = {'SVC', 'SVR', 'KNeighborsClassifier', 'KNeighborsRegressor'}

    scores = {}
    for name, model in models.items():
        if name in slow_models:
            scores[name] = 0.0
            continue
        try:
            # SVC/SVR and KNN benefit from scaling; tree models don't care
            use_X = X_scaled if name in ('SVC', 'SVR', 'KNeighborsClassifier',
                                          'KNeighborsRegressor', 'LogisticRegression',
                                          'Ridge', 'Lasso', 'ElasticNet') else X
            cv_scores = cross_val_score(
                model, use_X, y, cv=cv, scoring=scoring,
                error_score=0.0, n_jobs=1,
            )
            scores[name] = float(np.mean(cv_scores))
        except Exception:
            scores[name] = 0.0

    return scores


# =============================================================================
# BASELINES
# =============================================================================

def random_selection(model_names: List[str], seed: int = None) -> str:
    rng = random.Random(seed)
    return rng.choice(model_names)


def oracle_selection(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    task_type: str = 'classification',
    n_datasets: int = 500,
    cv_folds: int = 3,
    output_dir: str = None,
    resume: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Main benchmark loop.

    For each dataset:
      1. Load from OpenML
      2. Extract 40 meta-features
      3. Get PPO recommendations (top-3)
      4. Evaluate all models with CV → ground truth
      5. Compute metrics vs all baselines

    Returns a DataFrame with per-dataset results.
    """
    random.seed(seed)
    np.random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(_BENCH_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f'ppo_benchmark_{task_type}.jsonl')
    summary_path = os.path.join(output_dir, f'ppo_benchmark_{task_type}_summary.csv')

    # Load PPO selector
    print("\n[Benchmark] Loading PPO model selector...")
    selector = RLModelSelector()
    model_names = CLASSIFICATION_MODELS if task_type == 'classification' else REGRESSION_MODELS
    print(f"[Benchmark] PPO models for {task_type}: {model_names}")
    print(f"[Benchmark] use_rl = {selector.use_rl}")

    # Discover datasets
    candidates = fetch_candidate_datasets(task_type, n_target=n_datasets)
    if not candidates:
        print("[Benchmark] No candidates found. Check your OpenML connection.")
        return pd.DataFrame()

    # Resume: skip already-processed datasets
    processed_ids = set()
    results = []
    if resume and os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    results.append(row)
                    processed_ids.add(row['dataset_id'])
                except Exception:
                    pass
        print(f"[Benchmark] Resuming - {len(results)} datasets already processed")

    # ── Main loop ──────────────────────────────────────────────────────────────
    n_success = 0
    n_skip = 0
    n_errors = 0
    start_time = time.time()

    for i, cand in enumerate(candidates):
        if n_success >= n_datasets:
            break

        did = cand['id']
        if did in processed_ids:
            continue

        elapsed = time.time() - start_time
        print(f"\n[{n_success+1}/{n_datasets}] Dataset {did}  "
              f"(elapsed {elapsed/60:.1f}m, skip={n_skip}, err={n_errors})")

        # ── Load dataset ───────────────────────────────────────────────────
        t0 = time.time()
        result = load_openml_dataset(did, task_type)
        if result is None:
            print(f"  SKIP: Skipped (load failed or bad format)")
            n_skip += 1
            continue

        X, y, ds_name = result
        load_time = time.time() - t0
        print(f"  Dataset: {ds_name!r}  shape={X.shape}  "
              f"classes={len(np.unique(y)) if task_type=='classification' else 'N/A'}  "
              f"(loaded in {load_time:.1f}s)")

        # ── Extract meta-features ──────────────────────────────────────────
        try:
            meta = extract_meta_features(X, y, task_type)
            assert meta.shape == (N_META_FEATURES,), f"Expected {N_META_FEATURES}, got {meta.shape}"
        except Exception as e:
            print(f"  ERROR: Meta-feature extraction failed: {e}")
            n_errors += 1
            continue

        # ── PPO recommendations ────────────────────────────────────────────
        try:
            recs = selector.recommend(meta, task_type, top_k=3)
            ppo_top1 = recs[0][0] if recs else model_names[0]
            ppo_top3 = [r[0] for r in recs[:3]]
            ppo_confidences = {r[0]: r[1] for r in recs}
        except Exception as e:
            print(f"  ERROR: PPO recommendation failed: {e}")
            n_errors += 1
            continue

        # ── Evaluate all models ────────────────────────────────────────────
        t1 = time.time()
        try:
            scores = evaluate_all_models(X, y, task_type, cv_folds=cv_folds)
        except Exception as e:
            print(f"  ERROR: Model evaluation failed: {e}")
            n_errors += 1
            continue

        eval_time = time.time() - t1

        if not scores or all(s == 0.0 for s in scores.values()):
            print(f"  SKIP: All models scored 0.0 (degenerate dataset)")
            n_skip += 1
            continue

        # ── Ground truth ───────────────────────────────────────────────────
        oracle_model = oracle_selection(scores)
        oracle_score = scores[oracle_model]
        ppo_top1_score = scores.get(ppo_top1, 0.0)
        ppo_top3_scores = [scores.get(m, 0.0) for m in ppo_top3]
        ppo_best_top3_score = max(ppo_top3_scores) if ppo_top3_scores else 0.0

        # Rank PPO top-1 among all models (1 = best)
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        model_rank = {m: r + 1 for r, (m, _) in enumerate(sorted_models)}
        ppo_rank = model_rank.get(ppo_top1, len(model_names))

        # Baselines
        rf_name = 'RandomForestClassifier' if task_type == 'classification' else 'RandomForestRegressor'
        gb_name = 'GradientBoostingClassifier' if task_type == 'classification' else 'GradientBoostingRegressor'
        rf_score = scores.get(rf_name, 0.0)
        gb_score = scores.get(gb_name, 0.0)

        # Random baseline — average over 10 random picks
        random_scores = [scores.get(random_selection(model_names, seed=seed + j), 0.0)
                         for j in range(10)]
        random_avg_score = float(np.mean(random_scores))

        # ── Metrics ────────────────────────────────────────────────────────
        top1_correct = int(ppo_top1 == oracle_model)
        top3_correct = int(oracle_model in ppo_top3)
        regret = float(oracle_score - ppo_top1_score)
        regret_top3 = float(oracle_score - ppo_best_top3_score)
        ppo_vs_rf = float(ppo_top1_score - rf_score)
        ppo_vs_gb = float(ppo_top1_score - gb_score)
        ppo_vs_random = float(ppo_top1_score - random_avg_score)

        print(f"  Oracle: {oracle_model} ({oracle_score:.4f})  "
              f"PPO-top1: {ppo_top1} ({ppo_top1_score:.4f})  "
              f"Rank: {ppo_rank}/{len(scores)}  "
              f"Regret: {regret:.4f}  "
              f"Top1: {'OK' if top1_correct else 'NO'}  "
              f"Top3: {'OK' if top3_correct else 'NO'}")
        print(f"  RF: {rf_score:.4f}  GB: {gb_score:.4f}  "
              f"Random: {random_avg_score:.4f}  (eval: {eval_time:.1f}s)")

        # ── Store result ───────────────────────────────────────────────────
        row = {
            'dataset_id': did,
            'dataset_name': ds_name,
            'task_type': task_type,
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(np.unique(y))) if task_type == 'classification' else None,
            # PPO
            'ppo_top1': ppo_top1,
            'ppo_top3': ppo_top3,
            'ppo_top1_confidence': float(ppo_confidences.get(ppo_top1, 0.0)),
            'ppo_top1_score': ppo_top1_score,
            'ppo_best_top3_score': ppo_best_top3_score,
            # Ground truth
            'oracle_model': oracle_model,
            'oracle_score': oracle_score,
            'all_model_scores': scores,
            'model_ranks': {m: r for m, r in model_rank.items()},
            # Metrics
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'ppo_rank': ppo_rank,
            'regret': regret,
            'regret_top3': regret_top3,
            # Baselines
            'rf_score': rf_score,
            'gb_score': gb_score,
            'random_avg_score': random_avg_score,
            'ppo_vs_rf': ppo_vs_rf,
            'ppo_vs_gb': ppo_vs_gb,
            'ppo_vs_random': ppo_vs_random,
            # Timing
            'eval_time_s': round(eval_time, 2),
        }

        results.append(row)
        n_success += 1

        # Progressive save (JSONL — one record per line)
        with open(results_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    # ── Aggregate summary ──────────────────────────────────────────────────────
    if not results:
        print("\n[Benchmark] No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    total = len(df)
    print(f"\n{'='*70}")
    print(f"PPO BENCHMARK RESULTS - {task_type.upper()}  (n={total})")
    print(f"{'='*70}")

    top1_acc = df['top1_correct'].mean() * 100
    top3_acc = df['top3_correct'].mean() * 100
    avg_regret = df['regret'].mean()
    avg_regret_top3 = df['regret_top3'].mean()
    avg_rank = df['ppo_rank'].mean()
    median_rank = df['ppo_rank'].median()
    win_vs_rf = (df['ppo_vs_rf'] >= -0.005).mean() * 100  # within 0.5% of RF
    beat_rf = (df['ppo_vs_rf'] > 0).mean() * 100
    beat_gb = (df['ppo_vs_gb'] > 0).mean() * 100
    beat_random = (df['ppo_vs_random'] > 0).mean() * 100

    n_models = len(model_names)
    random_top1_baseline = 1.0 / n_models * 100  # random chance

    print(f"\n  MODEL SELECTION ACCURACY")
    print(f"    Top-1 accuracy  : {top1_acc:.1f}%  (random baseline: {random_top1_baseline:.1f}%)")
    print(f"    Top-3 accuracy  : {top3_acc:.1f}%  (random baseline: {min(3/n_models*100, 100):.1f}%)")
    print(f"    Lift vs random  : {top1_acc - random_top1_baseline:+.1f}pp")

    print(f"\n  REGRET (lower = better)")
    print(f"    Top-1 avg regret: {avg_regret:.4f}  (0 = oracle quality)")
    print(f"    Top-3 avg regret: {avg_regret_top3:.4f}")
    print(f"    Top-1 med regret: {df['regret'].median():.4f}")

    print(f"\n  AVERAGE RANK (1 = best possible, {n_models} = worst)")
    print(f"    Mean rank       : {avg_rank:.2f} / {n_models}")
    print(f"    Median rank     : {median_rank:.1f} / {n_models}")
    pct_top3_rank = (df['ppo_rank'] <= 3).mean() * 100
    print(f"    In top-3 rank   : {pct_top3_rank:.1f}% of datasets")

    print(f"\n  PPO vs BASELINES")
    print(f"    Beat RandomForest          : {beat_rf:.1f}% of datasets")
    print(f"    Within 0.5% of RF          : {win_vs_rf:.1f}% of datasets")
    print(f"    Beat GradientBoosting      : {beat_gb:.1f}% of datasets")
    print(f"    Beat random selection      : {beat_random:.1f}% of datasets")

    # Per-model oracle frequency (where is the best model?)
    print(f"\n  ORACLE MODEL FREQUENCY (ground truth best model distribution)")
    oracle_counts = df['oracle_model'].value_counts()
    for m, c in oracle_counts.items():
        ppo_chose = (df['ppo_top1'] == m).sum()
        print(f"    {m:<35s} oracle={c:3d} ({c/total*100:4.1f}%)  "
              f"ppo_chose={ppo_chose:3d} ({ppo_chose/total*100:4.1f}%)")

    # Best baselines comparison
    rf_avg = df['rf_score'].mean()
    gb_avg = df['gb_score'].mean()
    ppo_avg = df['ppo_top1_score'].mean()
    random_avg = df['random_avg_score'].mean()
    oracle_avg = df['oracle_score'].mean()

    print(f"\n  AVERAGE SCORES")
    print(f"    Oracle (upper bound)       : {oracle_avg:.4f}")
    print(f"    PPO top-1                  : {ppo_avg:.4f}  ({(ppo_avg/oracle_avg*100):.1f}% of oracle)")
    print(f"    Always-RandomForest        : {rf_avg:.4f}  ({(rf_avg/oracle_avg*100):.1f}% of oracle)")
    print(f"    Always-GradientBoosting    : {gb_avg:.4f}  ({(gb_avg/oracle_avg*100):.1f}% of oracle)")
    print(f"    Random selection           : {random_avg:.4f}  ({(random_avg/oracle_avg*100):.1f}% of oracle)")

    # Summary verdict
    print(f"\n{'='*70}")
    ppo_improvement = ppo_avg - random_avg
    if top1_acc > random_top1_baseline * 1.5:
        verdict = "STRONG - PPO significantly outperforms random model selection"
    elif top1_acc > random_top1_baseline * 1.2:
        verdict = "GOOD - PPO meaningfully outperforms random selection"
    elif top1_acc > random_top1_baseline:
        verdict = "MARGINAL - PPO slightly better than random"
    else:
        verdict = "WEAK - PPO does not outperform random selection (may need retraining)"

    print(f"  VERDICT: {verdict}")
    print(f"  Score lift over random: {ppo_improvement:+.4f}")
    print(f"{'='*70}")

    # Save summary CSV
    df.to_csv(summary_path, index=False)
    print(f"\n[Benchmark] Results saved:")
    print(f"  JSONL: {results_path}")
    print(f"  CSV  : {summary_path}")

    return df


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df: pd.DataFrame, task_type: str, output_dir: str) -> None:
    """Generate a detailed HTML report with charts."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Report] plotly not installed - skipping HTML report")
        return

    if df is None or df.empty:
        return

    model_names = CLASSIFICATION_MODELS if task_type == 'classification' else REGRESSION_MODELS
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Top-1 & Top-3 Accuracy vs Baselines',
            'Regret Distribution',
            'PPO Rank Distribution',
            'Oracle Model Frequency vs PPO Choice',
            'Average Scores: PPO vs Baselines',
            'PPO Score vs Oracle Score (per dataset)',
        ],
    )

    # 1. Accuracy bar
    n_models = len(model_names)
    random_top1 = 1.0 / n_models * 100
    random_top3 = min(3.0 / n_models * 100, 100)
    top1 = df['top1_correct'].mean() * 100
    top3 = df['top3_correct'].mean() * 100
    fig.add_trace(go.Bar(
        x=['Top-1', 'Top-3'],
        y=[top1, top3],
        name='PPO',
        marker_color='#2196F3',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=['Top-1', 'Top-3'],
        y=[random_top1, random_top3],
        name='Random',
        marker_color='#9E9E9E',
    ), row=1, col=1)

    # 2. Regret histogram
    fig.add_trace(go.Histogram(
        x=df['regret'].clip(0, 0.5),
        nbinsx=30,
        name='Regret',
        marker_color='#F44336',
    ), row=1, col=2)

    # 3. Rank distribution
    rank_counts = df['ppo_rank'].value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=rank_counts.index.astype(str),
        y=rank_counts.values,
        name='PPO Rank',
        marker_color='#4CAF50',
    ), row=2, col=1)

    # 4. Oracle vs PPO choice frequency
    oracle_counts = df['oracle_model'].value_counts()
    ppo_counts = df['ppo_top1'].value_counts()
    all_models = list(set(list(oracle_counts.index) + list(ppo_counts.index)))
    fig.add_trace(go.Bar(
        x=all_models,
        y=[oracle_counts.get(m, 0) for m in all_models],
        name='Oracle best',
        marker_color='#FF9800',
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=all_models,
        y=[ppo_counts.get(m, 0) for m in all_models],
        name='PPO choice',
        marker_color='#2196F3',
        opacity=0.7,
    ), row=2, col=2)

    # 5. Average score comparison
    avg_scores = {
        'Oracle': df['oracle_score'].mean(),
        'PPO Top-1': df['ppo_top1_score'].mean(),
        'Always-RF': df['rf_score'].mean(),
        'Always-GB': df['gb_score'].mean(),
        'Random': df['random_avg_score'].mean(),
    }
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0', '#9E9E9E']
    fig.add_trace(go.Bar(
        x=list(avg_scores.keys()),
        y=list(avg_scores.values()),
        name='Avg Score',
        marker_color=colors,
    ), row=3, col=1)

    # 6. PPO vs Oracle scatter
    fig.add_trace(go.Scatter(
        x=df['oracle_score'],
        y=df['ppo_top1_score'],
        mode='markers',
        name='Dataset',
        marker=dict(color='#2196F3', size=6, opacity=0.6),
    ), row=3, col=2)
    # Perfect line
    min_s = min(df['oracle_score'].min(), df['ppo_top1_score'].min())
    max_s = max(df['oracle_score'].max(), df['ppo_top1_score'].max())
    fig.add_trace(go.Scatter(
        x=[min_s, max_s], y=[min_s, max_s],
        mode='lines', name='Oracle=PPO',
        line=dict(color='red', dash='dash'),
    ), row=3, col=2)

    fig.update_layout(
        title=f'PPO Benchmark Results - {task_type.capitalize()} ({len(df)} datasets)',
        height=1000,
        showlegend=True,
    )

    report_path = os.path.join(output_dir, f'ppo_benchmark_{task_type}_report.html')
    fig.write_html(report_path)
    print(f"[Report] HTML report saved: {report_path}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark PPO model selector on unseen OpenML datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--task', choices=['classification', 'regression', 'both'],
        default='classification',
        help='Task type to benchmark (default: classification)',
    )
    parser.add_argument(
        '--n', type=int, default=500,
        help='Number of datasets to evaluate (default: 500)',
    )
    parser.add_argument(
        '--cv', type=int, default=3,
        help='CV folds for model evaluation (default: 3)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results (default: benchmark/results/)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run (skip already-processed datasets)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Generate HTML report after benchmarking',
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join(_BENCH_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PPO MODEL SELECTOR BENCHMARK")
    print(f"{'='*70}")
    print(f"Task     : {args.task}")
    print(f"Datasets : {args.n}")
    print(f"CV folds : {args.cv}")
    print(f"Output   : {output_dir}")
    print(f"Resume   : {args.resume}")
    print(f"Seed     : {args.seed}")
    print(f"{'='*70}")

    tasks = (
        ['classification', 'regression'] if args.task == 'both'
        else [args.task]
    )

    n_per_task = args.n // len(tasks) if args.task == 'both' else args.n

    all_dfs = {}
    for task in tasks:
        print(f"\n\n{'#'*70}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#'*70}")

        df = run_benchmark(
            task_type=task,
            n_datasets=n_per_task,
            cv_folds=args.cv,
            output_dir=output_dir,
            resume=args.resume,
            seed=args.seed,
        )
        all_dfs[task] = df

        if args.report and df is not None and not df.empty:
            generate_report(df, task, output_dir)

    # Combined summary if both tasks
    if args.task == 'both' and all(not df.empty for df in all_dfs.values()):
        print(f"\n{'='*70}")
        print(f"COMBINED SUMMARY (classification + regression)")
        print(f"{'='*70}")
        combined = pd.concat(all_dfs.values(), ignore_index=True)
        print(f"  Total datasets evaluated: {len(combined)}")
        print(f"  Overall Top-1 accuracy  : {combined['top1_correct'].mean()*100:.1f}%")
        print(f"  Overall Top-3 accuracy  : {combined['top3_correct'].mean()*100:.1f}%")
        print(f"  Overall avg regret      : {combined['regret'].mean():.4f}")
        print(f"  Overall avg rank        : {combined['ppo_rank'].mean():.2f}")

    print(f"\n[Done] Benchmark complete.")


if __name__ == '__main__':
    main()
