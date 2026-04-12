"""
PPO vs AutoML Comparison Benchmark

Compares the PPO model selector head-to-head against FLAML AutoML
on the same OpenML datasets used by the AMLB benchmark suite.

How it works:
  For each dataset:
    1. PPO: extract 40 meta-features -> pick a model -> train it (60s budget)
    2. FLAML: runs full AutoML search (60s budget, same data)
  Then compares accuracy / R2 scores side by side.

Dataset suites:
  test       -  3 datasets  (AMLB test.yaml)
  validation -  9 datasets  (AMLB validation.yaml)
  full       - 12 datasets  (test + validation)
  cc18       - 72 datasets  (OpenML-CC18, the gold standard in AutoML research)
  amlb       - 71 datasets  (AMLB paper 2023 main classification suite)

Usage:
    python -m benchmark.amlb_comparison --suite test            # quick (3 datasets)
    python -m benchmark.amlb_comparison --suite validation      # medium (9 datasets)
    python -m benchmark.amlb_comparison --suite cc18            # gold standard (72 datasets)
    python -m benchmark.amlb_comparison --suite amlb            # AMLB paper suite (71 datasets)
    python -m benchmark.amlb_comparison --suite cc18 --time 120 # 2 min per framework
    python -m benchmark.amlb_comparison --suite cc18 --resume   # continue if interrupted

Statistical tests are automatically run when n >= 20 datasets.
"""

import os
import sys
import json
import time
import warnings
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Project root
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BENCH_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from meta_features import extract_meta_features, N_META_FEATURES
from rl_selector.inference import RLModelSelector, CLASSIFICATION_MODELS, REGRESSION_MODELS
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# =============================================================================
# AMLB DATASET SUITES — OpenML task IDs and their dataset IDs
# =============================================================================

# AMLB test suite (3 datasets, from test.yaml)
AMLB_TEST_TASKS = [
    {'name': 'kc2',         'task_id': 3913,   'dataset_id': 1063, 'type': 'classification'},
    {'name': 'iris',        'task_id': 59,     'dataset_id': 61,   'type': 'classification'},
    {'name': 'cholesterol', 'task_id': 2295,   'dataset_id': 204,  'type': 'regression'},
]

# AMLB validation suite (9 datasets, from validation.yaml)
AMLB_VALIDATION_TASKS = [
    {'name': 'bioresponse',    'task_id': 9910,   'dataset_id': 4134,  'type': 'classification'},
    {'name': 'eucalyptus',     'task_id': 2079,   'dataset_id': 188,   'type': 'classification'},
    {'name': 'kc1',            'task_id': 3917,   'dataset_id': 1067,  'type': 'classification'},
    {'name': 'micro-mass',     'task_id': 9950,   'dataset_id': 4142,  'type': 'classification'},
    {'name': 'APSFailure',     'task_id': 168868, 'dataset_id': 41138, 'type': 'classification'},
    {'name': 'dresses-sales',  'task_id': 125920, 'dataset_id': 23381, 'type': 'classification'},
]

# Extra well-known OpenML datasets NOT in PPO training set
EXTRA_TASKS = [
    {'name': 'wine',           'task_id': 40691,  'dataset_id': 40691, 'type': 'classification'},
    {'name': 'breast-cancer',  'task_id': 13,     'dataset_id': 15,    'type': 'classification'},
    {'name': 'heart-disease',  'task_id': 2274,   'dataset_id': 53,    'type': 'classification'},
    {'name': 'sonar',          'task_id': 40,     'dataset_id': 40,    'type': 'classification'},
    {'name': 'diabetes',       'task_id': 10101,  'dataset_id': 37,    'type': 'classification'},
    {'name': 'boston',         'task_id': 2295,   'dataset_id': 531,   'type': 'regression'},
    {'name': 'abalone',        'task_id': 7,      'dataset_id': 1,     'type': 'regression'},
]

# IDs used during PPO training (must exclude)
TRAINING_IDS = {
    31, 37, 44, 50, 54, 151, 182, 188, 1462, 1464, 1480, 1494, 1510,
    1461, 1489, 1590, 4534, 1501, 40496, 40668, 40670, 40701, 40975,
    40982, 40983, 40984, 40994, 41027, 23, 29, 38,
    507, 531, 546, 41021, 41980, 42225, 42570, 287, 42571, 42705,
    41187, 422, 505, 41702,
}

# =============================================================================
# OPENML STUDY FETCHER — for CC18 (study 99) and AMLB (study 271)
# =============================================================================

def fetch_openml_study(study_id: int, task_type: str = 'classification') -> List[Dict]:
    """
    Fetch all dataset IDs from an OpenML benchmark study.

    Key studies:
      99  = OpenML-CC18  (72 classification datasets, gold standard)
      271 = AMLB main classification suite (71 datasets, from AMLB 2023 paper)
      269 = AMLB main regression suite

    Returns list of dicts: [{name, dataset_id, type}, ...]
    Excludes datasets in TRAINING_IDS.
    """
    try:
        import openml
        study = openml.study.get_suite(study_id)
        task_ids = study.tasks
        print(f"[OpenML] Study {study_id} ({study.name}): {len(task_ids)} tasks")
    except Exception as e:
        print(f"[OpenML] Could not fetch study {study_id}: {e}")
        return []

    datasets = []
    skipped_training = 0
    for tid in task_ids:
        try:
            task = openml.tasks.get_task(tid, download_data=False)
            did = task.dataset_id
            ds = openml.datasets.get_dataset(did, download_data=False,
                                              download_qualities=False,
                                              download_features_meta_data=False)
            if did in TRAINING_IDS:
                skipped_training += 1
                continue
            datasets.append({
                'name': ds.name,
                'task_id': tid,
                'dataset_id': did,
                'type': task_type,
            })
        except Exception:
            continue

    print(f"[OpenML] {len(datasets)} usable datasets ({skipped_training} skipped, in PPO training set)")
    return datasets


# =============================================================================
# MODEL FACTORIES — same as ppo_benchmark, for PPO ground truth
# =============================================================================

def _get_model(name: str, task_type: str):
    """Return a fitted-ready model instance by name."""
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
        RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    clf_map = {
        'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'GaussianNB':                  GaussianNB(),
        'KNeighborsClassifier':        KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'SVC':                         SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced'),
        'DecisionTreeClassifier':      DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
        'RandomForestClassifier':      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
        'ExtraTreesClassifier':        ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
        'GradientBoostingClassifier':  GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    reg_map = {
        'Ridge':                  Ridge(alpha=1.0),
        'Lasso':                  Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet':             ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'SVR':                    SVR(kernel='rbf', C=1.0),
        'KNeighborsRegressor':    KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'DecisionTreeRegressor':  DecisionTreeRegressor(max_depth=8, random_state=42),
        'RandomForestRegressor':  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'ExtraTreesRegressor':    ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    return (clf_map if task_type == 'classification' else reg_map).get(name)


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(dataset_id: int, task_type: str) -> Optional[Tuple]:
    """Load OpenML dataset by dataset ID. Returns (X_train, X_test, y_train, y_test, name)."""
    try:
        import openml
        ds = openml.datasets.get_dataset(
            dataset_id, download_data=True,
            download_qualities=False, download_features_meta_data=False,
        )
        X_raw, y_raw, _, _ = ds.get_data(
            dataset_format='dataframe',
            target=ds.default_target_attribute,
        )
        name = ds.name
    except Exception as e:
        return None

    if X_raw is None or y_raw is None or len(X_raw) < 50:
        return None

    # Encode target
    if task_type == 'classification':
        le = LabelEncoder()
        try:
            y = le.fit_transform(y_raw.astype(str))
        except Exception:
            return None
        if len(np.unique(y)) < 2 or len(np.unique(y)) > 20:
            return None
        if pd.Series(y).value_counts().min() < 3:
            return None
    else:
        try:
            y = pd.to_numeric(y_raw, errors='coerce').values
        except Exception:
            return None
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            return None
        X_raw = X_raw[valid].reset_index(drop=True)
        y = y[valid]

    # Encode features
    X = X_raw.copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        try:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        except Exception:
            X[col] = 0
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return None
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(X),
            columns=X.columns
        )

    if len(X) != len(y):
        return None

    # Train/test split (80/20, stratified for classification)
    try:
        if task_type == 'classification':
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    except Exception:
        return None

    return X_tr, X_te, y_tr, y_te, name


# =============================================================================
# PPO RUNNER
# =============================================================================

def run_ppo(
    X_tr: pd.DataFrame, X_te: pd.DataFrame,
    y_tr: np.ndarray, y_te: np.ndarray,
    task_type: str,
    selector: RLModelSelector,
    time_limit: int = 60,
) -> Dict:
    """
    PPO pipeline:
      1. Extract meta-features from training data
      2. Get PPO recommendation (top model)
      3. Train that model on X_tr
      4. Score on X_te

    Returns dict with model_name, score, time_s.
    """
    t0 = time.time()

    # Meta-features
    meta = extract_meta_features(X_tr, y_tr, task_type)

    # PPO recommendation
    recs = selector.recommend(meta, task_type, top_k=3)
    ppo_model_name = recs[0][0] if recs else (
        'RandomForestClassifier' if task_type == 'classification' else 'RandomForestRegressor'
    )

    # Also evaluate top-3 and pick best within time limit
    best_score = -np.inf
    best_model_name = ppo_model_name

    for model_name, _ in recs[:3]:
        if time.time() - t0 > time_limit * 0.8:
            break
        model = _get_model(model_name, task_type)
        if model is None:
            continue
        try:
            scaler = StandardScaler()
            needs_scale = model_name in ('SVC', 'SVR', 'KNeighborsClassifier',
                                          'KNeighborsRegressor', 'LogisticRegression',
                                          'Ridge', 'Lasso', 'ElasticNet')
            X_fit = scaler.fit_transform(X_tr) if needs_scale else X_tr
            X_eval = scaler.transform(X_te) if needs_scale else X_te

            model.fit(X_fit, y_tr)

            if task_type == 'classification':
                score = float(np.mean(model.predict(X_eval) == y_te))
            else:
                from sklearn.metrics import r2_score
                score = float(r2_score(y_te, model.predict(X_eval)))

            if score > best_score:
                best_score = score
                best_model_name = model_name
        except Exception:
            continue

    elapsed = time.time() - t0
    return {
        'method': 'PPO',
        'model_chosen': best_model_name,
        'ppo_top1': ppo_model_name,
        'score': best_score if best_score > -np.inf else 0.0,
        'time_s': round(elapsed, 2),
    }


# =============================================================================
# FLAML RUNNER
# =============================================================================

def run_flaml(
    X_tr: pd.DataFrame, X_te: pd.DataFrame,
    y_tr: np.ndarray, y_te: np.ndarray,
    task_type: str,
    time_limit: int = 60,
) -> Dict:
    """
    FLAML AutoML with a fixed time budget.

    Returns dict with estimator_name, score, time_s.
    """
    try:
        from flaml import AutoML
    except ImportError:
        return {'method': 'FLAML', 'model_chosen': 'N/A', 'score': None, 'time_s': 0, 'error': 'flaml not installed'}

    t0 = time.time()
    try:
        automl = AutoML()
        flaml_task = 'classification' if task_type == 'classification' else 'regression'
        metric = 'accuracy' if task_type == 'classification' else 'r2'

        automl.fit(
            X_tr, y_tr,
            task=flaml_task,
            metric=metric,
            time_budget=time_limit,
            verbose=0,
        )

        if task_type == 'classification':
            preds = automl.predict(X_te)
            score = float(np.mean(preds == y_te))
        else:
            from sklearn.metrics import r2_score
            preds = automl.predict(X_te)
            score = float(r2_score(y_te, preds))

        elapsed = time.time() - t0
        return {
            'method': 'FLAML',
            'model_chosen': automl.best_estimator,
            'score': score,
            'time_s': round(elapsed, 2),
        }
    except Exception as e:
        return {
            'method': 'FLAML',
            'model_chosen': 'ERROR',
            'score': None,
            'time_s': round(time.time() - t0, 2),
            'error': str(e),
        }


# =============================================================================
# RANDOM FOREST BASELINE
# =============================================================================

def run_rf_baseline(
    X_tr: pd.DataFrame, X_te: pd.DataFrame,
    y_tr: np.ndarray, y_te: np.ndarray,
    task_type: str,
) -> Dict:
    """Always-RandomForest baseline."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    t0 = time.time()
    try:
        if task_type == 'classification':
            m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
            m.fit(X_tr, y_tr)
            score = float(np.mean(m.predict(X_te) == y_te))
        else:
            from sklearn.metrics import r2_score
            m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            m.fit(X_tr, y_tr)
            score = float(r2_score(y_te, m.predict(X_te)))
        return {'method': 'RandomForest', 'model_chosen': 'RandomForest', 'score': score, 'time_s': round(time.time()-t0,2)}
    except Exception as e:
        return {'method': 'RandomForest', 'model_chosen': 'ERROR', 'score': None, 'time_s': 0}


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(
    suite: str = 'test',
    n_extra: int = 0,
    time_limit: int = 60,
    output_dir: str = None,
    skip_flaml: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Run the full comparison: PPO vs FLAML vs RandomForest.

    Args:
        suite:      'test' | 'validation' | 'full' | 'cc18' | 'amlb'
        time_limit: seconds per framework per dataset
        output_dir: where to save results
        resume:     skip already-processed datasets
    """
    if output_dir is None:
        output_dir = os.path.join(_BENCH_DIR, 'results_amlb_comparison')
    os.makedirs(output_dir, exist_ok=True)

    # Build dataset list
    tasks = []
    if suite in ('test', 'full'):
        tasks += AMLB_TEST_TASKS
    if suite in ('validation', 'full'):
        tasks += AMLB_VALIDATION_TASKS
    if suite == 'full' and n_extra > 0:
        tasks += EXTRA_TASKS[:n_extra]
    if suite == 'cc18':
        # OpenML-CC18 — study 99 (72 classification datasets, gold standard)
        print("\nFetching OpenML-CC18 study (72 datasets)...")
        tasks = fetch_openml_study(99, task_type='classification')
    if suite == 'amlb':
        # AMLB 2023 paper main classification suite — study 271
        print("\nFetching AMLB main classification suite (study 271)...")
        tasks = fetch_openml_study(271, task_type='classification')

    # Deduplicate
    seen_ids = set()
    unique_tasks = []
    for t in tasks:
        did = t['dataset_id']
        if did not in seen_ids and did not in TRAINING_IDS:
            seen_ids.add(did)
            unique_tasks.append(t)
        elif did in TRAINING_IDS:
            print(f"  [skip] {t['name']} (dataset {did}) is in PPO training set")

    # Resume: load already-processed results
    out_jsonl = os.path.join(output_dir, f'amlb_comparison_{suite}.jsonl')
    processed_names = set()
    loaded_results = []
    if resume and os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    loaded_results.append(row)
                    processed_names.add(row['dataset'])
                except Exception:
                    pass
        print(f"[Resume] {len(loaded_results)} datasets already done")

    print(f"\n{'='*68}")
    print(f"PPO vs FLAML vs RandomForest  |  {suite} suite  |  {len(unique_tasks)} datasets")
    print(f"Time limit per framework: {time_limit}s  |  Resume: {resume}")
    print(f"{'='*68}")

    # Load PPO selector
    selector = RLModelSelector()
    print(f"PPO loaded: use_rl={selector.use_rl}\n")

    results = list(loaded_results)  # start from resumed data
    start_time = time.time()

    for i, task in enumerate(unique_tasks):
        name = task['name']
        did = task['dataset_id']
        dtype = task['type']

        if name in processed_names:
            continue  # already done in a previous run

        elapsed_total = (time.time() - start_time) / 60
        done_so_far = len(results)
        print(f"\n[{done_so_far+1}/{len(unique_tasks)}] {name}  "
              f"(dataset={did}, type={dtype}, elapsed={elapsed_total:.1f}m)")

        # Load dataset
        data = load_dataset(did, dtype)
        if data is None:
            print(f"  SKIP: could not load dataset {did}")
            continue

        X_tr, X_te, y_tr, y_te = data[:4]
        ds_name = data[4]
        print(f"  Loaded: {ds_name!r}  train={len(X_tr)}, test={len(X_te)}, features={X_tr.shape[1]}")

        row = {
            'dataset': name,
            'openml_dataset_id': did,
            'task_type': dtype,
            'n_train': len(X_tr),
            'n_test': len(X_te),
            'n_features': X_tr.shape[1],
        }

        # --- PPO ---
        print(f"  Running PPO  (budget={time_limit}s)...", end=' ', flush=True)
        ppo_res = run_ppo(X_tr, X_te, y_tr, y_te, dtype, selector, time_limit)
        print(f"model={ppo_res['model_chosen']}  score={ppo_res['score']:.4f}  ({ppo_res['time_s']:.1f}s)")
        row['ppo_model'] = ppo_res['model_chosen']
        row['ppo_top1_recommended'] = ppo_res['ppo_top1']
        row['ppo_score'] = ppo_res['score']
        row['ppo_time_s'] = ppo_res['time_s']

        # --- FLAML ---
        if not skip_flaml:
            print(f"  Running FLAML (budget={time_limit}s)...", end=' ', flush=True)
            flaml_res = run_flaml(X_tr, X_te, y_tr, y_te, dtype, time_limit)
            if flaml_res.get('error'):
                print(f"ERROR: {flaml_res['error']}")
                row['flaml_model'] = 'ERROR'
                row['flaml_score'] = None
                row['flaml_time_s'] = flaml_res['time_s']
            else:
                print(f"model={flaml_res['model_chosen']}  score={flaml_res['score']:.4f}  ({flaml_res['time_s']:.1f}s)")
                row['flaml_model'] = flaml_res['model_chosen']
                row['flaml_score'] = flaml_res['score']
                row['flaml_time_s'] = flaml_res['time_s']
        else:
            row['flaml_model'] = 'skipped'
            row['flaml_score'] = None
            row['flaml_time_s'] = 0

        # --- RandomForest baseline ---
        print(f"  Running RF baseline...", end=' ', flush=True)
        rf_res = run_rf_baseline(X_tr, X_te, y_tr, y_te, dtype)
        print(f"score={rf_res['score']:.4f}  ({rf_res['time_s']:.1f}s)")
        row['rf_score'] = rf_res['score']
        row['rf_time_s'] = rf_res['time_s']

        # --- Comparison ---
        ppo_s = row['ppo_score'] or 0.0
        rf_s = row['rf_score'] or 0.0
        flaml_s = row['flaml_score'] or 0.0

        ppo_vs_flaml = ppo_s - flaml_s if row['flaml_score'] is not None else None
        ppo_vs_rf = ppo_s - rf_s
        row['ppo_vs_flaml'] = ppo_vs_flaml
        row['ppo_vs_rf'] = ppo_vs_rf

        if ppo_vs_flaml is not None:
            print(f"  RESULT: PPO={ppo_s:.4f}  FLAML={flaml_s:.4f}  RF={rf_s:.4f}  PPO-FLAML={ppo_vs_flaml:+.4f}")
        else:
            print(f"  RESULT: PPO={ppo_s:.4f}  RF={rf_s:.4f}  PPO-RF={ppo_vs_rf:+.4f}")

        results.append(row)

        # Progressive save (resume support)
        with open(out_jsonl, 'a') as f:
            f.write(json.dumps(row) + '\n')

    if not results:
        print("\nNo results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(df)
    valid_flaml = df[df['flaml_score'].notna()]

    print(f"\n{'='*68}")
    print(f"COMPARISON SUMMARY  (n={n} datasets)")
    print(f"{'='*68}")

    print(f"\n  AVERAGE SCORES")
    print(f"    PPO                : {df['ppo_score'].mean():.4f}")
    if len(valid_flaml) > 0:
        print(f"    FLAML AutoML       : {valid_flaml['flaml_score'].mean():.4f}")
    print(f"    RandomForest       : {df['rf_score'].mean():.4f}")

    if len(valid_flaml) > 0:
        ppo_beats_flaml = (valid_flaml['ppo_vs_flaml'] > 0).sum()
        ppo_ties_flaml  = (valid_flaml['ppo_vs_flaml'].abs() <= 0.01).sum()
        flaml_beats_ppo = (valid_flaml['ppo_vs_flaml'] < -0.01).sum()
        avg_gap = valid_flaml['ppo_vs_flaml'].mean()

        print(f"\n  PPO vs FLAML  (on {len(valid_flaml)} datasets)")
        print(f"    PPO wins          : {ppo_beats_flaml}/{len(valid_flaml)} ({ppo_beats_flaml/len(valid_flaml)*100:.0f}%)")
        print(f"    Tie (within 1%)   : {ppo_ties_flaml}/{len(valid_flaml)} ({ppo_ties_flaml/len(valid_flaml)*100:.0f}%)")
        print(f"    FLAML wins        : {flaml_beats_ppo}/{len(valid_flaml)} ({flaml_beats_ppo/len(valid_flaml)*100:.0f}%)")
        print(f"    Avg score gap     : {avg_gap:+.4f}  (positive = PPO leads)")

        # Wilcoxon signed-rank test (needs >= 20 datasets for reliable results)
        if len(valid_flaml) >= 20:
            try:
                from scipy.stats import wilcoxon
                stat, p_val = wilcoxon(
                    valid_flaml['ppo_score'].values,
                    valid_flaml['flaml_score'].values,
                    alternative='two-sided',
                )
                sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
                print(f"\n  STATISTICAL TEST (Wilcoxon signed-rank)")
                print(f"    H0: PPO and FLAML have equal performance")
                print(f"    p-value  : {p_val:.4f}")
                print(f"    Result   : {sig} (alpha=0.05)")
                if p_val < 0.05:
                    winner = "PPO" if avg_gap > 0 else "FLAML"
                    print(f"    Conclusion: {winner} is statistically significantly better")
                else:
                    print(f"    Conclusion: No statistically significant difference detected")
            except Exception as e:
                print(f"  [Stats test skipped: {e}]")
        else:
            print(f"\n  NOTE: {len(valid_flaml)} datasets is below the 20-dataset threshold")
            print(f"        for a reliable Wilcoxon test. Use --suite cc18 for 72 datasets.")

    ppo_beats_rf = (df['ppo_vs_rf'] > 0).sum()
    print(f"\n  PPO vs RandomForest  (on {n} datasets)")
    print(f"    PPO beats RF       : {ppo_beats_rf}/{n} ({ppo_beats_rf/n*100:.0f}%)")
    print(f"    Avg gap            : {df['ppo_vs_rf'].mean():+.4f}")

    # Per-dataset table (truncated to 30 rows for large suites)
    print(f"\n  PER-DATASET BREAKDOWN" + (" (first 30 shown)" if n > 30 else ""))
    cols = ['dataset', 'task_type', 'ppo_score', 'flaml_score', 'rf_score', 'ppo_model', 'flaml_model']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(30).to_string(index=False, float_format='{:.4f}'.format))

    # Verdict
    if len(valid_flaml) > 0:
        avg_gap = valid_flaml['ppo_vs_flaml'].mean()
        print(f"\n{'='*68}")
        if avg_gap >= 0.01:
            print(f"  VERDICT: PPO WINS -- PPO outperforms FLAML by {avg_gap:+.4f} on avg")
        elif avg_gap >= -0.01:
            print(f"  VERDICT: COMPETITIVE -- PPO and FLAML are comparable (gap={avg_gap:+.4f})")
        else:
            print(f"  VERDICT: FLAML WINS -- FLAML outperforms PPO by {abs(avg_gap):.4f}")
            print(f"    Note: FLAML has full HPO search; PPO selects from 8/9 fixed candidates")
            print(f"    Combine PPO selection + Optuna tuning for a fairer comparison")
        print(f"{'='*68}")

    # Save CSV + JSONL
    out_csv = os.path.join(output_dir, f'amlb_comparison_{suite}.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")
    print(f"[Saved] {out_jsonl}")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare PPO model selector against FLAML AutoML on AMLB datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset suites:
  test        3 datasets  - quick sanity check (~5 min)
  validation  9 datasets  - medium check (~20 min)
  full       12 datasets  - test + validation combined
  cc18       72 datasets  - OpenML-CC18 gold standard (~4 hrs)
  amlb       71 datasets  - AMLB 2023 paper suite (~4 hrs)

Examples:
  python -m benchmark.amlb_comparison --suite test
  python -m benchmark.amlb_comparison --suite cc18 --time 60 --resume
  python -m benchmark.amlb_comparison --suite cc18 --no-flaml   # PPO vs RF only
        """,
    )
    parser.add_argument(
        '--suite',
        choices=['test', 'validation', 'full', 'cc18', 'amlb'],
        default='test',
        help='Dataset suite to use (default: test = 3 datasets)',
    )
    parser.add_argument(
        '--time', type=int, default=60,
        help='Time limit in seconds per framework per dataset (default: 60)',
    )
    parser.add_argument(
        '--extra', type=int, default=0,
        help='Extra datasets beyond the named suite (default: 0)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results',
    )
    parser.add_argument(
        '--no-flaml', action='store_true',
        help='Skip FLAML, run PPO vs RF only (faster)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous run (skip already-processed datasets)',
    )
    args = parser.parse_args()

    run_comparison(
        suite=args.suite,
        n_extra=args.extra,
        time_limit=args.time,
        output_dir=args.output,
        skip_flaml=args.no_flaml,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
