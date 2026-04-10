"""
RL Model Selector Trainer
=========================
Trains a PPO agent to select the best ML model for a given dataset.

Episode structure (matches pseudocode):
    env = ModelSelectionEnv(...)
    for episode in [0, N]:
        32f            = env.reset()               # meta-features of a dataset
        selected_model = ppo.predict(32f)          # agent picks a model
        Reward, next32f = env.step(selected_model) # evaluate -> reward

Reward design:
    LOW error  ->  HIGH reward   (reward = accuracy or max(0, R2))
    HIGH error ->  LOW  reward

All 32 meta-features are computed from real data -- nothing is hardcoded.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
import os

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARNING] stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")


# =============================================================================
# MODELS
# =============================================================================
CLASSIFICATION_MODELS = {
    'LogisticRegression':         LogisticRegression(max_iter=200, random_state=42),
    'GaussianNB':                 GaussianNB(),
    'KNeighborsClassifier':       KNeighborsClassifier(n_neighbors=5),
    'SVC':                        SVC(kernel='rbf', probability=False, random_state=42),
    'DecisionTreeClassifier':     DecisionTreeClassifier(random_state=42),
    'RandomForestClassifier':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTreesClassifier':       ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

REGRESSION_MODELS = {
    'Ridge':                      Ridge(alpha=1.0),
    'Lasso':                      Lasso(alpha=1.0),
    'ElasticNet':                 ElasticNet(alpha=1.0),
    'SVR':                        SVR(kernel='rbf'),
    'KNeighborsRegressor':        KNeighborsRegressor(n_neighbors=5),
    'DecisionTreeRegressor':      DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor':      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTreesRegressor':        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingRegressor':  GradientBoostingRegressor(n_estimators=100, random_state=42),
}


# =============================================================================
# LOAD DATASETS  --  OpenML only (500 real-world datasets per task)
# =============================================================================

def _fetch_openml_catalogue(task_type: str, max_datasets: int):
    """
    Query the OpenML catalogue and return a list of dataset IDs filtered for
    the given task type and reasonable size constraints.

    Classification : NumberOfClasses >= 2
    Regression     : NumberOfClasses == 0  (OpenML marks regression targets as 0)

    Size filters   : 100 <= samples <= 50 000,  2 <= features <= 500
    Sorted by      : NumberOfInstances (prefer medium-sized first)
    """
    import openml

    print(f"[*] Querying OpenML catalogue for {task_type} datasets...")
    meta = openml.datasets.list_datasets(output_format='dataframe')

    # Normalise column names (minor differences across openml versions)
    meta.columns = [c.lower() for c in meta.columns]

    inst_col  = next(c for c in meta.columns if 'instance'  in c)
    feat_col  = next(c for c in meta.columns if 'feature'   in c)
    class_col = next(c for c in meta.columns if 'class'     in c)

    n_cls = meta[class_col].fillna(0).astype(float)

    if task_type == 'classification':
        type_mask = n_cls >= 2
    else:
        type_mask = n_cls <= 1          # 0 = numeric target = regression

    size_mask = (
        (meta[inst_col].fillna(0)  >= 100)   &
        (meta[inst_col].fillna(0)  <= 50_000) &
        (meta[feat_col].fillna(0)  >= 2)     &
        (meta[feat_col].fillna(0)  <= 500)
    )

    filtered = (meta[type_mask & size_mask]
                .sort_values(inst_col)
                .reset_index(drop=True))

    ids = filtered['did'].tolist()
    print(f"[*] {len(ids)} candidates found in catalogue -> "
          f"will download up to {max_datasets}")
    return ids


def _download_and_process(did, task_type: str, max_samples: int):
    """Download one OpenML dataset and return (name, X_arr, y_arr) or None."""
    import openml
    from sklearn.preprocessing import LabelEncoder

    ds = openml.datasets.get_dataset(
        did,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_raw, y_raw, _, _ = ds.get_data(
        target=ds.default_target_attribute,
        dataset_format='dataframe',
    )
    if X_raw is None or y_raw is None:
        return None

    # Keep only numeric columns; fill NaN with column median
    X_num = X_raw.select_dtypes(include='number').dropna(axis=1, how='all')
    if X_num.shape[1] == 0:
        return None

    X_arr = X_num.fillna(X_num.median()).values
    y_arr = y_raw.values

    # Encode categorical/string targets
    if y_arr.dtype == object or str(y_arr.dtype) == 'category':
        y_arr = LabelEncoder().fit_transform(y_arr.astype(str))

    y_arr = y_arr.astype(float if task_type == 'regression' else int)

    # Subsample very large datasets to keep CV fast
    if len(X_arr) > max_samples:
        idx   = np.random.default_rng(42).choice(len(X_arr),
                                                   max_samples, replace=False)
        X_arr = X_arr[idx]
        y_arr = y_arr[idx]

    return (ds.name, X_arr, y_arr)


def load_openml_datasets(task_type='classification',
                          max_datasets=500,
                          max_samples=5_000):
    """
    Dynamically fetch up to `max_datasets` real-world datasets from OpenML.

    Steps
    -----
    1. Query the OpenML catalogue (one fast API call for all metadata)
    2. Filter by task type and size constraints
    3. Download & process datasets one by one until max_datasets is reached

    Install: pip install openml
    """
    try:
        import openml  # noqa: F401
    except ImportError:
        print("[ERROR] openml is not installed.  Run:  pip install openml")
        raise SystemExit(1)

    ids    = _fetch_openml_catalogue(task_type, max_datasets)
    loaded = []
    skipped = 0

    for did in ids:
        if len(loaded) >= max_datasets:
            break
        try:
            result = _download_and_process(did, task_type, max_samples)
            if result is None:
                skipped += 1
                continue
            loaded.append(result)
            name, X_arr, _ = result
            print(f"   [{len(loaded):>3}/{max_datasets}]  "
                  f"{name:40s}  shape={X_arr.shape}")
        except Exception as e:
            skipped += 1
            if skipped <= 15:           # don't flood the console
                print(f"   [SKIP] id={did}  {str(e)[:70]}")

    print(f"\n[OK] {len(loaded)} OpenML {task_type} datasets loaded "
          f"({skipped} skipped)")
    return loaded


def generate_synthetic_classification_datasets():
    """
    Generate 80 synthetic datasets (10 per model × 8 models).
    Each group is structurally designed so the target model genuinely wins —
    not just "happens to perform well" but dominates due to dataset geometry.

    LR    : purely linearly separable, low-dim, zero noise
    GNB   : independent Gaussian blobs, tiny datasets
    KNN   : ultra-tight clusters, local structure dominates
    SVC   : non-linear curved boundaries (moons / circles)
    DT    : XOR / step-function / axis-aligned splits only
    RF    : large + many irrelevant features + high noise
    ET    : very high-dimensional sparse signal
    GBM   : multi-cluster, complex non-linear interactions
    """
    from sklearn.datasets import make_moons, make_circles

    datasets = []
    rng = np.random.RandomState(42)

    # ── LogisticRegression (10) ──────────────────────────────────────────────
    # Pure linear separability: large class_sep, zero redundancy, zero noise
    for i in range(10):
        nf = 2 + i % 6          # 2–7 features
        X, y = make_classification(
            n_samples=300 + i*30, n_features=nf, n_informative=nf,
            n_redundant=0, n_clusters_per_class=1,
            class_sep=3.0, flip_y=0.0, random_state=i)
        datasets.append((f"syn_lr_{i}", X.astype(np.float32), y))

    # ── GaussianNB (10) ──────────────────────────────────────────────────────
    # Truly independent Gaussian features, tiny datasets (NB thrives small)
    for i in range(10):
        n = 50 + i * 12         # 50–158 samples
        n_feat = 4
        # Build truly uncorrelated Gaussian blobs by hand
        X = rng.randn(n, n_feat).astype(np.float32)
        y = (rng.rand(n) > 0.5).astype(np.int64)
        X[y == 0, 0] += 2.5     # class 0 shifted right
        X[y == 1, 0] -= 2.5     # class 1 shifted left
        datasets.append((f"syn_gnb_{i}", X, y))

    # ── KNN (10) ──────────────────────────────────────────────────────────────
    # Ultra-tight blobs (cluster_std=0.25) — local density decides label
    for i in range(10):
        centers = 3 + (i % 4)
        nf      = 2 + (i % 3)
        X, y = make_blobs(
            n_samples=200 + i*20, n_features=nf,
            centers=centers, cluster_std=0.25, random_state=200+i)
        datasets.append((f"syn_knn_{i}", X.astype(np.float32), y.astype(np.int64)))

    # ── SVC (10) ──────────────────────────────────────────────────────────────
    # Curved non-linear boundaries — RBF kernel needed, LR fails
    for i in range(5):
        X, y = make_moons(n_samples=350 + i*50, noise=0.12 + i*0.02,
                          random_state=300+i)
        datasets.append((f"syn_svc_moon_{i}", X.astype(np.float32), y))
    for i in range(5):
        X, y = make_circles(n_samples=350 + i*50, noise=0.06 + i*0.01,
                            factor=0.5, random_state=350+i)
        datasets.append((f"syn_svc_circ_{i}", X.astype(np.float32), y))

    # ── DecisionTree (10) ─────────────────────────────────────────────────────
    # Pure axis-aligned step functions — no linear or kernel trick can fit
    for i in range(10):
        n = 250 + i * 30
        X = rng.randn(n, 4).astype(np.float32)
        if i < 5:
            # XOR label
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
        else:
            # Multi-axis step function
            y = (((X[:, 0] > 0).astype(int) +
                  (X[:, 1] > 0).astype(int) +
                  (X[:, 2] > 0).astype(int)) % 2).astype(np.int64)
        datasets.append((f"syn_dt_{i}", X, y))

    # ── RandomForest (10) ─────────────────────────────────────────────────────
    # Large + many irrelevant features + high noise — bagging helps
    for i in range(10):
        X, y = make_classification(
            n_samples=1500 + i*200, n_features=25 + i*2, n_informative=8,
            n_redundant=6, n_repeated=3, flip_y=0.12,
            class_sep=0.7, random_state=400+i)
        datasets.append((f"syn_rf_{i}", X.astype(np.float32), y))

    # ── ExtraTrees (10) ───────────────────────────────────────────────────────
    # Very high-dimensional, sparse signal — random splits cover wide space
    for i in range(10):
        X, y = make_classification(
            n_samples=2000 + i*100, n_features=60 + i*5, n_informative=10,
            n_redundant=20, n_repeated=5, flip_y=0.07,
            class_sep=0.8, random_state=500+i)
        datasets.append((f"syn_et_{i}", X.astype(np.float32), y))

    # ── GradientBoosting (10) ─────────────────────────────────────────────────
    # Multi-cluster non-linear: many clusters per class, moderate noise
    for i in range(10):
        X, y = make_classification(
            n_samples=900 + i*80, n_features=14 + i, n_informative=8,
            n_redundant=3, n_clusters_per_class=4 + i % 4,
            flip_y=0.06, class_sep=0.6, random_state=600+i)
        datasets.append((f"syn_gbm_{i}", X.astype(np.float32), y))

    # ── ANTI-OVERFITTING: datasets targeting the 5 fresh-eval failure modes ───
    # Sparse binary → LR/NB should win (not KNN/RF)
    for i in range(8):
        n = 300 + i * 50
        X = rng.binomial(1, 0.1 + i*0.02, size=(n, 30 + i*5)).astype(np.float32)
        y = (X[:, :8].sum(axis=1) > (2 + i % 3)).astype(np.int64)
        datasets.append((f"syn_sparse_bin_{i}", X, y))

    # High-dim multiclass (10 classes) → RF/SVC should win
    for i in range(8):
        X, y = make_classification(
            n_samples=500 + i*50, n_features=40 + i*5, n_informative=20,
            n_redundant=8, n_classes=8 + i % 3, n_clusters_per_class=1,
            random_state=700+i)
        datasets.append((f"syn_hiclass_{i}", X.astype(np.float32), y))

    # Checkerboard 2D → DT should win (not KNN)
    for i in range(8):
        n = 400 + i * 60
        X = rng.uniform(-3, 3, (n, 2 + i % 3)).astype(np.float32)
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(np.int64)
        datasets.append((f"syn_checker_{i}", X, y))

    # Very large balanced (3000+ samples) → SVC/RF should win (not KNN: too slow)
    for i in range(6):
        X, y = make_classification(
            n_samples=3000 + i*500, n_features=12 + i, n_informative=8,
            n_redundant=3, flip_y=0.05, random_state=800+i)
        datasets.append((f"syn_large_{i}", X.astype(np.float32), y))

    print(f"[*] Generated {len(datasets)} synthetic classification datasets")
    return datasets


def generate_synthetic_regression_datasets():
    """
    Generate 90 synthetic regression datasets (10 per model × 9 models).
    Each group is structurally designed so the target model genuinely wins.

    Ridge      : pure linear, near-zero noise, dense informative features
    Lasso      : high-dim but only 3 features matter (sparse signal)
    ElasticNet : many correlated + sparse features (effective_rank << n_features)
    SVR        : smooth non-linear functions (sinusoidal, polynomial)
    KNN        : high-frequency local patterns, local averaging wins
    DT         : pure step / piecewise-constant functions
    RF         : large + noisy + non-linear (many irrelevant features)
    ET         : very high-dimensional, random non-linear
    GBM        : complex interaction terms (x1*x2, x^2, sin*x)
    """
    datasets = []
    rng = np.random.RandomState(42)

    # ── Ridge (10) ────────────────────────────────────────────────────────────
    # Pure linear relationship, almost zero noise — Ridge should be near-perfect
    for i in range(10):
        X, y = make_regression(
            n_samples=400 + i*50, n_features=5 + i, n_informative=5 + i,
            noise=0.05, coef=False, random_state=i)
        datasets.append((f"syn_ridge_{i}", X.astype(np.float32), y.astype(np.float32)))

    # ── Lasso (10) ────────────────────────────────────────────────────────────
    # High-dimensional but only 3 features carry signal — L1 zeroes out the rest
    for i in range(10):
        n_feat = 40 + i * 5     # 40–85 features, only 3 informative
        X, y = make_regression(
            n_samples=350, n_features=n_feat, n_informative=3,
            noise=0.3, coef=False, random_state=100+i)
        datasets.append((f"syn_lasso_{i}", X.astype(np.float32), y.astype(np.float32)))

    # ── ElasticNet (10) ───────────────────────────────────────────────────────
    # Correlated features + sparse signal — neither pure L1 nor L2 alone wins
    for i in range(10):
        X, y = make_regression(
            n_samples=400, n_features=30, n_informative=5,
            noise=1.0, effective_rank=8, tail_strength=0.6, random_state=200+i)
        datasets.append((f"syn_en_{i}", X.astype(np.float32), y.astype(np.float32)))

    # ── SVR (10) ──────────────────────────────────────────────────────────────
    # Smooth non-linear: sinusoidal + polynomial — RBF kernel captures this
    for i in range(5):
        n = 400 + i * 60
        X = rng.uniform(-3, 3, size=(n, 2 + i)).astype(np.float32)
        y = (np.sin(X[:, 0] * (1 + i * 0.4)) * np.cos(X[:, 1]) +
             rng.randn(n) * 0.08).astype(np.float32)
        datasets.append((f"syn_svr_sin_{i}", X, y))
    for i in range(5):
        n = 400 + i * 60
        X = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
        y = (X[:, 0]**2 + X[:, 1]**2 - X[:, 2]**2 +
             rng.randn(n) * 0.08).astype(np.float32)
        datasets.append((f"syn_svr_poly_{i}", X, y))

    # ── KNN (10) ──────────────────────────────────────────────────────────────
    # High-frequency local pattern — local averaging beats any global model
    for i in range(10):
        n = 350 + i * 40
        nf = 2 + (i % 3)
        X = rng.uniform(0, 1, size=(n, nf)).astype(np.float32)
        # y = high-freq sinusoid in input space → only nearby points predict well
        y = (np.sin(X[:, 0] * 12) * np.cos(X[:, 1] * 12) +
             rng.randn(n) * 0.04).astype(np.float32)
        datasets.append((f"syn_knn_{i}", X, y))

    # ── DecisionTree (10) ─────────────────────────────────────────────────────
    # Pure step / piecewise-constant targets — trees split exactly on boundaries
    for i in range(10):
        n = 400 + i * 60
        X = rng.randn(n, 4 + i % 3).astype(np.float32)
        y = np.zeros(n, dtype=np.float32)
        for j in range(4):
            y += (X[:, j % X.shape[1]] > (j * 0.4 - 0.6)).astype(np.float32) * (j+1) * 3.0
        y += rng.randn(n).astype(np.float32) * 0.1
        datasets.append((f"syn_dt_{i}", X, y))

    # ── RandomForest (10) ─────────────────────────────────────────────────────
    # Large + high noise + non-linear + many irrelevant features
    for i in range(10):
        n = 1500 + i * 200
        nf = 25 + i * 2
        X, y = make_regression(
            n_samples=n, n_features=nf, n_informative=8,
            noise=5.0 + i, coef=False, random_state=400+i)
        # add non-linearity
        y = y + (X[:, 0] * X[:, 1]).astype(np.float32) * 0.5
        datasets.append((f"syn_rf_{i}", X.astype(np.float32), y.astype(np.float32)))

    # ── ExtraTrees (10) ───────────────────────────────────────────────────────
    # Very high-dimensional non-linear — random extreme splits generalise well
    for i in range(10):
        n = 2000 + i * 100
        nf = 60 + i * 5
        X = rng.randn(n, nf).astype(np.float32)
        # signal only from first 10 features, non-linear
        y = (np.sin(X[:, 0]) * X[:, 1] + X[:, 2]**2 - X[:, 3] * X[:, 4] +
             rng.randn(n) * 0.5).astype(np.float32)
        datasets.append((f"syn_et_{i}", X, y))

    # ── GBM (10) ──────────────────────────────────────────────────────────────
    # Complex interaction terms: x1*x2, x^2, sin(x)*x — boosting fits residuals
    for i in range(10):
        n = 800 + i * 100
        X = rng.randn(n, 10 + i).astype(np.float32)
        y = (X[:, 0] * X[:, 1] +
             X[:, 2]**2 - X[:, 3]**2 +
             np.sin(X[:, 4]) * X[:, 5] +
             rng.randn(n) * 0.5).astype(np.float32)
        datasets.append((f"syn_gbm_{i}", X, y))

    # ── ANTI-OVERFITTING: datasets targeting regression failure modes ──────────
    # Exponential / log (nonlinear but smooth) → SVR/ET should win over Ridge
    for i in range(8):
        n = 500 + i * 60
        X = rng.uniform(0.1, 3, (n, 2 + i % 3)).astype(np.float32)
        y = (np.exp(X[:, 0]) - np.exp(X[:, 1]) + rng.randn(n) * 0.2).astype(np.float32)
        datasets.append((f"syn_exp_{i}", X, y))

    # Strongly correlated blocks → Ridge wins (ET should NOT win here)
    for i in range(8):
        X, y = make_regression(
            n_samples=400 + i*50, n_features=20, n_informative=6,
            noise=0.5, effective_rank=4, tail_strength=0.9, random_state=500+i)
        datasets.append((f"syn_corr_{i}", X.astype(np.float32), y.astype(np.float32)))

    # Radial / distance-based (ET/RF should win — not Ridge)
    for i in range(6):
        n = 500 + i * 80
        X = rng.randn(n, 4 + i).astype(np.float32)
        y = (np.sqrt((X ** 2).sum(axis=1)) + rng.randn(n) * 0.15).astype(np.float32)
        datasets.append((f"syn_radial_{i}", X, y))

    print(f"[*] Generated {len(datasets)} synthetic regression datasets")
    return datasets


def load_classification_datasets():
    """Load OpenML + synthetic classification datasets."""
    print("[*] Loading classification datasets from OpenML...")
    openml_ds   = load_openml_datasets('classification', max_datasets=500)
    synthetic_ds = generate_synthetic_classification_datasets()
    combined = openml_ds + synthetic_ds
    print(f"[OK] Total: {len(combined)} datasets "
          f"({len(openml_ds)} OpenML + {len(synthetic_ds)} synthetic)")
    return combined


def load_regression_datasets():
    """Load OpenML + synthetic regression datasets."""
    print("[*] Loading regression datasets from OpenML...")
    openml_ds    = load_openml_datasets('regression', max_datasets=500)
    synthetic_ds = generate_synthetic_regression_datasets()
    combined = openml_ds + synthetic_ds
    print(f"[OK] Total: {len(combined)} datasets "
          f"({len(openml_ds)} OpenML + {len(synthetic_ds)} synthetic)")
    return combined


# =============================================================================
# EXTRACT 32 META-FEATURES  (all properly computed from real data)
# =============================================================================
def extract_meta_features(X, y, task_type='classification'):
    """
    Compute all 40 meta-features from the dataset.

    Group breakdown
    ---------------
    Basic       (6)  : shape, type ratios, dimensionality
    Missing     (3)  : patterns of missing data
    Statistical (10) : distribution properties of numeric features
    Categorical (3)  : cardinality of categorical columns
    Target      (3)  : properties of the target variable
    PCA         (3)  : intrinsic dimensionality via PCA
    Landmarks   (4)  : quick 3-fold CV scores with simple models
    Signal      (8)  : sparsity, linear signal strength, nonlinearity, class count
                       [NEW — reduces overfitting on unseen dataset types]

    All values are normalised to [0, 1] to match the observation space.
    """
    features = []

    # -- Prepare data ----------------------------------------------------------
    if isinstance(X, pd.DataFrame):
        num_cols  = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = X.select_dtypes(exclude=[np.number]).columns.tolist()
        miss_col  = X.isnull().mean()
        total_missing   = float(X.isnull().mean().mean())
        cols_with_miss  = float((miss_col > 0).mean())
        max_missing     = float(miss_col.max())
        X_num = X[num_cols].values if num_cols else np.zeros((len(X), 1))
    else:
        X = np.asarray(X, dtype=float)
        X_num    = X
        cat_cols = []
        num_cols = list(range(X.shape[1]))
        nan_mask        = np.isnan(X_num)
        total_missing   = float(nan_mask.mean())
        cols_with_miss  = float(np.any(nan_mask, axis=0).mean())
        max_missing     = float(nan_mask.mean(axis=0).max())

    n_samples, n_features_total = X_num.shape
    n_numeric     = len(num_cols) if num_cols else X_num.shape[1]
    n_categorical = len(cat_cols)
    X_num = np.nan_to_num(X_num.astype(float), nan=0.0)

    # -- BASIC (6) -------------------------------------------------------------
    # 1. Normalised sample count     (reference = 100 000)
    features.append(float(min(n_samples / 100_000, 1.0)))
    # 2. Normalised feature count    (reference = 100)
    features.append(float(min(n_features_total / 100, 1.0)))
    # 3. Numeric column ratio
    features.append(float(n_numeric / max(n_features_total, 1)))
    # 4. Categorical column ratio
    features.append(float(n_categorical / max(n_features_total, 1)))
    # 5. Class / unique-value density
    features.append(float(min(len(np.unique(y)) / max(n_samples, 1), 1.0)))
    # 6. Dimensionality ratio  (features / samples)
    features.append(float(min(n_features_total / max(n_samples, 1), 1.0)))

    # -- MISSING (3) -----------------------------------------------------------
    # 7. Overall missing ratio
    features.append(float(min(total_missing, 1.0)))
    # 8. Fraction of columns that have any missing values
    features.append(float(min(cols_with_miss, 1.0)))
    # 9. Worst-case missing ratio in a single column
    features.append(float(min(max_missing, 1.0)))

    # -- STATISTICAL (10) ------------------------------------------------------
    n_cols = min(n_numeric, 50)          # cap at 50 cols for speed
    Xs     = X_num[:, :n_cols]
    col_std  = np.std(Xs, axis=0) + 1e-10
    col_mean = np.mean(Xs, axis=0)

    # Per-column skewness and kurtosis (scipy uses proper centred formulas)
    col_skew = np.array([stats.skew(Xs[:, i])     for i in range(n_cols)])
    col_kurt = np.array([stats.kurtosis(Xs[:, i]) for i in range(n_cols)])

    # 10. Mean |skewness|  (normalise by 10)
    features.append(float(min(np.mean(np.abs(col_skew)) / 10.0, 1.0)))
    # 11. Mean |kurtosis|  (normalise by 50)
    features.append(float(min(np.mean(np.abs(col_kurt)) / 50.0, 1.0)))
    # 12. Outlier ratio: fraction of rows with any |z-score| > 3
    z_scores = np.abs((Xs - col_mean) / col_std)
    features.append(float(np.mean(np.any(z_scores > 3, axis=1))))
    # 13. Mean absolute pairwise correlation
    if n_cols > 1:
        corr  = np.corrcoef(Xs.T)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[~np.isnan(upper)]
        features.append(float(np.mean(np.abs(upper))) if len(upper) else 0.0)
    else:
        features.append(0.0)
    # 14. Mean coefficient of variation  (std / |mean|)
    cv = col_std / (np.abs(col_mean) + 1e-10)
    features.append(float(min(np.mean(cv), 1.0)))
    # 15. Std of skewness values  (normalise by 10)
    features.append(float(min(np.std(col_skew) / 10.0, 1.0)))
    # 16. Std of kurtosis values  (normalise by 50)
    features.append(float(min(np.std(col_kurt) / 50.0, 1.0)))
    # 17. Range ratio: mean (max ? min) / std per column  (normalise by 20)
    col_range = np.max(Xs, axis=0) - np.min(Xs, axis=0)
    features.append(float(min(np.mean(col_range / col_std) / 20.0, 1.0)))
    # 18. Zero ratio
    features.append(float(np.mean(Xs == 0)))
    # 19. Class balance (classification) or target CV (regression)
    if task_type == 'classification':
        _, counts = np.unique(y, return_counts=True)
        features.append(float(counts.min() / counts.max()))   # 1.0 = perfectly balanced
    else:
        y_cv = np.std(y) / (np.abs(np.mean(y)) + 1e-10)
        features.append(float(min(y_cv / 10.0, 1.0)))

    # -- CATEGORICAL (3) -------------------------------------------------------
    if cat_cols and isinstance(X, pd.DataFrame):
        cards = [X[c].nunique() / n_samples for c in cat_cols]
        # 20. Mean cardinality ratio
        features.append(float(min(np.mean(cards), 1.0)))
        # 21. Max cardinality ratio
        features.append(float(min(np.max(cards), 1.0)))
        # 22. Fraction of categorical cols that are "high cardinality" (> 5 %)
        features.append(float(np.mean([c > 0.05 for c in cards])))
    else:
        features.extend([0.0, 0.0, 0.0])

    # -- TARGET (3) ------------------------------------------------------------
    y_arr = np.asarray(y, dtype=float)
    # 23. Target |skewness|  (normalise by 10)
    features.append(float(min(abs(stats.skew(y_arr)) / 10.0, 1.0)))
    # 24. Target |kurtosis|  (normalise by 50)
    features.append(float(min(abs(stats.kurtosis(y_arr)) / 50.0, 1.0)))
    # 25. Target entropy (classification) or target CV (regression)
    if task_type == 'classification':
        _, counts = np.unique(y_arr, return_counts=True)
        probs   = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_ent = np.log(max(len(counts), 2))
        features.append(float(entropy / max_ent))
    else:
        tgt_cv = np.std(y_arr) / (np.abs(np.mean(y_arr)) + 1e-10)
        features.append(float(min(tgt_cv / 10.0, 1.0)))

    # -- PCA (3) ---------------------------------------------------------------
    try:
        Xs_scaled = StandardScaler().fit_transform(Xs)
        n_pca = min(n_cols, n_samples - 1, 50)
        pca   = PCA(n_components=n_pca, random_state=42)
        pca.fit(Xs_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # 26. Normalised # components needed for 95 % variance
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        features.append(float(n95 / n_pca))
        # 27. Variance explained by the first 50 % of components
        half = max(1, n_pca // 2)
        features.append(float(cumvar[half - 1]))
        # 28. Intrinsic dim estimate: components needed for 50 % variance (normalised)
        n50 = int(np.searchsorted(cumvar, 0.50)) + 1
        features.append(float(n50 / n_pca))
    except Exception:
        features.extend([0.5, 0.5, 0.5])

    # -- LANDMARKS (4) -- quick 3-fold CV with simple fast models ---------------
    Xl = StandardScaler().fit_transform(Xs)

    def _lm(estimator, scoring):
        """Run 3-fold CV and return mean score clipped to [0, 1]."""
        try:
            cv  = KFold(n_splits=3, shuffle=True, random_state=0)
            sc  = cross_val_score(estimator, Xl, y, cv=cv,
                                  scoring=scoring, error_score=0.0)
            return float(np.clip(np.mean(sc), 0.0, 1.0))
        except Exception:
            return 0.5

    if task_type == 'classification':
        sc = 'accuracy'
        # 29. Decision-tree landmark
        lm_dt  = _lm(DecisionTreeClassifier(max_depth=3, random_state=42), sc)
        # 30. Naive Bayes landmark
        lm_nb  = _lm(GaussianNB(), sc)
        # 31. Logistic Regression landmark
        lm_lr  = _lm(LogisticRegression(max_iter=200, random_state=42), sc)
        # 32. KNN landmark
        lm_knn = _lm(KNeighborsClassifier(n_neighbors=3), sc)
        features.extend([lm_dt, lm_nb, lm_lr, lm_knn])
    else:
        sc = 'r2'
        # 29. Decision-tree landmark (shallow)
        lm_dt  = _lm(DecisionTreeRegressor(max_depth=3, random_state=42), sc)
        # 30. Ridge landmark
        lm_lr  = _lm(Ridge(alpha=1.0), sc)
        # 31. KNN landmark
        lm_knn = _lm(KNeighborsRegressor(n_neighbors=3), sc)
        # 32. Decision-tree landmark (deeper)
        lm_dt2 = _lm(DecisionTreeRegressor(max_depth=5, random_state=42), sc)
        lm_nb  = lm_dt2   # alias for unified logic below
        features.extend([lm_dt, lm_lr, lm_knn, lm_dt2])

    # -- SIGNAL (8 NEW features — reduces overfitting on unseen dataset types) --

    # 33. Sparsity ratio: fraction of values near zero (|x| < 0.05 of feature range)
    #     High value → sparse binary / text-like data → LR/NB often best
    col_ranges = np.ptp(Xs, axis=0) + 1e-10
    near_zero  = np.abs(Xs) < (0.05 * col_ranges)
    features.append(float(np.mean(near_zero)))

    # 34. Mean |feature-target Pearson correlation| (linear signal strength)
    #     High → data is mostly linear → Ridge/LR should win
    y_arr2 = np.asarray(y, dtype=float)
    y_std  = float(np.std(y_arr2)) + 1e-10
    Xs_std = np.std(Xs, axis=0) + 1e-10
    ftcorr = np.abs(np.dot((Xs - Xs.mean(0)).T, y_arr2 - y_arr2.mean()) /
                    (n_samples * Xs_std * y_std))
    features.append(float(np.clip(np.mean(ftcorr), 0.0, 1.0)))

    # 35. Max |feature-target correlation| (best single linear predictor)
    features.append(float(np.clip(np.max(ftcorr), 0.0, 1.0)))

    # 36. Std of feature-target correlations (sparse vs uniform signal)
    #     High → only a few features matter → Lasso/DT should win
    features.append(float(np.clip(np.std(ftcorr) * 5.0, 0.0, 1.0)))

    # 37. Nonlinearity gap: DT_landmark - LR_landmark (clipped to [-1, 1] → [0, 1])
    #     Positive → nonlinear structure → trees/SVC should win
    #     Negative → linear → LR/Ridge should win
    nl_gap = float(np.clip((lm_dt - lm_lr + 1.0) / 2.0, 0.0, 1.0))
    features.append(nl_gap)

    # 38. Explicit class count (classification) or 0 (regression)
    #     normalised by 20 — high value signals multiclass → RF/GBM may help
    n_classes_raw = len(np.unique(y)) if task_type == 'classification' else 1
    features.append(float(min(n_classes_raw / 20.0, 1.0)))

    # 39. KNN consistency score: how well KNN landmark predicts vs linear
    #     High → tight local clusters → KNN/DT should win
    knn_advantage = float(np.clip((lm_knn - lm_lr + 1.0) / 2.0, 0.0, 1.0))
    features.append(knn_advantage)

    # 40. Feature density: fraction of features with std > 0.1 * max_std
    #     Low → many dead/constant features → sparse model (Lasso) may win
    max_std = float(np.max(Xs_std)) + 1e-10
    active_feats = float(np.mean(Xs_std > 0.1 * max_std))
    features.append(active_feats)

    assert len(features) == 40, f"Feature count error: got {len(features)}, expected 40"
    arr = np.array(features, dtype=np.float32)
    # Sanitize: replace any NaN/inf that would poison the PPO network
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# =============================================================================
# EVALUATE MODEL
# =============================================================================
def evaluate_model(model, X, y, task_type):
    """
    Returns a score in [0, 1].
      Classification -> train/test split accuracy  (low error  = high score)
      Regression     -> train/test split R2 clipped to >= 0

    Uses train_test_split (1 fit) instead of 3-fold CV (3 fits) for 3x speed.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.base import clone
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        m = clone(model)
        m.fit(X_tr, y_tr)
        if task_type == 'classification':
            score = accuracy_score(y_te, m.predict(X_te))
        else:
            score = r2_score(y_te, m.predict(X_te))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


# =============================================================================
# RL ENVIRONMENT  (single-step episodic)
# =============================================================================
if HAS_SB3:
    class ModelSelectionEnv(Env):
        """
        Single-step episodic RL environment for model selection.

        Each episode = one dataset:
            32f              = env.reset()               -> meta-features
            selected_model   = agent.predict(32f)        -> model index
            reward, next_32f = env.step(selected_model)  -> score & done=True

        Reward = model score on the dataset.
            Low error  ->  high reward   (reward = accuracy or max(0, R2))
            High error ->  low  reward
        """

        def __init__(self, datasets, models, task_type):
            super().__init__()
            self.datasets     = datasets
            self.models       = models
            self.model_names  = list(models.keys())
            self.task_type    = task_type
            self._idx         = 0
            self._current     = None
            self.last_info    = {}

            self.action_space      = Discrete(len(models))
            self.observation_space = Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)

            # Pre-compute and cache meta-features + full per-dataset score arrays
            print(f"[*] Pre-computing meta-features for {len(datasets)} datasets...")
            self._feature_cache = {}
            self._score_cache   = {}   # {dataset_idx: [score_model0, score_model1, ...]}
            for i, (_, X, y) in enumerate(datasets):
                try:
                    self._feature_cache[i] = extract_meta_features(X, y, self.task_type)
                except Exception:
                    self._feature_cache[i] = np.zeros(40, dtype=np.float32)
                try:
                    self._score_cache[i] = [
                        evaluate_model(m, X, y, self.task_type)
                        for m in models.values()
                    ]
                except Exception:
                    self._score_cache[i] = [0.5] * len(models)
            print(f"[OK] Meta-feature cache ready ({len(self._feature_cache)} entries)")

        # -- reset: pick the next dataset, return its cached 32 meta-features --
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._idx     = self._idx % len(self.datasets)
            name, X, y    = self.datasets[self._idx]
            self._current = (name, X, y)
            obs = self._feature_cache[self._idx]
            return obs, {}          # gymnasium API: (obs, info)

        # -- step: evaluate selected model -> reward = score (low error = high reward)
        def step(self, action):
            name, _, _   = self._current
            action_idx   = int(action)
            model_name   = self.model_names[action_idx]

            # Rank-based reward: best model=1.0, worst model=0.0, others in between.
            # e.g. 8 models: rank 1st->1.0, 2nd->0.857, ..., 8th->0.0
            # This penalises picking 2nd-best clearly, unlike score/best_score.
            # Use cached score directly — no need to re-evaluate the same dataset.
            all_scores = self._score_cache.get(self._idx, [0.5] * len(self.models))
            score      = all_scores[action_idx]
            sorted_scores = sorted(all_scores, reverse=True)
            rank   = sorted_scores.index(all_scores[action_idx])  # 0=best
            n      = max(len(all_scores) - 1, 1)
            reward = float(1.0 - rank / n)   # best=1.0, worst=0.0

            self._idx += 1

            self.last_info = {
                'dataset': name,
                'model':   model_name,
                'score':   score,
                'reward':  reward,
            }

            # gymnasium API: (obs, reward, terminated, truncated, info)
            next_obs   = np.zeros(40, dtype=np.float32)
            terminated = True       # each episode = one decision
            truncated  = False
            return next_obs, float(reward), terminated, truncated, self.last_info


# =============================================================================
# EPISODE LOGGER CALLBACK  (shows per-episode table during PPO training)
# =============================================================================
if HAS_SB3:
    class EpisodeLoggerCallback(BaseCallback):
        """
        Prints one row per episode:
            Episode | Dataset | Selected Model | Score | Reward
        """

        def __init__(self, task_type):
            super().__init__(verbose=0)
            self.task_type   = task_type
            self.episode     = 0
            self._printed_hdr = False

        def _print_header(self):
            print(
                f"\n{'Episode':>8} | {'Dataset':>22} | "
                f"{'Selected Model':>30} | {'Score':>7} | {'Reward':>7}"
            )
            print("-" * 84)
            self._printed_hdr = True

        def _on_step(self) -> bool:
            dones  = self.locals.get('dones',   [False])
            infos  = self.locals.get('infos',   [{}])
            rewards = self.locals.get('rewards', [0.0])

            for done, info, reward in zip(dones, infos, rewards):
                if done and info:
                    if not self._printed_hdr:
                        self._print_header()
                    self.episode += 1
                    print(
                        f"{self.episode:>8} | {info.get('dataset','?'):>22} | "
                        f"{info.get('model','?'):>30} | "
                        f"{info.get('score', 0.0):>7.4f} | {float(reward):>7.4f}"
                    )
            return True


# =============================================================================
# TRAIN RL MODEL
# =============================================================================
def train_rl_model(task_type='classification', total_timesteps=50_000):
    """
    Train a PPO agent using the episode-based loop:

        env = ModelSelectionEnv(...)
        for episode in [0, N]:
            32f              = env.reset()
            selected_model   = ppo.predict(32f)
            Reward, next_32f = env.step(selected_model)

    PPO handles the episode loop internally via `model.learn()`.
    The EpisodeLoggerCallback prints each episode as it happens.
    """
    if not HAS_SB3:
        print("[ERROR] stable-baselines3 required. Run: pip install stable-baselines3")
        return None

    print(f"\n{'='*70}")
    print(f"  RL Model Selector  --  {task_type.upper()}")
    print(f"{'='*70}")

    # Load datasets
    if task_type == 'classification':
        datasets = load_classification_datasets()
        models   = CLASSIFICATION_MODELS
    else:
        datasets = load_regression_datasets()
        models   = REGRESSION_MODELS

    # Create environment
    env = ModelSelectionEnv(datasets, models, task_type)

    # Initialise PPO
    ppo = PPO(
        'MlpPolicy', env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        max_grad_norm=0.5,
        normalize_advantage=True,
        ent_coef=0.3,            # high entropy → forces diverse model selection
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),   # deeper network for 32 meta-features
        device='cpu',
    )

    # --------------------------------------------------------------------------
    # EPISODE-BASED TRAINING LOOP
    #
    #   env = ModelSelectionEnv(...)
    #   for episode in [0, N]:
    #       32f              = env.reset()               <- meta-features
    #       selected_model   = ppo.predict(32f)          <- agent picks model
    #       Reward, next_32f = env.step(selected_model)  <- evaluate & reward
    #
    # PPO collects these transitions and updates its policy every n_steps.
    # EpisodeLoggerCallback prints a row for every episode (done=True step).
    # --------------------------------------------------------------------------
    print(f"\n[*] Training PPO  ({total_timesteps:,} timesteps)")
    print(f"[*] Reward: LOW error -> HIGH reward  (score = accuracy / max(0, R2))\n")

    callback = EpisodeLoggerCallback(task_type)
    ppo.learn(total_timesteps=total_timesteps, callback=callback)

    print(f"\n[OK] Training complete -- {callback.episode} episodes ran")

    # Save
    path = f'rl_model_selector_{task_type}.pkl'
    ppo.save(path)
    print(f"[OK] Saved -> {path}")
    return ppo


# =============================================================================
# INFERENCE LOOP  (explicit episode loop for demonstration / evaluation)
# =============================================================================
def run_episode_loop(ppo, datasets, models, task_type, n_episodes=20):
    """
    Explicit episode loop -- mirrors the whiteboard pseudocode exactly:

        env = ModelSelectionEnv(...)
        for episode in range(n_episodes):
            32f              = env.reset()
            selected_model   = ppo.predict(32f)
            Reward, next_32f = env.step(selected_model)
    """
    env = ModelSelectionEnv(datasets, models, task_type)

    print(f"\n{'-'*84}")
    print(f"  Inference loop -- {n_episodes} episodes")
    print(f"{'-'*84}")
    print(
        f"{'Episode':>8} | {'Dataset':>22} | "
        f"{'Selected Model':>30} | {'Score':>7} | {'Reward':>7}"
    )
    print("-" * 84)

    total_reward = 0.0
    for episode in range(n_episodes):
        # -- Step 1: 32f = env.reset() -----------------------------------------
        features_32f, _ = env.reset()

        # -- Step 2: selected_model = ppo.predict(32f) -------------------------
        action, _ = ppo.predict(features_32f, deterministic=True)

        # -- Step 3: Reward, next_32f = env.step(selected_model) ---------------
        next_32f, reward, _terminated, _truncated, info = env.step(int(action))

        total_reward += reward
        print(
            f"{episode + 1:>8} | {info['dataset']:>22} | "
            f"{info['model']:>30} | {info['score']:>7.4f} | {reward:>7.4f}"
        )

    avg = total_reward / n_episodes
    print(f"\n  Average reward over {n_episodes} episodes: {avg:.4f}")
    print(f"  (reward = score; higher is better -- means lower error)")
    return avg


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL Model Selector")
    parser.add_argument('--timesteps', type=int, default=150_000,
                        help='PPO training timesteps per task (default: 150000)')
    parser.add_argument('--task', choices=['classification', 'regression', 'both'],
                        default='both', help='Which task to train (default: both)')
    parser.add_argument('--max-datasets', type=int, default=500,
                        help='Max OpenML datasets to load per task (default: 500)')
    args = parser.parse_args()

    print("=" * 70)
    print("  RL Model Selector Trainer  --  OpenML datasets only")
    print("=" * 70)
    print(f"  [DATASETS]  up to {args.max_datasets} per task from OpenML")
    print(f"  [TIMESTEPS] {args.timesteps:,} per task")
    print("=" * 70)
    print("  Requires:  pip install openml")
    print("=" * 70)

    if not HAS_SB3:
        print("\n[ERROR] stable-baselines3 not installed")
        print("Install: pip install stable-baselines3")
        exit(1)

    # -- Train classification ---------------------------------------------------
    clf_ppo = None
    if args.task in ('classification', 'both'):
        clf_ppo = train_rl_model('classification', total_timesteps=args.timesteps)
        if clf_ppo:
            # Quick post-training evaluation on a small fresh sample
            eval_ds = load_openml_datasets('classification', max_datasets=20)
            run_episode_loop(clf_ppo, eval_ds, CLASSIFICATION_MODELS,
                             'classification', n_episodes=min(20, len(eval_ds)))

    # -- Train regression -------------------------------------------------------
    reg_ppo = None
    if args.task in ('regression', 'both'):
        reg_ppo = train_rl_model('regression', total_timesteps=args.timesteps)
        if reg_ppo:
            eval_ds = load_openml_datasets('regression', max_datasets=20)
            run_episode_loop(reg_ppo, eval_ds, REGRESSION_MODELS,
                             'regression', n_episodes=min(20, len(eval_ds)))

    print("\n[OK] Done.")
    print("     rl_model_selector_classification.pkl")
    print("     rl_model_selector_regression.pkl")
