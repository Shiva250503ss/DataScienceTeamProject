"""
RL Model Selector — Validation Test (Extended)
================================================
Tests whether the trained PPO agent (from train_rl_model_selector.py)
picks good models for unseen datasets.

Test suites:
  A. Synthetic datasets     — 32 classification + 36 regression (diverse geometries)
  B. OpenML held-out data   — 30 classification + 30 regression real-world datasets
                               (downloaded with seeds/IDs never used in training)

Checks:
  [BEST]  — RL picked the exact best model
  [CLOSE] — RL's model is within 5 % accuracy / 10 % R² of the best
  [MISS]  — RL's model is further away
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import (make_classification, make_regression,
                               make_moons, make_circles, make_blobs,
                               make_swiss_roll, make_s_curve,
                               make_friedman1, make_friedman2, make_friedman3)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Import the SAME functions / models used during training ──────────────────
from train_rl_model_selector import (
    extract_meta_features,
    evaluate_model,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
)

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARNING] stable-baselines3 not installed.")

try:
    import openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False
    print("[WARNING] openml not installed — real-world tests will be skipped.")


# =============================================================================
# HELPERS
# =============================================================================
# Paths must match what train_rl_model_selector.py saves
CLF_MODEL_PATH = "rl_model_selector_classification.pkl"
REG_MODEL_PATH = "rl_model_selector_regression.pkl"

CLF_NAMES = list(CLASSIFICATION_MODELS.keys())   # same order as action space
REG_NAMES = list(REGRESSION_MODELS.keys())


def _load_ppo(path):
    """Load a saved PPO model; return None on failure."""
    try:
        m = PPO.load(path)
        print(f"[OK] Loaded {path}")
        return m
    except Exception as e:
        print(f"[FAIL] Could not load {path}: {e}")
        return None


def _all_scores(X, y, model_dict, task_type):
    """3-fold CV score for every model in the dict."""
    scoring = 'accuracy' if task_type == 'classification' else 'r2'
    cv      = KFold(n_splits=3, shuffle=True, random_state=42)
    Xs      = StandardScaler().fit_transform(X)
    scores  = {}
    for name, model in model_dict.items():
        try:
            s = cross_val_score(model, Xs, y, cv=cv,
                                scoring=scoring, error_score=0.0)
            scores[name] = float(np.clip(np.mean(s), 0.0, 1.0))
        except Exception:
            scores[name] = 0.0
    return scores


# =============================================================================
# OPENML HELD-OUT DATASET LOADER
# =============================================================================

# These dataset IDs were hand-picked to be diverse and are NOT in the training
# pool (training uses the first 500 sorted by size; these are larger / different)
OPENML_CLF_IDS = [
    40966,  # MiceProtein (1080s, 77f, 8 classes — protein expression)
    1462,   # banknote-authentication (1372s, 4f — image features)
    1510,   # wdbc (569s, 30f — breast cancer Wisconsin)
    40982,  # steel-plates-fault (1941s, 27f, 7 classes)
    1464,   # blood-transfusion (748s, 4f — imbalanced)
    40983,  # wilt (4839s, 5f — remote sensing)
    40975,  # car (1728s, 6f, 4 classes — car evaluation)
    1063,   # kc2 (522s, 21f — software defect)
    1068,   # pc1 (1109s, 21f — software defect)
    40984,  # segment (2310s, 19f, 7 classes — image segmentation)
    23,     # cmc (1473s, 9f, 3 classes — contraceptive)
    29,     # credit-approval (690s, 6f)
    31,     # credit-g (1000s, 7f — German credit)
    37,     # diabetes (768s, 8f — Pima Indians)
    50,     # tic-tac-toe (958s, 9f)
    54,     # vehicle (846s, 18f, 4 classes)
    188,    # eucalyptus (736s, 16f, 5 classes)
    458,    # analcatdata_authorship (841s, 70f, 4 classes)
    469,    # analcatdata_dmft (797s, 4f, 6 classes)
    1049,   # pc4 (1458s, 37f — software defect)
    1050,   # pc3 (1563s, 37f — software defect)
    1067,   # kc1 (2109s, 21f — software defect)
    1461,   # bank-marketing (45211s, 7f)
    1471,   # eeg-eye-state (14980s, 14f)
    1478,   # har (10299s, 561f, 6 classes — human activity)
    40670,  # dna (3186s, 180f, 3 classes)
    40701,  # churn (5000s, 18f)
    41027,  # jungle_chess (44819s, 6f, 3 classes)
    4534,   # PhishingWebsites (11055s, 30f)
    1480,   # ilpd (583s, 9f — Indian liver patient)
]

OPENML_REG_IDS = [
    531,    # boston (506s, 13f — housing prices)
    42225,  # diamonds (53940s, 6f — diamond prices)
    507,    # space_ga (3107s, 6f — space shuttle)
    42570,  # superconduct (21263s, 81f — superconductor temp)
    42571,  # abalone (4177s, 7f — abalone age)
    287,    # wine_quality (1599s, 11f — red wine)
    422,    # analcatdata_vehicle (48s, 3f)
    41021,  # Moneyball (1232s, 14f)
    42636,  # SAT11-HAND-runtime (4440s, 115f)
    574,    # house_16H (22784s, 16f)
    41540,  # black_friday (166821s, 5f)
    41980,  # Brazilian_houses (10692s, 8f)
    546,    # sensory (576s, 11f)
    547,    # no2 (500s, 7f)
    550,    # quake (2178s, 3f)
    41539,  # MiamiHousing2016 (13932s, 14f)
    41983,  # particulate-matter (398s, 7f)
    1030,   # ERA (1000s, 4f)
    23515,  # Bike_Sharing_Demand (17379s, 11f)
    42726,  # SatelliteTemperature (1066s, 23f)
    541,    # sleeve (9s, 10f — small)
    215,    #2dplanes (40768s, 10f)
    216,    # elevators (16599s, 18f)
    218,    # house_8L (22784s, 8f)
    537,    # houses (20640s, 8f)
    42688,  # Mercedes_Benz_Greener (4209s, 359f)
    42563,  # nyc-taxi-green-dec-2016 (581835s, 16f)
    42092,  # Bike_Sharing (731s, 13f)
    42727,  # us_crime (1994s, 100f)
    42730,  # socmob (1156s, 5f)
]


def _download_openml_dataset(did, task_type, max_samples=5000):
    """Download one OpenML dataset; return (name, X, y) or None."""
    try:
        ds = openml.datasets.get_dataset(
            did, download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_raw, y_raw, _, _ = ds.get_data(
            target=ds.default_target_attribute,
            dataset_format='dataframe',
        )
        if X_raw is None or y_raw is None:
            return None

        X_num = X_raw.select_dtypes(include='number').dropna(axis=1, how='all')
        if X_num.shape[1] == 0:
            return None

        X_arr = X_num.fillna(X_num.median()).values.astype(np.float32)
        y_arr = y_raw.values

        if y_arr.dtype == object or str(y_arr.dtype) == 'category':
            y_arr = LabelEncoder().fit_transform(y_arr.astype(str))

        y_arr = y_arr.astype(float if task_type == 'regression' else int)

        if len(X_arr) > max_samples:
            idx = np.random.default_rng(99).choice(len(X_arr), max_samples, replace=False)
            X_arr, y_arr = X_arr[idx], y_arr[idx]

        return (ds.name, X_arr, y_arr)
    except Exception as e:
        print(f"   [SKIP] OpenML id={did}: {str(e)[:80]}")
        return None


def _load_openml_held_out(id_list, task_type, max_datasets=30):
    """Download held-out OpenML datasets for evaluation."""
    datasets = []
    for did in id_list:
        if len(datasets) >= max_datasets:
            break
        result = _download_openml_dataset(did, task_type)
        if result is not None:
            name, X, y = result
            print(f"   [{len(datasets)+1:>2}/{max_datasets}] {name:40s} shape={X.shape}")
            datasets.append(result)
    return datasets


# =============================================================================
# SYNTHETIC CLASSIFICATION DATASETS (expanded: 32 tests)
# =============================================================================
def _make_diverse_clf_dataset(i, rng):
    """Return (X, y, description) — each index produces a structurally different dataset."""
    kind = i % 16
    if kind == 0:   # linearly separable (LR should win)
        nf = 3 + i % 5
        X, y = make_classification(n_samples=400+i*40, n_features=nf,
                                   n_informative=nf, n_redundant=0,
                                   class_sep=3.0, flip_y=0.0, random_state=700+i)
        desc = f"linear sep | {nf}f"
    elif kind == 1: # tiny independent Gaussian (GNB should win)
        n = 80 + i * 10
        X = rng.randn(n, 4).astype(np.float32)
        y = (rng.rand(n) > 0.5).astype(np.int64)
        X[y==0, 0] += 2.5; X[y==1, 0] -= 2.5
        desc = f"tiny Gaussian | {n}s"
    elif kind == 2: # tight blobs (KNN should win)
        X, y = make_blobs(n_samples=300+i*30, n_features=2, centers=4,
                          cluster_std=0.25, random_state=800+i)
        y = y.astype(np.int64)
        desc = "tight blobs 2D"
    elif kind == 3: # moons (SVC/nonlinear should win)
        X, y = make_moons(n_samples=400+i*50, noise=0.1, random_state=900+i)
        desc = "moons (nonlinear)"
    elif kind == 4: # circles (SVC should win)
        X, y = make_circles(n_samples=400+i*50, noise=0.05, factor=0.5,
                            random_state=950+i)
        desc = "circles (nonlinear)"
    elif kind == 5: # XOR (DT should win)
        n = 300 + i * 40
        X = rng.randn(n, 4).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
        desc = f"XOR | {n}s"
    elif kind == 6: # large noisy many-feature (RF/ET should win)
        nf = 30 + i * 3
        X, y = make_classification(n_samples=1500+i*100, n_features=nf,
                                   n_informative=8, n_redundant=10,
                                   flip_y=0.12, class_sep=0.7,
                                   random_state=1000+i)
        desc = f"noisy {nf}f large"
    elif kind == 7: # multi-cluster (GBM should win)
        X, y = make_classification(n_samples=800+i*80, n_features=12,
                                   n_informative=8, n_redundant=2,
                                   n_clusters_per_class=4, flip_y=0.06,
                                   class_sep=0.6, random_state=1100+i)
        desc = "multi-cluster GBM"
    # ── NEW kinds below ──────────────────────────────────────────────────────
    elif kind == 8: # Swiss roll binarized
        X, t = make_swiss_roll(n_samples=600, noise=0.4, random_state=2000+i)
        y = (t > np.median(t)).astype(int)
        desc = "Swiss roll 2-class"
    elif kind == 9: # heavily imbalanced 95:5
        X, y = make_classification(n_samples=1000+i*50, n_features=15,
                                   n_informative=8, weights=[0.95, 0.05],
                                   flip_y=0.01, random_state=2100+i)
        desc = "imbalanced 95:5"
    elif kind == 10: # 6-class multiclass
        X, y = make_classification(n_samples=900+i*40, n_features=20,
                                   n_informative=12, n_classes=6,
                                   n_clusters_per_class=1, random_state=2200+i)
        desc = "6-class 20f"
    elif kind == 11: # sparse binary features
        n = 500 + i * 30
        X = rng.binomial(1, 0.12, size=(n, 60)).astype(np.float32)
        y = (X[:, :10].sum(axis=1) > 2).astype(np.int64)
        desc = f"sparse binary 60f"
    elif kind == 12: # checkerboard
        n = 600 + i * 40
        X = rng.uniform(-3, 3, (n, 2)).astype(np.float32)
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(np.int64)
        desc = "checkerboard 2D"
    elif kind == 13: # very high noise (35% label flip)
        X, y = make_classification(n_samples=600+i*30, n_features=20,
                                   n_informative=5, n_redundant=5,
                                   flip_y=0.35, random_state=2300+i)
        desc = "very noisy 35% flip"
    elif kind == 14: # wide dataset (more features than samples)
        X, y = make_classification(n_samples=150, n_features=200,
                                   n_informative=15, n_redundant=30,
                                   random_state=2400+i)
        desc = "wide 200f > 150s"
    else:           # very large 5000+ samples
        X, y = make_classification(n_samples=5000+i*200, n_features=25,
                                   n_informative=15, n_redundant=5,
                                   flip_y=0.04, random_state=2500+i)
        desc = f"very large {5000+i*200}s"
    return np.array(X, dtype=np.float32), np.array(y), desc


# =============================================================================
# SYNTHETIC REGRESSION DATASETS (expanded: 36 tests)
# =============================================================================
def _make_diverse_reg_dataset(i, rng):
    """Return (X, y, description) — each index produces a structurally different dataset."""
    kind = i % 18
    if kind == 0:   # pure linear (Ridge should win)
        X, y = make_regression(n_samples=400+i*40, n_features=5+i%6,
                               n_informative=5+i%6, noise=0.05,
                               coef=False, random_state=700+i)
        desc = "pure linear"
    elif kind == 1: # sparse high-dim (Lasso should win)
        nf = 40 + i * 5
        X, y = make_regression(n_samples=350, n_features=nf, n_informative=3,
                               noise=0.3, coef=False, random_state=800+i)
        desc = f"sparse {nf}f"
    elif kind == 2: # correlated+sparse (ElasticNet should win)
        X, y = make_regression(n_samples=400, n_features=30, n_informative=5,
                               noise=1.0, effective_rank=8, tail_strength=0.6,
                               random_state=900+i)
        desc = "correlated sparse"
    elif kind == 3: # sinusoidal (SVR should win)
        n = 400 + i * 60
        X = rng.uniform(-3, 3, size=(n, 2)).astype(np.float32)
        y = (np.sin(X[:, 0] * 2) * np.cos(X[:, 1]) + rng.randn(n) * 0.08).astype(np.float32)
        desc = "sinusoidal"
    elif kind == 4: # local high-freq (KNN should win)
        n = 400 + i * 50
        X = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
        y = (np.sin(X[:, 0]*12) * np.cos(X[:, 1]*12) + rng.randn(n)*0.04).astype(np.float32)
        desc = "high-freq local"
    elif kind == 5: # step function (DT should win)
        n = 500 + i * 60
        X = rng.randn(n, 4).astype(np.float32)
        y = sum((X[:, j] > (j*0.4-0.6)).astype(np.float32) * (j+1)*3.0 for j in range(4))
        y += rng.randn(n).astype(np.float32) * 0.1
        desc = "step function"
    elif kind == 6: # large noisy nonlinear (RF should win)
        n = 1500 + i * 100
        nf = 25 + i * 2
        X, y = make_regression(n_samples=n, n_features=nf, n_informative=8,
                               noise=6.0, coef=False, random_state=1000+i)
        y = y.astype(np.float32) + (X[:, 0] * X[:, 1]) * 0.5
        desc = f"noisy {nf}f large"
    elif kind == 7: # very high-dim random nonlinear (ET should win)
        n = 1500 + i * 100
        nf = 60 + i * 5
        X = rng.randn(n, nf).astype(np.float32)
        y = (np.sin(X[:, 0]) * X[:, 1] + X[:, 2]**2 - X[:, 3]*X[:, 4] +
             rng.randn(n) * 0.5).astype(np.float32)
        desc = f"high-dim {nf}f nonlinear"
    elif kind == 8: # interaction terms (GBM should win)
        n = 900 + i * 80
        X = rng.randn(n, 10).astype(np.float32)
        y = (X[:,0]*X[:,1] + X[:,2]**2 - X[:,3]**2 +
             np.sin(X[:,4])*X[:,5] + rng.randn(n)*0.5).astype(np.float32)
        desc = "interaction terms"
    # ── NEW kinds below ──────────────────────────────────────────────────────
    elif kind == 9: # Friedman #1 (nonlinear interactions)
        X, y = make_friedman1(n_samples=800+i*50, n_features=10, noise=0.3,
                              random_state=3000+i)
        desc = "Friedman1 10f"
    elif kind == 10: # Friedman #2 (physics-based)
        X, y = make_friedman2(n_samples=600+i*40, noise=0.5, random_state=3100+i)
        desc = "Friedman2 physics"
    elif kind == 11: # Friedman #3 (arctan-based)
        X, y = make_friedman3(n_samples=600+i*40, noise=0.3, random_state=3200+i)
        desc = "Friedman3 arctan"
    elif kind == 12: # exponential relationship
        n = 600 + i * 50
        X = rng.uniform(0.1, 3, (n, 3)).astype(np.float32)
        y = (np.exp(X[:, 0]) - np.exp(X[:, 1]) + X[:, 2]**2 +
             rng.randn(n) * 0.3).astype(np.float32)
        desc = "exponential 3f"
    elif kind == 13: # radial distance
        n = 700 + i * 50
        X = rng.randn(n, 5).astype(np.float32)
        y = (np.sqrt((X ** 2).sum(axis=1)) + rng.randn(n) * 0.1).astype(np.float32)
        desc = "radial distance 5f"
    elif kind == 14: # very high noise
        X, y = make_regression(n_samples=600+i*30, n_features=10,
                               n_informative=5, noise=25.0,
                               random_state=3300+i)
        desc = "very high noise s=25"
    elif kind == 15: # wide: more features than samples
        X, y = make_regression(n_samples=120, n_features=200,
                               n_informative=5, noise=0.5,
                               random_state=3400+i)
        desc = "wide 200f > 120s"
    elif kind == 16: # cubic polynomial
        n = 800 + i * 60
        X = rng.uniform(-2, 2, (n, 4)).astype(np.float32)
        y = (X[:, 0]**3 - 2*X[:, 1]**2 + X[:, 2]*X[:, 3] +
             rng.randn(n) * 0.3).astype(np.float32)
        desc = "cubic polynomial 4f"
    else:           # very large linear
        X, y = make_regression(n_samples=5000+i*200, n_features=20,
                               n_informative=20, noise=1.0,
                               random_state=3500+i)
        desc = f"very large linear {5000+i*200}s"
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), desc


# =============================================================================
# RUN TEST SUITE (synthetic + real-world)
# =============================================================================
def _run_test(ppo, X, y, desc, model_dict, model_names, task_type, close_thresh):
    """Run a single test case and return the result dict."""
    try:
        feats = extract_meta_features(X, y, task_type=task_type)
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        print(f"   [ERROR] Feature extraction failed: {e}")
        return None

    action, _ = ppo.predict(feats, deterministic=True)
    rl_choice = model_names[int(action)]

    scores     = _all_scores(X, y, model_dict, task_type)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    rl_score   = scores[rl_choice]

    is_best  = rl_choice == best_model
    gap      = best_score - rl_score
    is_close = gap < close_thresh

    metric = "accuracy" if task_type == 'classification' else "R²"
    status = "[BEST]" if is_best else ("[CLOSE]" if is_close else "[MISS]")

    print(f"   RL choice  : {rl_choice:<35}  {metric} = {rl_score:.4f}")
    print(f"   Actual best: {best_model:<35}  {metric} = {best_score:.4f}")
    print(f"   Result     : {status}   gap = {gap:.4f}")

    return {'is_best': is_best, 'is_close': is_close,
            'gap': gap, 'rl_choice': rl_choice}


def test_classification(n_synthetic=32, n_openml=30):
    print("\n" + "=" * 70)
    print("  CLASSIFICATION TEST — synthetic + real-world")
    print("=" * 70)

    ppo = _load_ppo(CLF_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results_syn = []
    results_real = []
    rng = np.random.RandomState(42)

    # ── A. Synthetic datasets ────────────────────────────────────────────────
    print(f"\n--- Synthetic datasets ({n_synthetic}) ---")
    for i in range(n_synthetic):
        X, y, desc = _make_diverse_clf_dataset(i, rng)
        print(f"\n[Syn {i+1}]  {X.shape[0]}s | {X.shape[1]}f | {desc}")
        r = _run_test(ppo, X, y, desc, CLASSIFICATION_MODELS, CLF_NAMES,
                      'classification', 0.05)
        if r:
            results_syn.append(r)

    # ── B. OpenML real-world datasets ────────────────────────────────────────
    if HAS_OPENML:
        print(f"\n--- OpenML held-out real-world datasets (up to {n_openml}) ---")
        print("[*] Downloading held-out datasets (never seen during training)...")
        openml_ds = _load_openml_held_out(OPENML_CLF_IDS, 'classification',
                                           max_datasets=n_openml)
        for idx, (name, X, y) in enumerate(openml_ds):
            print(f"\n[Real {idx+1}]  {X.shape[0]}s | {X.shape[1]}f | {name}")
            r = _run_test(ppo, X, y, name, CLASSIFICATION_MODELS, CLF_NAMES,
                          'classification', 0.05)
            if r:
                results_real.append(r)
    else:
        print("\n[SKIP] OpenML not installed — skipping real-world tests")

    # ── Combined summary ─────────────────────────────────────────────────────
    all_results = results_syn + results_real
    syn_pass  = _summarise(results_syn,  "CLASSIFICATION — SYNTHETIC",
                           "accuracy gap", "5 %")
    real_pass = True
    if results_real:
        real_pass = _summarise(results_real, "CLASSIFICATION — REAL-WORLD",
                               "accuracy gap", "5 %")
    _summarise(all_results, "CLASSIFICATION — COMBINED", "accuracy gap", "5 %")
    return syn_pass and real_pass


def test_regression(n_synthetic=36, n_openml=30):
    print("\n" + "=" * 70)
    print("  REGRESSION TEST — synthetic + real-world")
    print("=" * 70)

    ppo = _load_ppo(REG_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results_syn = []
    results_real = []
    rng = np.random.RandomState(99)

    # ── A. Synthetic datasets ────────────────────────────────────────────────
    print(f"\n--- Synthetic datasets ({n_synthetic}) ---")
    for i in range(n_synthetic):
        X, y, desc = _make_diverse_reg_dataset(i, rng)
        print(f"\n[Syn {i+1}]  {X.shape[0]}s | {X.shape[1]}f | {desc}")
        r = _run_test(ppo, X, y, desc, REGRESSION_MODELS, REG_NAMES,
                      'regression', 0.10)
        if r:
            results_syn.append(r)

    # ── B. OpenML real-world datasets ────────────────────────────────────────
    if HAS_OPENML:
        print(f"\n--- OpenML held-out real-world datasets (up to {n_openml}) ---")
        print("[*] Downloading held-out datasets (never seen during training)...")
        openml_ds = _load_openml_held_out(OPENML_REG_IDS, 'regression',
                                           max_datasets=n_openml)
        for idx, (name, X, y) in enumerate(openml_ds):
            print(f"\n[Real {idx+1}]  {X.shape[0]}s | {X.shape[1]}f | {name}")
            r = _run_test(ppo, X, y, name, REGRESSION_MODELS, REG_NAMES,
                          'regression', 0.10)
            if r:
                results_real.append(r)
    else:
        print("\n[SKIP] OpenML not installed — skipping real-world tests")

    # ── Combined summary ─────────────────────────────────────────────────────
    all_results = results_syn + results_real
    syn_pass  = _summarise(results_syn,  "REGRESSION — SYNTHETIC",
                           "R² gap", "10 %")
    real_pass = True
    if results_real:
        real_pass = _summarise(results_real, "REGRESSION — REAL-WORLD",
                               "R² gap", "10 %")
    _summarise(all_results, "REGRESSION — COMBINED", "R² gap", "10 %")
    return syn_pass and real_pass


# =============================================================================
# SUMMARY HELPER
# =============================================================================
def _summarise(results, label, gap_label, close_thresh):
    if not results:
        print(f"\n[FAIL] No tests completed")
        return False

    n       = len(results)
    n_best  = sum(1 for r in results if r['is_best'])
    n_close = sum(1 for r in results if r['is_close'])
    avg_gap = float(np.mean([r['gap'] for r in results]))

    # Model diversity
    from collections import Counter
    picks = Counter(r['rl_choice'] for r in results)

    print(f"\n{'='*70}")
    print(f"  {label} SUMMARY")
    print(f"{'='*70}")
    print(f"   Exact best    : {n_best}/{n}  ({100*n_best/n:.0f} %)")
    print(f"   Within {close_thresh} : {n_close}/{n}  ({100*n_close/n:.0f} %)")
    print(f"   Avg {gap_label}  : {avg_gap:.4f}")
    print(f"   Model diversity:")
    for m, cnt in picks.most_common():
        print(f"      {m:<35} x{cnt}")

    n_unique = len(picks)
    passed = n_close >= n * 0.5 and n_unique >= 3   # must pick ≥3 different models
    grade  = "PASS" if passed else "NEEDS MORE TRAINING"
    print(f"   Unique models picked: {n_unique}")
    print(f"   Verdict       : [{grade}]")
    return passed


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if not HAS_SB3:
        print("[ERROR] stable-baselines3 not installed")
        sys.exit(1)

    np.random.seed(7)     # reproducible test runs

    print("\n" + "=" * 70)
    print("  RL MODEL SELECTOR — EXTENDED VALIDATION TEST")
    print("  Synthetic: 32 clf + 36 reg  |  Real-world: 30 clf + 30 reg")
    print("=" * 70)

    clf_pass = test_classification(n_synthetic=32, n_openml=30)
    reg_pass = test_regression(n_synthetic=36, n_openml=30)

    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    if clf_pass and reg_pass:
        print("  [PASS] RL selector works well on both synthetic AND real-world data.")
    elif clf_pass or reg_pass:
        print("  [PARTIAL] One task type is working; other needs more training.")
        print("  Tip: increase total_timesteps in train_rl_model_selector.py")
    else:
        print("  [NEEDS WORK] Run training with more timesteps / OpenML data.")
        print("  Tip: python train_rl_model_selector.py --timesteps 500000")
