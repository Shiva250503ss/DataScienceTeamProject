"""
Fresh Evaluation (Extended)
============================
Tests the trained RL agent on data it has NEVER seen:
  A. 25 synthetic classification + 25 synthetic regression (unique seeds & structures)
  B. 20 real-world OpenML classification + 20 regression (held-out dataset IDs)

All seeds, dataset types, and OpenML IDs are distinct from training AND test_rl_selector.py.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.datasets import (make_classification, make_regression,
                               make_moons, make_circles, make_blobs,
                               make_swiss_roll, make_s_curve,
                               make_friedman1, make_friedman2, make_friedman3)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

from train_rl_model_selector import (
    extract_meta_features, CLASSIFICATION_MODELS, REGRESSION_MODELS
)
from stable_baselines3 import PPO

try:
    import openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False
    print("[WARNING] openml not installed — real-world evaluation will be skipped.")

# ── Load models ───────────────────────────────────────────────────────────────
clf_ppo = PPO.load('rl_model_selector_classification.pkl')
reg_ppo = PPO.load('rl_model_selector_regression.pkl')
print('[OK] Models loaded\n')

CLF_NAMES = list(CLASSIFICATION_MODELS.keys())
REG_NAMES = list(REGRESSION_MODELS.keys())


def eval_all(X, y, model_dict, scoring):
    cv = KFold(n_splits=3, shuffle=True, random_state=7)
    Xs = StandardScaler().fit_transform(X)
    scores = {}
    for name, model in model_dict.items():
        try:
            s = cross_val_score(model, Xs, y, cv=cv, scoring=scoring, error_score=0.0)
            scores[name] = float(np.clip(np.mean(s), 0.0, 1.0))
        except Exception:
            scores[name] = 0.0
    return scores


# =============================================================================
# OPENML HELD-OUT LOADER  (IDs different from training AND test_rl_selector.py)
# =============================================================================

# Classification IDs — hand-picked, NOT in training pool or test_rl_selector.py
FRESH_CLF_IDS = [
    1466,   # cardiotocography (2126s, 21f, 10 classes)
    1489,   # phoneme (5404s, 5f)
    1494,   # qsar-biodeg (1055s, 41f)
    1504,   # steel-plates-fault (1941s, 27f)
    4538,   # GesturePhaseSegmentation (9873s, 32f, 5 classes)
    40496,  # LED-display-domain-7digit (500s, 7f, 10 classes)
    40668,  # connect-4 (67557s, 42f, 3 classes)
    40685,  # shuttle (58000s, 9f, 7 classes)
    40900,  # Satellite (6435s, 36f, 6 classes)
    40927,  # CIFAR_10_small (10000s, 3072f, 10 classes)
    40979,  # mfeat-factors (2000s, 216f, 10 classes)
    40994,  # climate-model-simulation (540s, 18f)
    41138,  # APSFailure (76000s, 170f)
    1053,   # jm1 (10885s, 21f — software defect)
    1038,   # gina_agnostic (3468s, 970f)
    40978,  # optdigits (5620s, 64f, 10 classes)
    40981,  # Australian (690s, 6f)
    1043,   # ada_agnostic (4562s, 48f)
    1120,   # MagicTelescope (19020s, 10f)
    1169,   # airlines (539383s, 7f)
]

FRESH_REG_IDS = [
    42225,  # diamonds (53940s, 6f)
    42570,  # superconduct (21263s, 81f)
    505,    # tecator (240s, 124f — food science)
    534,    # analcatdata_apnea3 (104s, 3f)
    529,    # analcatdata_election2000 (67s, 14f)
    556,    # analcatdata_vehicle (48s, 3f)
    42183,  # kin8nm (8192s, 8f)
    42571,  # abalone (4177s, 7f)
    42207,  # cpu_act (8192s, 21f)
    42563,  # nyc-taxi-green-dec-2016 (581835s, 16f)
    216,    # elevators (16599s, 18f)
    218,    # house_8L (22784s, 8f)
    537,    # houses (20640s, 8f)
    215,    # 2dplanes (40768s, 10f)
    42688,  # Mercedes_Benz_Greener (4209s, 359f)
    42092,  # Bike_Sharing (731s, 13f)
    574,    # house_16H (22784s, 16f)
    42726,  # SatelliteTemperature (1066s, 23f)
    41540,  # black_friday (166821s, 5f)
    550,    # quake (2178s, 3f)
]


def _download_openml(did, task_type, max_samples=5000):
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
            idx = np.random.default_rng(77).choice(len(X_arr), max_samples, replace=False)
            X_arr, y_arr = X_arr[idx], y_arr[idx]

        return (ds.name, X_arr, y_arr)
    except Exception as e:
        print(f'   [SKIP] id={did}: {str(e)[:80]}')
        return None


# =============================================================================
# FRESH CLASSIFICATION DATASETS  (seed range 5000+, 25 datasets)
# =============================================================================
def make_fresh_clf(idx, rng):
    seed = 5000 + idx * 37
    kind = idx % 25
    if kind == 0:
        X, t = make_swiss_roll(n_samples=500, noise=0.3, random_state=seed)
        y = (t > np.median(t)).astype(int)
        desc = 'Swiss Roll 2-class'
    elif kind == 1:
        X, y = make_classification(n_samples=600, n_features=64, n_informative=30,
                                   n_redundant=10, n_classes=10,
                                   n_clusters_per_class=1, random_state=seed)
        desc = 'High-dim 64f 10-class'
    elif kind == 2:
        X = rng.binomial(1, 0.15, size=(400, 50)).astype(np.float32)
        y = (X[:, :10].sum(axis=1) > 2).astype(int)
        desc = 'Sparse binary 50f'
    elif kind == 3:
        X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                                   weights=[0.9, 0.1], flip_y=0.02, random_state=seed)
        desc = 'Imbalanced 9:1'
    elif kind == 4:
        X, y = make_classification(n_samples=700, n_features=15, n_informative=10,
                                   n_classes=5, n_clusters_per_class=1, random_state=seed)
        desc = '5-class 15f'
    elif kind == 5:
        n = 500
        X = rng.randn(n, 5).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int)
        desc = '3D XOR 5f'
    elif kind == 6:
        X, y = make_classification(n_samples=450, n_features=4, n_informative=4,
                                   n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                   class_sep=2.5, flip_y=0.0, random_state=seed)
        desc = 'Linear 3-class 4f'
    elif kind == 7:
        X, y = make_classification(n_samples=3000, n_features=10, n_informative=7,
                                   n_redundant=2, flip_y=0.05, random_state=seed)
        desc = 'Large 3000s 10f'
    elif kind == 8:
        X, y = make_circles(n_samples=250, noise=0.08, factor=0.4, random_state=seed)
        desc = 'Small circles 250s'
    elif kind == 9:
        n = 600
        X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
        desc = 'Checkerboard 2D'
    elif kind == 10:
        X, y = make_classification(n_samples=500, n_features=20, n_informative=5,
                                   n_redundant=5, flip_y=0.30, random_state=seed)
        desc = 'Very noisy 30% flip'
    elif kind == 11:
        X, y = make_blobs(n_samples=120, n_features=3, centers=4,
                          cluster_std=0.5, random_state=seed)
        y = y.astype(int)
        desc = 'Tiny 120s 4-class'
    elif kind == 12:
        X, y = make_classification(n_samples=400, n_features=100, n_informative=8,
                                   n_redundant=15, flip_y=0.05, random_state=seed)
        desc = 'Wide sparse 100f'
    elif kind == 13:
        X, y = make_blobs(n_samples=800, n_features=3, centers=6,
                          cluster_std=0.3, random_state=seed)
        y = y.astype(int)
        desc = '3D 6-cluster blobs'
    elif kind == 14:
        X, y = make_classification(n_samples=5000, n_features=20, n_informative=12,
                                   n_redundant=4, flip_y=0.06, random_state=seed)
        desc = 'Very large 5000s 20f'
    # ── NEW kinds (15-24) ────────────────────────────────────────────────────
    elif kind == 15:
        # Concentric rings with 3 classes
        X1, _ = make_circles(n_samples=300, noise=0.04, factor=0.3, random_state=seed)
        X2, _ = make_circles(n_samples=300, noise=0.04, factor=0.7, random_state=seed+1)
        X = np.vstack([X1[_==0], X1[_==1], X2[_==1]])
        y = np.array([0]*len(X1[_==0]) + [1]*len(X1[_==1]) + [2]*len(X2[_==1]))
        desc = 'Concentric rings 3-class'
    elif kind == 16:
        # Extremely imbalanced 99:1
        X, y = make_classification(n_samples=2000, n_features=12, n_informative=8,
                                   weights=[0.99, 0.01], flip_y=0.0, random_state=seed)
        desc = 'Extreme imbalance 99:1'
    elif kind == 17:
        # S-curve binarized
        X, t = make_s_curve(n_samples=600, noise=0.2, random_state=seed)
        y = (t > np.median(t)).astype(int)
        desc = 'S-curve 2-class'
    elif kind == 18:
        # Overlapping moons (high noise)
        X, y = make_moons(n_samples=800, noise=0.35, random_state=seed)
        desc = 'Noisy moons n=0.35'
    elif kind == 19:
        # 8-class blobs varying density
        centers = rng.randn(8, 4).astype(np.float32) * 3
        X, y = make_blobs(n_samples=1200, n_features=4, centers=centers,
                          cluster_std=[0.2, 0.5, 1.0, 0.3, 0.8, 0.4, 0.6, 0.9],
                          random_state=seed)
        y = y.astype(int)
        desc = '8-class varying density'
    elif kind == 20:
        # Feature-rich low-sample (p >> n)
        X, y = make_classification(n_samples=80, n_features=300,
                                   n_informative=20, n_redundant=50,
                                   random_state=seed)
        desc = 'p>>n: 300f 80s'
    elif kind == 21:
        # Spiral pattern (2D)
        n = 400
        theta = np.linspace(0, 4*np.pi, n) + rng.randn(n)*0.2
        r = np.linspace(0.5, 3, n)
        X0 = np.column_stack([r*np.cos(theta), r*np.sin(theta)]).astype(np.float32)
        X1 = np.column_stack([r*np.cos(theta+np.pi), r*np.sin(theta+np.pi)]).astype(np.float32)
        X = np.vstack([X0, X1])
        y = np.array([0]*n + [1]*n)
        desc = 'Spiral pattern 2D'
    elif kind == 22:
        # Uniform noise baseline (should be ~50% accuracy for all)
        n = 500
        X = rng.uniform(-1, 1, (n, 10)).astype(np.float32)
        y = rng.randint(0, 2, n)
        desc = 'Pure noise baseline'
    elif kind == 23:
        # 12-class many-feature
        X, y = make_classification(n_samples=1500, n_features=40, n_informative=25,
                                   n_redundant=10, n_classes=12,
                                   n_clusters_per_class=1, random_state=seed)
        desc = '12-class 40f'
    else:
        # Mixed: half moons + half blobs
        Xm, ym = make_moons(n_samples=400, noise=0.1, random_state=seed)
        Xb, yb = make_blobs(n_samples=400, n_features=2, centers=2,
                            cluster_std=0.5, random_state=seed+1)
        yb = yb + 2  # shift to classes 2,3
        X = np.vstack([Xm, Xb])
        y = np.concatenate([ym, yb])
        desc = 'Mixed moons+blobs 4-class'
    return np.array(X, dtype=np.float32), np.array(y), desc


# =============================================================================
# FRESH REGRESSION DATASETS  (seed range 7000+, 25 datasets)
# =============================================================================
def make_fresh_reg(idx, rng):
    seed = 7000 + idx * 41
    kind = idx % 25
    if kind == 0:
        X, y = make_regression(n_samples=600, n_features=12, n_informative=12,
                               noise=0.1, random_state=seed)
        desc = 'Linear 12f 600s'
    elif kind == 1:
        X, y = make_regression(n_samples=300, n_features=80, n_informative=2,
                               noise=0.2, random_state=seed)
        desc = 'Ultra-sparse 2/80f'
    elif kind == 2:
        n = 500
        X = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
        y = (np.sin(X[:, 0] * 1.5) + 0.5 * np.cos(X[:, 1] * 2.5) +
             rng.randn(n) * 0.05).astype(np.float32)
        desc = 'Wave sin+cos 3f'
    elif kind == 3:
        n = 600
        X = rng.randn(n, 4).astype(np.float32)
        y = (np.sqrt((X ** 2).sum(axis=1)) + rng.randn(n) * 0.1).astype(np.float32)
        desc = 'Radial distance 4f'
    elif kind == 4:
        n = 700
        X = rng.randn(n, 5).astype(np.float32)
        y = (np.floor(X[:, 0] * 2) / 2 + np.floor(X[:, 1] * 2) / 2 +
             rng.randn(n) * 0.05).astype(np.float32)
        desc = 'Quantized step 5f'
    elif kind == 5:
        X, y = make_regression(n_samples=500, n_features=25, n_informative=8,
                               noise=2.0, effective_rank=5, tail_strength=0.8,
                               random_state=seed)
        desc = 'Block-correlated 25f'
    elif kind == 6:
        n = 600
        X = rng.uniform(0, 3, (n, 2)).astype(np.float32)
        y = (np.exp(X[:, 0]) - np.exp(X[:, 1]) + rng.randn(n) * 0.2).astype(np.float32)
        desc = 'Exponential 2f'
    elif kind == 7:
        X, y = make_regression(n_samples=4000, n_features=15, n_informative=15,
                               noise=0.5, random_state=seed)
        desc = 'Large linear 4000s'
    elif kind == 8:
        n = 500
        X = rng.uniform(0.1, 3, (n, 4)).astype(np.float32)
        y = (np.log(X[:, 0]) * X[:, 1] + rng.randn(n) * 0.1).astype(np.float32)
        desc = 'Log-product 4f'
    elif kind == 9:
        n = 800
        X = rng.uniform(-2, 2, (n, 5)).astype(np.float32)
        y = (X[:, 0] ** 3 - 2 * X[:, 1] ** 2 + X[:, 2] * X[:, 3] +
             rng.randn(n) * 0.3).astype(np.float32)
        desc = 'Polynomial x^3 5f'
    elif kind == 10:
        X, y = make_regression(n_samples=80, n_features=4, n_informative=4,
                               noise=0.5, random_state=seed)
        desc = 'Tiny 80s 4f'
    elif kind == 11:
        X, y = make_regression(n_samples=600, n_features=10, n_informative=5,
                               noise=20.0, random_state=seed)
        desc = 'High-noise sigma=20'
    elif kind == 12:
        n = 1000
        X = rng.randn(n, 200).astype(np.float32)
        y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 + rng.randn(n) * 0.5).astype(np.float32)
        desc = 'Very high-dim 200f'
    elif kind == 13:
        n = 500
        X = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)
        y = (np.sin(X[:, 0]) * np.sin(X[:, 1] * 2) + rng.randn(n) * 0.03).astype(np.float32)
        desc = 'Periodic sin*sin 2f'
    elif kind == 14:
        X, t = make_s_curve(n_samples=600, noise=0.2, random_state=seed)
        y = t.astype(np.float32)
        desc = 'S-Curve 3D'
    # ── NEW kinds (15-24) ────────────────────────────────────────────────────
    elif kind == 15:
        X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.5,
                              random_state=seed)
        desc = 'Friedman1 10f 1000s'
    elif kind == 16:
        X, y = make_friedman2(n_samples=800, noise=1.0, random_state=seed)
        desc = 'Friedman2 physics'
    elif kind == 17:
        X, y = make_friedman3(n_samples=800, noise=0.5, random_state=seed)
        desc = 'Friedman3 arctan'
    elif kind == 18:
        # Absolute value (V-shape — trees should win)
        n = 700
        X = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
        y = (np.abs(X[:, 0]) + np.abs(X[:, 1]) - X[:, 2]**2 +
             rng.randn(n) * 0.2).astype(np.float32)
        desc = 'Abs value V-shape 3f'
    elif kind == 19:
        # Heteroscedastic (variance depends on X)
        n = 800
        X = rng.uniform(0, 5, (n, 3)).astype(np.float32)
        noise = rng.randn(n).astype(np.float32) * (0.1 + X[:, 0] * 0.3)
        y = (2 * X[:, 0] + X[:, 1]**2 + noise).astype(np.float32)
        desc = 'Heteroscedastic 3f'
    elif kind == 20:
        # Max function (piecewise)
        n = 600
        X = rng.randn(n, 4).astype(np.float32)
        y = (np.maximum(X[:, 0], X[:, 1]) + np.minimum(X[:, 2], X[:, 3]) +
             rng.randn(n) * 0.1).astype(np.float32)
        desc = 'Max/Min piecewise 4f'
    elif kind == 21:
        # Very large dataset
        X, y = make_regression(n_samples=8000, n_features=20, n_informative=15,
                               noise=1.0, random_state=seed)
        desc = 'Very large 8000s 20f'
    elif kind == 22:
        # Swiss roll regression
        X, t = make_swiss_roll(n_samples=800, noise=0.3, random_state=seed)
        y = t.astype(np.float32)
        desc = 'Swiss roll regression'
    elif kind == 23:
        # Multiplicative interactions only
        n = 700
        X = rng.uniform(0.5, 3, (n, 6)).astype(np.float32)
        y = (X[:, 0] * X[:, 1] * X[:, 2] / (X[:, 3] + X[:, 4]) +
             rng.randn(n) * 0.2).astype(np.float32)
        desc = 'Multiplicative 6f'
    else:
        # Pure noise baseline (R2 should be ~0 for all)
        n = 500
        X = rng.randn(n, 10).astype(np.float32)
        y = rng.randn(n).astype(np.float32)
        desc = 'Pure noise baseline'
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), desc


# =============================================================================
# RUN EVALUATION
# =============================================================================
rng = np.random.RandomState(2025)

# ── A. Synthetic Classification ──────────────────────────────────────────────
N_SYN_CLF = 25
print('=' * 95)
print(f'  FRESH SYNTHETIC CLASSIFICATION  ({N_SYN_CLF} datasets, never seen in training/testing)')
print('=' * 95)
print(f'  {"#":<4} {"Dataset Type":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
print('-' * 95)

clf_syn_results = []
for i in range(N_SYN_CLF):
    X, y, desc = make_fresh_clf(i, rng)
    feats = extract_meta_features(X, y, 'classification')
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    action, _ = clf_ppo.predict(feats, deterministic=True)
    rl_choice = CLF_NAMES[int(action)]
    scores = eval_all(X, y, CLASSIFICATION_MODELS, 'accuracy')
    best = max(scores, key=scores.get)
    gap = scores[best] - scores[rl_choice]
    status = '[BEST]' if rl_choice == best else ('[CLOSE]' if gap < 0.05 else '[MISS]')
    print(f'  {i+1:<4} {desc:<26} {rl_choice:<30} {best:<30} {gap:>6.4f}  {status}')
    clf_syn_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.05})

# ── B. OpenML Classification ────────────────────────────────────────────────
clf_real_results = []
if HAS_OPENML:
    N_REAL_CLF = 20
    print()
    print('=' * 95)
    print(f'  FRESH OPENML CLASSIFICATION  ({N_REAL_CLF} real-world datasets, held out)')
    print('=' * 95)
    print(f'  {"#":<4} {"Dataset Name":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
    print('-' * 95)

    loaded = 0
    for did in FRESH_CLF_IDS:
        if loaded >= N_REAL_CLF:
            break
        result = _download_openml(did, 'classification')
        if result is None:
            continue
        name, X, y = result
        loaded += 1
        feats = extract_meta_features(X, y, 'classification')
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
        action, _ = clf_ppo.predict(feats, deterministic=True)
        rl_choice = CLF_NAMES[int(action)]
        scores = eval_all(X, y, CLASSIFICATION_MODELS, 'accuracy')
        best = max(scores, key=scores.get)
        gap = scores[best] - scores[rl_choice]
        status = '[BEST]' if rl_choice == best else ('[CLOSE]' if gap < 0.05 else '[MISS]')
        print(f'  {loaded:<4} {name[:26]:<26} {rl_choice:<30} {best:<30} {gap:>6.4f}  {status}')
        clf_real_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.05})

# ── Classification Summary ──────────────────────────────────────────────────
print()
picks = Counter(r['rl'] for r in clf_syn_results)
n_ok = sum(1 for r in clf_syn_results if r['ok'])
print(f'  SYNTHETIC CLF: Within 5% accuracy : {n_ok}/{N_SYN_CLF}  ({100*n_ok/N_SYN_CLF:.0f}%)')
print(f'    Unique models: {len(picks)}')
for m, c in picks.most_common():
    print(f'      {m:<35} x{c}')

if clf_real_results:
    picks_r = Counter(r['rl'] for r in clf_real_results)
    n_ok_r = sum(1 for r in clf_real_results if r['ok'])
    print(f'  REAL-WORLD CLF: Within 5% accuracy : {n_ok_r}/{len(clf_real_results)}  '
          f'({100*n_ok_r/len(clf_real_results):.0f}%)')
    print(f'    Unique models: {len(picks_r)}')
    for m, c in picks_r.most_common():
        print(f'      {m:<35} x{c}')

# ── A. Synthetic Regression ──────────────────────────────────────────────────
N_SYN_REG = 25
print()
print('=' * 95)
print(f'  FRESH SYNTHETIC REGRESSION  ({N_SYN_REG} datasets, never seen in training/testing)')
print('=' * 95)
print(f'  {"#":<4} {"Dataset Type":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
print('-' * 95)

reg_syn_results = []
for i in range(N_SYN_REG):
    X, y, desc = make_fresh_reg(i, rng)
    feats = extract_meta_features(X, y, 'regression')
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    action, _ = reg_ppo.predict(feats, deterministic=True)
    rl_choice = REG_NAMES[int(action)]
    scores = eval_all(X, y, REGRESSION_MODELS, 'r2')
    best = max(scores, key=scores.get)
    gap = scores[best] - scores[rl_choice]
    status = '[BEST]' if rl_choice == best else ('[CLOSE]' if gap < 0.10 else '[MISS]')
    print(f'  {i+1:<4} {desc:<26} {rl_choice:<30} {best:<30} {gap:>6.4f}  {status}')
    reg_syn_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.10})

# ── B. OpenML Regression ────────────────────────────────────────────────────
reg_real_results = []
if HAS_OPENML:
    N_REAL_REG = 20
    print()
    print('=' * 95)
    print(f'  FRESH OPENML REGRESSION  ({N_REAL_REG} real-world datasets, held out)')
    print('=' * 95)
    print(f'  {"#":<4} {"Dataset Name":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
    print('-' * 95)

    loaded = 0
    for did in FRESH_REG_IDS:
        if loaded >= N_REAL_REG:
            break
        result = _download_openml(did, 'regression')
        if result is None:
            continue
        name, X, y = result
        loaded += 1
        feats = extract_meta_features(X, y, 'regression')
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
        action, _ = reg_ppo.predict(feats, deterministic=True)
        rl_choice = REG_NAMES[int(action)]
        scores = eval_all(X, y, REGRESSION_MODELS, 'r2')
        best = max(scores, key=scores.get)
        gap = scores[best] - scores[rl_choice]
        status = '[BEST]' if rl_choice == best else ('[CLOSE]' if gap < 0.10 else '[MISS]')
        print(f'  {loaded:<4} {name[:26]:<26} {rl_choice:<30} {best:<30} {gap:>6.4f}  {status}')
        reg_real_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.10})

# ── Regression Summary ──────────────────────────────────────────────────────
print()
picks = Counter(r['rl'] for r in reg_syn_results)
n_ok = sum(1 for r in reg_syn_results if r['ok'])
print(f'  SYNTHETIC REG: Within 10% R2 gap : {n_ok}/{N_SYN_REG}  ({100*n_ok/N_SYN_REG:.0f}%)')
print(f'    Unique models: {len(picks)}')
for m, c in picks.most_common():
    print(f'      {m:<35} x{c}')

if reg_real_results:
    picks_r = Counter(r['rl'] for r in reg_real_results)
    n_ok_r = sum(1 for r in reg_real_results if r['ok'])
    print(f'  REAL-WORLD REG: Within 10% R2 gap : {n_ok_r}/{len(reg_real_results)}  '
          f'({100*n_ok_r/len(reg_real_results):.0f}%)')
    print(f'    Unique models: {len(picks_r)}')
    for m, c in picks_r.most_common():
        print(f'      {m:<35} x{c}')


# ── Final verdict ─────────────────────────────────────────────────────────────
all_clf = clf_syn_results + clf_real_results
all_reg = reg_syn_results + reg_real_results

clf_ok  = sum(1 for r in all_clf if r['ok'])
reg_ok  = sum(1 for r in all_reg if r['ok'])
clf_div = len(Counter(r['rl'] for r in all_clf))
reg_div = len(Counter(r['rl'] for r in all_reg))

print()
print('=' * 95)
print('  FINAL VERDICT')
print('=' * 95)
print(f'  Total Classification : {clf_ok}/{len(all_clf)} within 5%  '
      f'({100*clf_ok/max(len(all_clf),1):.0f}%),  {clf_div} unique models')
print(f'  Total Regression     : {reg_ok}/{len(all_reg)} within 10% '
      f'({100*reg_ok/max(len(all_reg),1):.0f}%),  {reg_div} unique models')

clf_pass = clf_ok >= len(all_clf) * 0.6 and clf_div >= 3
reg_pass = reg_ok >= len(all_reg) * 0.6 and reg_div >= 3
overall = 'PASS' if (clf_pass and reg_pass) else 'NEEDS WORK'
print(f'\n  Classification : {"PASS" if clf_pass else "NEEDS WORK"}')
print(f'  Regression     : {"PASS" if reg_pass else "NEEDS WORK"}')
print(f'  Overall        : {overall}')
