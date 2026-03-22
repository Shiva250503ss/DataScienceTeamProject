"""
Fresh Evaluation — 15 completely unseen datasets per task.
Uses random seeds, dataset types, and structures never seen in training or testing.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.datasets import (make_classification, make_regression,
                               make_moons, make_circles, make_blobs,
                               make_swiss_roll, make_s_curve)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter

from train_rl_model_selector import (
    extract_meta_features, CLASSIFICATION_MODELS, REGRESSION_MODELS
)
from stable_baselines3 import PPO

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
# FRESH CLASSIFICATION DATASETS  (seed range 5000+, never used before)
# =============================================================================
def make_fresh_clf(idx, rng):
    seed = 5000 + idx * 37
    kind = idx % 15
    if kind == 0:
        # Swiss roll binarized by median angle
        X, t = make_swiss_roll(n_samples=500, noise=0.3, random_state=seed)
        y = (t > np.median(t)).astype(int)
        desc = 'Swiss Roll 2-class'
    elif kind == 1:
        # High-dim pixel-like: 64 features, 10 classes
        X, y = make_classification(n_samples=600, n_features=64, n_informative=30,
                                   n_redundant=10, n_classes=10,
                                   n_clusters_per_class=1, random_state=seed)
        desc = 'High-dim 64f 10-class'
    elif kind == 2:
        # Sparse binary features
        X = rng.binomial(1, 0.15, size=(400, 50)).astype(np.float32)
        y = (X[:, :10].sum(axis=1) > 2).astype(int)
        desc = 'Sparse binary 50f'
    elif kind == 3:
        # Heavily imbalanced 9:1
        X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                                   weights=[0.9, 0.1], flip_y=0.02, random_state=seed)
        desc = 'Imbalanced 9:1'
    elif kind == 4:
        # 5-class multiclass
        X, y = make_classification(n_samples=700, n_features=15, n_informative=10,
                                   n_classes=5, n_clusters_per_class=1, random_state=seed)
        desc = '5-class 15f'
    elif kind == 5:
        # 3D XOR with noise
        n = 500
        X = rng.randn(n, 5).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int)
        desc = '3D XOR 5f'
    elif kind == 6:
        # Perfect linear 3-class
        X, y = make_classification(n_samples=450, n_features=4, n_informative=4,
                                   n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                   class_sep=2.5, flip_y=0.0, random_state=seed)
        desc = 'Linear 3-class 4f'
    elif kind == 7:
        # Large balanced dataset
        X, y = make_classification(n_samples=3000, n_features=10, n_informative=7,
                                   n_redundant=2, flip_y=0.05, random_state=seed)
        desc = 'Large 3000s 10f'
    elif kind == 8:
        # Small non-linear circles
        X, y = make_circles(n_samples=250, noise=0.08, factor=0.4, random_state=seed)
        desc = 'Small circles 250s'
    elif kind == 9:
        # Checkerboard 2D — axis-aligned decision boundary
        n = 600
        X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
        desc = 'Checkerboard 2D'
    elif kind == 10:
        # Very high noise (30% label flip)
        X, y = make_classification(n_samples=500, n_features=20, n_informative=5,
                                   n_redundant=5, flip_y=0.30, random_state=seed)
        desc = 'Very noisy 30% flip'
    elif kind == 11:
        # Tiny 4-class blobs
        X, y = make_blobs(n_samples=120, n_features=3, centers=4,
                          cluster_std=0.5, random_state=seed)
        y = y.astype(int)
        desc = 'Tiny 120s 4-class'
    elif kind == 12:
        # Wide sparse 100 features
        X, y = make_classification(n_samples=400, n_features=100, n_informative=8,
                                   n_redundant=15, flip_y=0.05, random_state=seed)
        desc = 'Wide sparse 100f'
    elif kind == 13:
        # 3D tight 6-cluster blobs
        X, y = make_blobs(n_samples=800, n_features=3, centers=6,
                          cluster_std=0.3, random_state=seed)
        y = y.astype(int)
        desc = '3D 6-cluster blobs'
    else:
        # Very large dataset
        X, y = make_classification(n_samples=5000, n_features=20, n_informative=12,
                                   n_redundant=4, flip_y=0.06, random_state=seed)
        desc = 'Very large 5000s 20f'
    return np.array(X, dtype=np.float32), np.array(y), desc


# =============================================================================
# FRESH REGRESSION DATASETS  (seed range 7000+, never used before)
# =============================================================================
def make_fresh_reg(idx, rng):
    seed = 7000 + idx * 41
    kind = idx % 15
    if kind == 0:
        # Pure linear many features
        X, y = make_regression(n_samples=600, n_features=12, n_informative=12,
                               noise=0.1, random_state=seed)
        desc = 'Linear 12f 600s'
    elif kind == 1:
        # Ultra-sparse: only 2 of 80 features matter
        X, y = make_regression(n_samples=300, n_features=80, n_informative=2,
                               noise=0.2, random_state=seed)
        desc = 'Ultra-sparse 2/80f'
    elif kind == 2:
        # Wave / oscillation target
        n = 500
        X = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
        y = (np.sin(X[:, 0] * 1.5) + 0.5 * np.cos(X[:, 1] * 2.5) +
             rng.randn(n) * 0.05).astype(np.float32)
        desc = 'Wave sin+cos 3f'
    elif kind == 3:
        # Radial (Euclidean distance from origin)
        n = 600
        X = rng.randn(n, 4).astype(np.float32)
        y = (np.sqrt((X ** 2).sum(axis=1)) + rng.randn(n) * 0.1).astype(np.float32)
        desc = 'Radial distance 4f'
    elif kind == 4:
        # Quantized / step function via floor
        n = 700
        X = rng.randn(n, 5).astype(np.float32)
        y = (np.floor(X[:, 0] * 2) / 2 + np.floor(X[:, 1] * 2) / 2 +
             rng.randn(n) * 0.05).astype(np.float32)
        desc = 'Quantized step 5f'
    elif kind == 5:
        # Correlated block structure
        X, y = make_regression(n_samples=500, n_features=25, n_informative=8,
                               noise=2.0, effective_rank=5, tail_strength=0.8,
                               random_state=seed)
        desc = 'Block-correlated 25f'
    elif kind == 6:
        # Exponential relationship
        n = 600
        X = rng.uniform(0, 3, (n, 2)).astype(np.float32)
        y = (np.exp(X[:, 0]) - np.exp(X[:, 1]) + rng.randn(n) * 0.2).astype(np.float32)
        desc = 'Exponential 2f'
    elif kind == 7:
        # Very large linear
        X, y = make_regression(n_samples=4000, n_features=15, n_informative=15,
                               noise=0.5, random_state=seed)
        desc = 'Large linear 4000s'
    elif kind == 8:
        # Log-product (heteroscedastic)
        n = 500
        X = rng.uniform(0.1, 3, (n, 4)).astype(np.float32)
        y = (np.log(X[:, 0]) * X[:, 1] + rng.randn(n) * 0.1).astype(np.float32)
        desc = 'Log-product 4f'
    elif kind == 9:
        # Cubic polynomial with interactions
        n = 800
        X = rng.uniform(-2, 2, (n, 5)).astype(np.float32)
        y = (X[:, 0] ** 3 - 2 * X[:, 1] ** 2 + X[:, 2] * X[:, 3] +
             rng.randn(n) * 0.3).astype(np.float32)
        desc = 'Polynomial x^3 5f'
    elif kind == 10:
        # Tiny dataset
        X, y = make_regression(n_samples=80, n_features=4, n_informative=4,
                               noise=0.5, random_state=seed)
        desc = 'Tiny 80s 4f'
    elif kind == 11:
        # Very high noise
        X, y = make_regression(n_samples=600, n_features=10, n_informative=5,
                               noise=20.0, random_state=seed)
        desc = 'High-noise sigma=20'
    elif kind == 12:
        # Very high-dim 200 features
        n = 1000
        X = rng.randn(n, 200).astype(np.float32)
        y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 + rng.randn(n) * 0.5).astype(np.float32)
        desc = 'Very high-dim 200f'
    elif kind == 13:
        # Near-periodic signal
        n = 500
        X = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)
        y = (np.sin(X[:, 0]) * np.sin(X[:, 1] * 2) + rng.randn(n) * 0.03).astype(np.float32)
        desc = 'Periodic sin*sin 2f'
    else:
        # S-curve latent variable regression
        X, t = make_s_curve(n_samples=600, noise=0.2, random_state=seed)
        y = t.astype(np.float32)
        desc = 'S-Curve 3D'
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), desc


# =============================================================================
# RUN EVALUATION
# =============================================================================
rng = np.random.RandomState(2025)

# ── Classification ────────────────────────────────────────────────────────────
print('=' * 90)
print('  FRESH UNSEEN CLASSIFICATION DATASETS  (15 datasets, never seen in training/testing)')
print('=' * 90)
print(f'  {"#":<4} {"Dataset Type":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
print('-' * 90)

clf_results = []
for i in range(15):
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
    clf_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.05})

picks = Counter(r['rl'] for r in clf_results)
n_ok = sum(1 for r in clf_results if r['ok'])
print()
print(f'  Within 5% accuracy : {n_ok}/15  ({100*n_ok/15:.0f}%)')
print(f'  Unique models used  : {len(picks)}')
for m, c in picks.most_common():
    print(f'    {m:<35} x{c}')

# ── Regression ────────────────────────────────────────────────────────────────
print()
print('=' * 90)
print('  FRESH UNSEEN REGRESSION DATASETS  (15 datasets, never seen in training/testing)')
print('=' * 90)
print(f'  {"#":<4} {"Dataset Type":<26} {"RL Pick":<30} {"True Best":<30} {"Gap":>6}  Result')
print('-' * 90)

reg_results = []
for i in range(15):
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
    reg_results.append({'rl': rl_choice, 'best': best, 'gap': gap, 'ok': gap < 0.10})

picks = Counter(r['rl'] for r in reg_results)
n_ok = sum(1 for r in reg_results if r['ok'])
print()
print(f'  Within 10% R2 gap  : {n_ok}/15  ({100*n_ok/15:.0f}%)')
print(f'  Unique models used  : {len(picks)}')
for m, c in picks.most_common():
    print(f'    {m:<35} x{c}')

# ── Final verdict ─────────────────────────────────────────────────────────────
clf_ok = sum(1 for r in clf_results if r['ok'])
reg_ok = sum(1 for r in reg_results if r['ok'])
clf_div = len(Counter(r['rl'] for r in clf_results))
reg_div = len(Counter(r['rl'] for r in reg_results))

print()
print('=' * 90)
print('  FINAL VERDICT')
print('=' * 90)
clf_pass = clf_ok >= 11 and clf_div >= 3
reg_pass = reg_ok >= 11 and reg_div >= 3
print(f'  Classification : {"PASS" if clf_pass else "NEEDS WORK"}  '
      f'({clf_ok}/15 close, {clf_div} unique models)')
print(f'  Regression     : {"PASS" if reg_pass else "NEEDS WORK"}  '
      f'({reg_ok}/15 close, {reg_div} unique models)')
