"""
RL Model Selector — Validation Test
=====================================
Tests whether the trained PPO agent (from train_rl_model_selector.py)
picks good models for unseen datasets.

Checks:
  [BEST]  — RL picked the exact best model
  [CLOSE] — RL's model is within 5 % accuracy / 10 % R² of the best
  [MISS]  — RL's model is further away

Bugs fixed vs. original:
  1. Model paths now match what train_rl_model_selector.py actually saves
  2. Feature extractor is the SAME extract_meta_features() used in training
  3. Model dicts are the SAME CLASSIFICATION_MODELS / REGRESSION_MODELS used
     in training — no action-space mismatch or hacky `% len(...)` workaround
  4. Landmarks are computed during testing (same as training)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import (make_classification, make_regression,
                               make_moons, make_circles, make_blobs)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

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
# CLASSIFICATION TEST
# =============================================================================
def _make_diverse_clf_dataset(i, rng):
    """Return (X, y, description) — each index produces a structurally different dataset."""
    kind = i % 8
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
    else:           # multi-cluster (GBM should win)
        X, y = make_classification(n_samples=800+i*80, n_features=12,
                                   n_informative=8, n_redundant=2,
                                   n_clusters_per_class=4, flip_y=0.06,
                                   class_sep=0.6, random_state=1100+i)
        desc = "multi-cluster GBM"
    return np.array(X, dtype=np.float32), np.array(y), desc


def test_classification(n_tests=16):
    print("\n" + "=" * 70)
    print("  CLASSIFICATION TEST — does RL pick good algorithms?")
    print("=" * 70)

    ppo = _load_ppo(CLF_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results = []
    rng = np.random.RandomState(42)

    for i in range(n_tests):
        X, y, desc = _make_diverse_clf_dataset(i, rng)
        n_samples, n_features = X.shape

        print(f"\n[Test {i+1}]  {n_samples}s | {n_features}f | {desc}")

        # ── Step 1: extract the SAME 32 features used during training ─────────
        try:
            features_32f = extract_meta_features(X, y, task_type='classification')
            features_32f = np.nan_to_num(features_32f, nan=0.0,
                                          posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"   [ERROR] Feature extraction failed: {e}")
            continue

        # ── Step 2: RL picks a model ──────────────────────────────────────────
        action, _ = ppo.predict(features_32f, deterministic=True)
        action     = int(action)
        # action is guaranteed in [0, len(CLF_NAMES)) — no modulo hack needed
        rl_choice  = CLF_NAMES[action]

        # ── Step 3: evaluate ALL models to find the true best ─────────────────
        scores     = _all_scores(X, y, CLASSIFICATION_MODELS, 'classification')
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        rl_score   = scores[rl_choice]

        is_best  = rl_choice == best_model
        is_close = (best_score - rl_score) < 0.05      # within 5 %

        status = "[BEST]" if is_best else ("[CLOSE]" if is_close else "[MISS]")

        print(f"   RL choice  : {rl_choice:<35}  accuracy = {rl_score:.4f}")
        print(f"   Actual best: {best_model:<35}  accuracy = {best_score:.4f}")
        print(f"   Result     : {status}   gap = {best_score - rl_score:.4f}")

        results.append({'is_best': is_best, 'is_close': is_close,
                        'gap': best_score - rl_score, 'rl_choice': rl_choice})

    return _summarise(results, "CLASSIFICATION", gap_label="accuracy gap",
                      close_thresh="5 %")


# =============================================================================
# REGRESSION TEST
# =============================================================================
def _make_diverse_reg_dataset(i, rng):
    """Return (X, y, description) — each index produces a structurally different dataset."""
    kind = i % 9
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
    else:           # interaction terms (GBM should win)
        n = 900 + i * 80
        X = rng.randn(n, 10).astype(np.float32)
        y = (X[:,0]*X[:,1] + X[:,2]**2 - X[:,3]**2 +
             np.sin(X[:,4])*X[:,5] + rng.randn(n)*0.5).astype(np.float32)
        desc = "interaction terms"
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), desc


def test_regression(n_tests=18):
    print("\n" + "=" * 70)
    print("  REGRESSION TEST — does RL pick good algorithms?")
    print("=" * 70)

    ppo = _load_ppo(REG_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results = []
    rng = np.random.RandomState(99)

    for i in range(n_tests):
        X, y, desc = _make_diverse_reg_dataset(i, rng)
        n_samples, n_features = X.shape

        print(f"\n[Test {i+1}]  {n_samples}s | {n_features}f | {desc}")

        # ── Step 1: extract the SAME 32 features used during training ─────────
        try:
            features_32f = extract_meta_features(X, y, task_type='regression')
            features_32f = np.nan_to_num(features_32f, nan=0.0,
                                          posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"   [ERROR] Feature extraction failed: {e}")
            continue

        # ── Step 2: RL picks a model ──────────────────────────────────────────
        action, _ = ppo.predict(features_32f, deterministic=True)
        action     = int(action)
        rl_choice  = REG_NAMES[action]

        # ── Step 3: evaluate ALL models to find the true best ─────────────────
        scores     = _all_scores(X, y, REGRESSION_MODELS, 'regression')
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        rl_score   = scores[rl_choice]

        is_best  = rl_choice == best_model
        is_close = (best_score - rl_score) < 0.10      # within 10 % R²

        status = "[BEST]" if is_best else ("[CLOSE]" if is_close else "[MISS]")

        print(f"   RL choice  : {rl_choice:<35}  R² = {rl_score:.4f}")
        print(f"   Actual best: {best_model:<35}  R² = {best_score:.4f}")
        print(f"   Result     : {status}   gap = {best_score - rl_score:.4f}")

        results.append({'is_best': is_best, 'is_close': is_close,
                        'gap': best_score - rl_score, 'rl_choice': rl_choice})

    return _summarise(results, "REGRESSION", gap_label="R² gap",
                      close_thresh="10 %")


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
    print("  RL MODEL SELECTOR — VALIDATION TEST")
    print("=" * 70)

    clf_pass = test_classification(n_tests=16)
    reg_pass = test_regression(n_tests=18)

    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    if clf_pass and reg_pass:
        print("  [PASS] RL selector works well for both tasks.")
    elif clf_pass or reg_pass:
        print("  [PARTIAL] One task type is working; other needs more training.")
        print("  Tip: increase total_timesteps in train_rl_model_selector.py")
    else:
        print("  [NEEDS WORK] Run training with more timesteps / OpenML data.")
        print("  Tip: python train_rl_model_selector.py --openml")
