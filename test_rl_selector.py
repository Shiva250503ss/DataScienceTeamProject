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
from sklearn.datasets import make_classification, make_regression
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
CLF_MODEL_PATH = "rl_model_selector_classification"   # PPO.save() adds .zip
REG_MODEL_PATH = "rl_model_selector_regression"

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
def test_classification(n_tests=8):
    print("\n" + "=" * 70)
    print("  CLASSIFICATION TEST — does RL pick good algorithms?")
    print("=" * 70)

    ppo = _load_ppo(CLF_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results = []

    for i in range(n_tests):
        n_samples  = np.random.randint(300, 1200)
        n_features = np.random.randint(5, 40)
        n_classes  = np.random.randint(2, 5)

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=min(2, n_features // 4),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=100 + i,          # unseen seeds (training used 42+i)
        )

        print(f"\n[Test {i+1}]  {n_samples} samples | {n_features} features | "
              f"{n_classes} classes")

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
                        'gap': best_score - rl_score})

    return _summarise(results, "CLASSIFICATION", gap_label="accuracy gap",
                      close_thresh="5 %")


# =============================================================================
# REGRESSION TEST
# =============================================================================
def test_regression(n_tests=8):
    print("\n" + "=" * 70)
    print("  REGRESSION TEST — does RL pick good algorithms?")
    print("=" * 70)

    ppo = _load_ppo(REG_MODEL_PATH)
    if ppo is None:
        print("[SKIP] Train first:  python train_rl_model_selector.py")
        return False

    results = []

    for i in range(n_tests):
        n_samples  = np.random.randint(300, 1200)
        n_features = np.random.randint(5, 40)

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=5.0 + i * 2,
            random_state=200 + i,
        )

        print(f"\n[Test {i+1}]  {n_samples} samples | {n_features} features")

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
                        'gap': best_score - rl_score})

    return _summarise(results, "REGRESSION", gap_label="R² gap",
                      close_thresh="10 %")


# =============================================================================
# SUMMARY HELPER
# =============================================================================
def _summarise(results, label, gap_label, close_thresh):
    if not results:
        print(f"\n[FAIL] No tests completed")
        return False

    n      = len(results)
    n_best  = sum(1 for r in results if r['is_best'])
    n_close = sum(1 for r in results if r['is_close'])
    avg_gap = float(np.mean([r['gap'] for r in results]))

    print(f"\n{'='*70}")
    print(f"  {label} SUMMARY")
    print(f"{'='*70}")
    print(f"   Exact best    : {n_best}/{n}  ({100*n_best/n:.0f} %)")
    print(f"   Within {close_thresh} : {n_close}/{n}  ({100*n_close/n:.0f} %)")
    print(f"   Avg {gap_label}  : {avg_gap:.4f}")

    passed = n_close >= n * 0.5    # pass if ≥ 50 % are close to best
    grade  = "PASS" if passed else "NEEDS MORE TRAINING"
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

    clf_pass = test_classification(n_tests=8)
    reg_pass = test_regression(n_tests=8)

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
