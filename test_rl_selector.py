"""
Quick test script to verify the RL Model Selector is working correctly.
Compares RL's top choice vs actual best performer on random datasets.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Import the actual MetaFeatureExtractor from the project
from rl_model_selector_classification_regression import MetaFeatureExtractor, TaskType, RLModelSelectorConfig

# Classification models to test
CLF_MODELS = {
    0: ('LogisticRegression', LogisticRegression(max_iter=500)),
    1: ('RandomForestClassifier', RandomForestClassifier(n_estimators=50)),
    2: ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=50)),
    3: ('KNeighborsClassifier', KNeighborsClassifier()),
    4: ('SVC', SVC()),
}

# Regression models to test  
REG_MODELS = {
    0: ('Ridge', Ridge()),
    1: ('RandomForestRegressor', RandomForestRegressor(n_estimators=50)),
    2: ('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=50)),
    3: ('KNeighborsRegressor', KNeighborsRegressor()),
    4: ('SVR', SVR()),
}

def test_classification():
    print("\n" + "="*70)
    print(" CLASSIFICATION TEST - Checking if RL picks good algorithms")
    print("="*70)
    
    try:
        model = PPO.load("./rl_model_selector_gpu/models/ppo_clf_gpu")
        print("[OK] Loaded classification model")
    except Exception as e:
        print(f"[FAIL] Could not load classification model: {e}")
        return False
    
    extractor = MetaFeatureExtractor()
    results = []
    n_tests = 5
    
    for i in range(n_tests):
        # Generate random dataset
        n_samples = np.random.randint(200, 1000)
        n_features = np.random.randint(5, 30)
        n_classes = np.random.randint(2, 5)
        
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=max(2, n_features//2),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=i
        )
        
        # Convert to DataFrame/Series for MetaFeatureExtractor
        X_df = pd.DataFrame(X, columns=[f'f{j}' for j in range(n_features)])
        y_series = pd.Series(y)
        
        print(f"\n[Test {i+1}] {n_samples} samples, {n_features} features, {n_classes} classes")
        
        # Get RL prediction using the actual feature extractor
        try:
            features = extractor.extract(X_df, y_series, TaskType.CLASSIFICATION, compute_landmarks=False)
            features_array = np.array([features[name] for name in MetaFeatureExtractor.FEATURE_NAMES], dtype=np.float32)
            # Replace NaN with 0
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            action, _ = model.predict(features_array, deterministic=True)
            rl_choice = int(action) % len(CLF_MODELS)
            rl_model_name = CLF_MODELS.get(rl_choice, CLF_MODELS[0])[0]
        except Exception as e:
            print(f"   [ERROR] Feature extraction failed: {e}")
            continue
        
        # Standardize data for model testing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test all models and find actual best
        scores = {}
        for idx, (name, clf) in CLF_MODELS.items():
            try:
                score = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy').mean()
                scores[name] = score
            except:
                scores[name] = 0
        
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        rl_score = scores.get(rl_model_name, 0)
        
        is_best = rl_model_name == best_model
        is_close = (best_score - rl_score) < 0.05  # Within 5%
        
        status = "[BEST]" if is_best else ("[CLOSE]" if is_close else "[MISS]")
        
        print(f"   RL Choice: {rl_model_name} (Accuracy: {rl_score:.3f})")
        print(f"   Actual Best: {best_model} (Accuracy: {best_score:.3f})")
        print(f"   Result: {status}")
        
        results.append({
            'is_best': is_best,
            'is_close': is_close,
            'gap': best_score - rl_score
        })
    
    # Summary
    if not results:
        print("\n[FAIL] No tests completed successfully")
        return False
        
    n_best = sum(1 for r in results if r['is_best'])
    n_close = sum(1 for r in results if r['is_close'])
    avg_gap = np.mean([r['gap'] for r in results])
    
    print(f"\n{'='*70}")
    print(f" CLASSIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"   Picked BEST model: {n_best}/{len(results)} ({100*n_best/len(results):.0f}%)")
    print(f"   Within 5%% of best: {n_close}/{len(results)} ({100*n_close/len(results):.0f}%)")
    print(f"   Average gap from best: {avg_gap:.3f}")
    
    return n_close >= len(results) * 0.5  # Pass if 50%+ are close to best

def test_regression():
    print("\n" + "="*70)
    print(" REGRESSION TEST - Checking if RL picks good algorithms")
    print("="*70)
    
    try:
        model = PPO.load("./rl_model_selector_gpu/models/ppo_reg_gpu")
        print("[OK] Loaded regression model")
    except Exception as e:
        print(f"[FAIL] Could not load regression model: {e}")
        return False
    
    extractor = MetaFeatureExtractor()
    results = []
    n_tests = 5
    
    for i in range(n_tests):
        # Generate random dataset
        n_samples = np.random.randint(200, 1000)
        n_features = np.random.randint(5, 30)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//2),
            noise=10,
            random_state=i
        )
        
        # Convert to DataFrame/Series for MetaFeatureExtractor
        X_df = pd.DataFrame(X, columns=[f'f{j}' for j in range(n_features)])
        y_series = pd.Series(y)
        
        print(f"\n[Test {i+1}] {n_samples} samples, {n_features} features")
        
        # Get RL prediction using the actual feature extractor
        try:
            features = extractor.extract(X_df, y_series, TaskType.REGRESSION, compute_landmarks=False)
            features_array = np.array([features[name] for name in MetaFeatureExtractor.FEATURE_NAMES], dtype=np.float32)
            # Replace NaN with 0
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            action, _ = model.predict(features_array, deterministic=True)
            rl_choice = int(action) % len(REG_MODELS)
            rl_model_name = REG_MODELS.get(rl_choice, REG_MODELS[0])[0]
        except Exception as e:
            print(f"   [ERROR] Feature extraction failed: {e}")
            continue
        
        # Standardize data for model testing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test all models and find actual best
        scores = {}
        for idx, (name, reg) in REG_MODELS.items():
            try:
                score = cross_val_score(reg, X_scaled, y, cv=3, scoring='r2').mean()
                scores[name] = score
            except:
                scores[name] = -999
        
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        rl_score = scores.get(rl_model_name, -999)
        
        is_best = rl_model_name == best_model
        is_close = (best_score - rl_score) < 0.1  # Within 10% R2
        
        status = "[BEST]" if is_best else ("[CLOSE]" if is_close else "[MISS]")
        
        print(f"   RL Choice: {rl_model_name} (R2: {rl_score:.3f})")
        print(f"   Actual Best: {best_model} (R2: {best_score:.3f})")
        print(f"   Result: {status}")
        
        results.append({
            'is_best': is_best,
            'is_close': is_close,
            'gap': best_score - rl_score
        })
    
    # Summary
    if not results:
        print("\n[FAIL] No tests completed successfully")
        return False
        
    n_best = sum(1 for r in results if r['is_best'])
    n_close = sum(1 for r in results if r['is_close'])
    avg_gap = np.mean([r['gap'] for r in results])
    
    print(f"\n{'='*70}")
    print(f" REGRESSION SUMMARY")
    print(f"{'='*70}")
    print(f"   Picked BEST model: {n_best}/{len(results)} ({100*n_best/len(results):.0f}%)")
    print(f"   Within 10%% of best: {n_close}/{len(results)} ({100*n_close/len(results):.0f}%)")
    print(f"   Average gap from best: {avg_gap:.3f}")
    
    return n_close >= len(results) * 0.5

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RL MODEL SELECTOR VALIDATION TEST")
    print("="*70)
    
    clf_pass = test_classification()
    reg_pass = test_regression()
    
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)
    
    if clf_pass and reg_pass:
        print("[PASS] The RL Model Selector is working well!")
        print("       It consistently picks good (if not always the best) algorithms.")
    elif clf_pass or reg_pass:
        print("[PARTIAL] The RL Model Selector is partially working.")
        print("          One task type performs better than the other.")
    else:
        print("[NEEDS WORK] The RL Model Selector could use more training.")
        print("             Consider running with more timesteps (--timesteps 200000)")
