# rl_selector/data_collection.py

"""
Real Dataset Collection from OpenML for RL Model Selector Training.

This module downloads real datasets from OpenML (https://www.openml.org/),
extracts meta-features using our ProfilerAgent, and trains all candidate
models to record their real cross-validation scores.

The result is a JSON file containing:
  [
    {
      "dataset_id": 31,
      "dataset_name": "credit-g",
      "meta_features": [0.54, 0.32, ...],   # 32 real meta-features
      "model_scores": {"XGBClassifier": 0.76, "LGBMClassifier": 0.74, ...}
    },
    ...
  ]

This real data trains the PPO agent to make accurate model recommendations.

Usage:
    python -m rl_selector.data_collection --task classification --n 150
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Regression models
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')


# =========================================================================
# CURATED DATASET IDS
# =========================================================================
# These are well-known, clean OpenML datasets that represent diverse
# real-world problems. Hand-picked to cover different:
#   - Sizes (100 to 100K+ rows)
#   - Dimensionalities (4 to 500+ features)
#   - Domains (finance, health, biology, engineering, social)
#   - Difficulty levels (easy to hard)

CLASSIFICATION_DATASET_IDS = [
    # Small datasets (<1000 rows)
    31,     # credit-g (1000 rows, 20 features — credit risk)
    37,     # diabetes (768 rows, 8 features — Pima Indians diabetes)
    44,     # spambase (4601 rows, 57 features — email spam)
    50,     # tic-tac-toe (958 rows, 9 features — game outcome)
    54,     # vehicle (846 rows, 18 features — vehicle type)
    
    # Medium datasets (1000-10K rows)
    151,    # electricity (45312 rows, 8 features — price up/down)
    182,    # satimage (6430 rows, 36 features — satellite imagery)
    188,    # eucalyptus (736 rows, 19 features — tree species)
    1462,   # banknote-authentication (1372 rows, 4 features)
    1464,   # blood-transfusion (748 rows, 4 features)
    1480,   # ilpd (583 rows, 10 features — Indian liver patient)
    1494,   # qsar-biodeg (1055 rows, 41 features — chemical)
    1510,   # wdbc (569 rows, 30 features — breast cancer Wisconsin)
    
    # Larger datasets (10K+ rows)
    1461,   # bank-marketing (45211 rows, 16 features)
    1489,   # phoneme (5404 rows, 5 features — speech)
    1590,   # adult (48842 rows, 14 features — income prediction)
    4534,   # PhishingWebsites (11055 rows, 30 features)
    
    # High-dimensional
    1501,   # semeion (1593 rows, 256 features — handwriting)
    40496,  # LED-display-domain-7digit (500 rows, 7 features)
    40668,  # connect-4 (67557 rows, 42 features)
    40670,  # dna (3186 rows, 180 features — biology)
    40701,  # churn (5000 rows, 20 features — customer churn)
    40975,  # car (1728 rows, 6 features — car evaluation)
    40982,  # steel-plates-fault (1941 rows, 27 features)
    40983,  # wilt (4839 rows, 5 features — vegetation)
    40984,  # segment (2310 rows, 19 features — image segment)
    
    # Multi-class
    40994,  # climate-model-simulation (540 rows, 18 features)
    41027,  # jungle_chess (44819 rows, 6 features)
    
    # Additional diversity
    23,     # cmc (1473 rows, 9 features — contraceptive method)
    29,     # credit-approval (690 rows, 15 features)
    38,     # sick (3772 rows, 29 features — thyroid)
]

REGRESSION_DATASET_IDS = [
    # Classic regression
    507,    # wine-quality-red (1599 rows, 11 features)
    531,    # boston (506 rows, 13 features — house prices)
    546,    # sensory (576 rows, 11 features)
    
    # Medium
    41021,  # Moneyball (1232 rows, 15 features — baseball)
    41980,  # diamonds (53940 rows, 9 features — diamond prices)
    42225,  # kin8nm (8192 rows, 8 features — kinematics)
    42570,  # california_housing (20640 rows, 8 features)
    
    # Larger / higher-dim
    287,    # wine-quality-white (4898 rows, 11 features)
    42571,  # cpu_act (8192 rows, 21 features)
    42705,  # abalone (4177 rows, 8 features — biology)
    
    # Additional variety
    41187,  # bike_sharing_demand subset
    422,    # analcatdata_vineyard (52 rows, 3 features)
    505,    # tecator (240 rows, 124 features — high dim)
    41702,  # pol (15000 rows, 48 features)
]


def get_classification_models() -> Dict:
    """Get all classification models with safe default parameters."""
    return {
        'XGBClassifier': XGBClassifier(
            n_estimators=100, max_depth=6, tree_method='auto',
            random_state=42, eval_metric='logloss', verbosity=0
        ),
        'LGBMClassifier': LGBMClassifier(
            n_estimators=100, max_depth=6, random_state=42, verbose=-1
        ),
        'CatBoostClassifier': CatBoostClassifier(
            iterations=100, depth=6, random_state=42, verbose=0
        ),
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'ExtraTreesClassifier': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'GradientBoostingClassifier': GradientBoostingClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'SVC': SVC(
            probability=True, random_state=42
        ),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'GaussianNB': GaussianNB(),
    }


def get_regression_models() -> Dict:
    """Get all regression models with safe default parameters."""
    return {
        'XGBRegressor': XGBRegressor(
            n_estimators=100, max_depth=6, tree_method='auto',
            random_state=42, verbosity=0
        ),
        'LGBMRegressor': LGBMRegressor(
            n_estimators=100, max_depth=6, random_state=42, verbose=-1
        ),
        'CatBoostRegressor': CatBoostRegressor(
            iterations=100, depth=6, random_state=42, verbose=0
        ),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'ExtraTreesRegressor': ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        ),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
    }


def extract_meta_features(df: pd.DataFrame, target_col: str,
                          task_type: str) -> Optional[np.ndarray]:
    """
    Extract 32 meta-features from a real dataset.
    
    This is a standalone version of ProfilerAgent._extract_meta_features()
    that works without the full pipeline state, for use during data collection.
    
    Args:
        df: The full DataFrame (features + target)
        target_col: Name of the target column
        task_type: 'classification' or 'regression'
    
    Returns:
        numpy array of 32 float32 meta-features, or None if extraction fails
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge as RidgeLM
        from sklearn.neighbors import KNeighborsClassifier as KNC, KNeighborsRegressor as KNR
        
        features = []
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Detect numeric/categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_numeric = X[numeric_cols].fillna(X[numeric_cols].median()) if numeric_cols else pd.DataFrame()
        
        n_samples = len(df)
        n_features = len(X.columns)
        
        # === BASIC (6) ===
        features.append(np.log10(n_samples + 1) / 6)
        features.append(np.log10(n_features + 1) / 4)
        features.append(len(numeric_cols) / max(n_features, 1))
        features.append(len(categorical_cols) / max(n_features, 1))
        features.append(y.nunique() / 100 if task_type == 'classification' else 0)
        features.append(n_features / max(n_samples, 1))
        
        # === MISSING (3) ===
        missing_ratio = df.isna().sum().sum() / (n_samples * len(df.columns))
        cols_with_missing = (df.isna().sum() > 0).sum() / len(df.columns)
        max_missing = df.isna().mean().max()
        features.extend([missing_ratio, cols_with_missing, max_missing])
        
        # === STATISTICAL (10) ===
        if len(X_numeric.columns) > 0:
            skewness = X_numeric.skew()
            kurtosis = X_numeric.kurtosis()
            
            features.append(np.clip(skewness.mean(), -10, 10) / 10)
            features.append(np.clip(kurtosis.mean(), -100, 100) / 100)
            
            Q1 = X_numeric.quantile(0.25)
            Q3 = X_numeric.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X_numeric < Q1 - 1.5 * IQR) | (X_numeric > Q3 + 1.5 * IQR)).sum().sum()
            features.append(outliers / (n_samples * len(X_numeric.columns)))
            
            if len(X_numeric.columns) > 1:
                corr = X_numeric.corr().abs()
                features.append(corr.values[np.triu_indices_from(corr.values, 1)].mean())
            else:
                features.append(0)
            
            cv = X_numeric.std() / (X_numeric.mean().abs() + 1e-10)
            features.append(np.clip(cv.mean(), 0, 10) / 10)
            features.append(np.clip(cv.std(), 0, 10) / 10)
            features.append(np.clip(skewness.std(), 0, 10) / 10)
            features.append(np.clip(kurtosis.std(), 0, 100) / 100)
            
            ranges = (X_numeric.max() - X_numeric.min()) / (X_numeric.std() + 1e-10)
            features.append(np.clip(ranges.mean(), 0, 100) / 100)
            features.append((X_numeric == 0).sum().sum() / (n_samples * len(X_numeric.columns)))
        else:
            features.extend([0] * 10)
        
        # === CATEGORICAL (3) ===
        if len(categorical_cols) > 0:
            cardinalities = [df[c].nunique() for c in categorical_cols]
            features.append(np.mean(cardinalities) / 100)
            features.append(np.max(cardinalities) / 1000)
            features.append(sum(1 for c in cardinalities if c > 20) / len(categorical_cols))
        else:
            features.extend([0, 0, 0])
        
        # === TARGET (3) ===
        if task_type == 'classification':
            vc = y.value_counts(normalize=True)
            features.append(1 - vc.max())
            features.extend([0, 0])
        else:
            y_num = pd.to_numeric(y, errors='coerce').dropna()
            if len(y_num) > 0:
                features.append(0)
                features.append(np.clip(y_num.skew(), -10, 10) / 10)
                features.append(np.clip(y_num.kurtosis(), -100, 100) / 100)
            else:
                features.extend([0, 0, 0])
        
        # === PCA (3) ===
        if len(X_numeric.columns) > 1:
            try:
                X_scaled = (X_numeric - X_numeric.mean()) / (X_numeric.std() + 1e-10)
                X_scaled = X_scaled.fillna(0)
                pca = PCA()
                pca.fit(X_scaled)
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                n_95 = np.argmax(cumvar >= 0.95) + 1
                var_50 = cumvar[min(len(cumvar) - 1, int(len(cumvar) * 0.5))]
                features.append(n_95 / len(X_numeric.columns))
                features.append(var_50)
                features.append(n_95 / max(n_samples, 1))
            except:
                features.extend([0.5, 0.5, 0.01])
        else:
            features.extend([1.0, 1.0, 0.01])
        
        # === LANDMARKS (4) — quick model scores ===
        if len(X_numeric.columns) > 0 and n_samples >= 50:
            sample_size = min(1000, n_samples)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X_s = X_numeric.iloc[idx].values
            y_s = y.iloc[idx]
            
            if task_type == 'classification':
                le = LabelEncoder()
                y_s = le.fit_transform(y_s.astype(str))
                scoring = 'accuracy'
            else:
                y_s = pd.to_numeric(y_s, errors='coerce').fillna(0).values
                scoring = 'r2'
            
            for ModelClass in [
                DecisionTreeClassifier(max_depth=3, random_state=42) if task_type == 'classification' else DecisionTreeRegressor(max_depth=3, random_state=42),
                GaussianNB() if task_type == 'classification' else LinearRegression(),
                LogisticRegression(max_iter=100, random_state=42) if task_type == 'classification' else RidgeLM(random_state=42),
                KNC(n_neighbors=3) if task_type == 'classification' else KNR(n_neighbors=3),
            ]:
                try:
                    score = cross_val_score(ModelClass, X_s, y_s, cv=3, scoring=scoring)
                    features.append(np.clip(score.mean(), 0, 1))
                except:
                    features.append(0.5)
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        # Ensure exactly 32
        features = features[:32]
        while len(features) < 32:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"    Meta-feature extraction failed: {e}")
        return None


def prepare_dataset_for_modeling(df: pd.DataFrame, target_col: str,
                                  task_type: str) -> tuple:
    """
    Prepare a raw OpenML dataset for model training.
    
    Quick preprocessing:
      - Drop columns with >50% missing
      - Impute remaining missing (median for numeric, mode for categorical)
      - Label-encode categorical features
      - Label-encode target (for classification)
      - Scale features
    
    Returns:
        (X_prepared, y_prepared) or (None, None) if preparation fails
    """
    try:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # Drop columns with >50% missing
        missing_pct = X.isna().mean()
        X = X.loc[:, missing_pct < 0.5]
        
        if len(X.columns) == 0:
            return None, None
        
        # Separate numeric and categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute numeric: median
        if numeric_cols:
            imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        # Impute + encode categorical: mode then label encode
        for col in categorical_cols:
            X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Encode target for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
        else:
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.median())
        
        # Drop any remaining NaN rows
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None, None
        
        return X, y
    
    except Exception as e:
        print(f"    Dataset preparation failed: {e}")
        return None, None


def evaluate_all_models(X: pd.DataFrame, y: pd.Series,
                        task_type: str, cv_folds: int = 3) -> Dict[str, float]:
    """
    Train and evaluate ALL candidate models on a single dataset.
    
    Uses 3-fold CV (not 5) for speed during data collection.
    Records the mean CV score for each model.
    
    Args:
        X: Prepared feature matrix
        y: Prepared target vector
        task_type: 'classification' or 'regression'
        cv_folds: Number of CV folds (3 for speed)
    
    Returns:
        Dict mapping model name → mean CV score
    """
    if task_type == 'classification':
        models = get_classification_models()
        scoring = 'accuracy'
    else:
        models = get_regression_models()
        scoring = 'r2'
    
    model_scores = {}
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring,
                                     error_score='raise')
            score = float(np.clip(scores.mean(), 0, 1))
            model_scores[name] = round(score, 4)
        except Exception as e:
            # If a model fails, give it a low default score
            model_scores[name] = 0.5 if task_type == 'classification' else 0.0
    
    return model_scores


def collect_training_data(task_type: str = 'classification',
                          n_datasets: int = 150,
                          save_path: Optional[str] = None) -> List[Dict]:
    """
    Main function: Download real datasets, extract meta-features, evaluate models.
    
    Process for each dataset:
      1. Download from OpenML by ID
      2. Extract 32 meta-features
      3. Prepare data (impute, encode, scale)
      4. Train all 10-11 models with 3-fold CV
      5. Record scores
    
    Skips datasets that:
      - Fail to download
      - Are too small (<50 rows) or too large (>100K rows)
      - Have meta-feature extraction failures
    
    Args:
        task_type: 'classification' or 'regression'
        n_datasets: Max number of datasets to collect
        save_path: Optional path to save JSON results
    
    Returns:
        List of dicts with 'meta_features', 'model_scores', 'dataset_name', 'dataset_id'
    """
    import openml
    
    print(f"\n{'='*70}")
    print(f"  COLLECTING REAL TRAINING DATA FOR RL MODEL SELECTOR")
    print(f"  Task: {task_type} | Target datasets: {n_datasets}")
    print(f"{'='*70}\n")
    
    # Get curated dataset IDs
    if task_type == 'classification':
        dataset_ids = CLASSIFICATION_DATASET_IDS.copy()
    else:
        dataset_ids = REGRESSION_DATASET_IDS.copy()
    
    # If we need more, fetch popular ones from OpenML
    if len(dataset_ids) < n_datasets:
        print("Fetching additional dataset IDs from OpenML...")
        try:
            datasets_df = openml.datasets.list_datasets(output_format='dataframe')
            
            if task_type == 'classification':
                # Filter: classification tasks, reasonable size
                filtered = datasets_df[
                    (datasets_df['NumberOfInstances'] >= 100) &
                    (datasets_df['NumberOfInstances'] <= 100000) &
                    (datasets_df['NumberOfFeatures'] >= 3) &
                    (datasets_df['NumberOfFeatures'] <= 500) &
                    (datasets_df['NumberOfClasses'] >= 2) &
                    (datasets_df['NumberOfClasses'] <= 50)
                ]
            else:
                filtered = datasets_df[
                    (datasets_df['NumberOfInstances'] >= 100) &
                    (datasets_df['NumberOfInstances'] <= 100000) &
                    (datasets_df['NumberOfFeatures'] >= 3) &
                    (datasets_df['NumberOfFeatures'] <= 500)
                ]
            
            # Sort by number of downloads (popularity = quality proxy)
            if 'NumberOfDownloads' in filtered.columns:
                filtered = filtered.sort_values('NumberOfDownloads', ascending=False)
            
            additional_ids = [int(did) for did in filtered.index.tolist()
                            if int(did) not in dataset_ids]
            dataset_ids.extend(additional_ids[:n_datasets - len(dataset_ids)])
        except Exception as e:
            print(f"  Warning: Could not fetch additional datasets ({e})")
    
    # Limit to requested count
    dataset_ids = dataset_ids[:n_datasets]
    
    training_data = []
    success_count = 0
    fail_count = 0
    
    for i, dataset_id in enumerate(dataset_ids):
        print(f"\n[{i+1}/{len(dataset_ids)}] Processing dataset ID: {dataset_id}")
        
        try:
            # Step 1: Download dataset from OpenML
            dataset = openml.datasets.get_dataset(
                dataset_id,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False
            )
            
            df, y_array, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format='dataframe',
                target=dataset.default_target_attribute
            )
            
            target_col = dataset.default_target_attribute
            dataset_name = dataset.name
            
            print(f"  Dataset: {dataset_name}")
            print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} cols | Target: {target_col}")
            
            # Skip if too small or too large
            if len(df) < 50:
                print(f"  ⏭ Skipped: too few rows ({len(df)})")
                fail_count += 1
                continue
            if len(df) > 100000:
                # Sample down to 100K for speed
                df = df.sample(n=100000, random_state=42)
                print(f"  📉 Sampled down to 100,000 rows")
            
            # Step 2: Extract 32 meta-features
            print(f"  Extracting meta-features...")
            meta_features = extract_meta_features(df, target_col, task_type)
            
            if meta_features is None:
                print(f"  ⏭ Skipped: meta-feature extraction failed")
                fail_count += 1
                continue
            
            # Step 3: Prepare data for ML
            print(f"  Preparing data...")
            X_prepared, y_prepared = prepare_dataset_for_modeling(df, target_col, task_type)
            
            if X_prepared is None:
                print(f"  ⏭ Skipped: data preparation failed")
                fail_count += 1
                continue
            
            # Step 4: Evaluate all models
            print(f"  Training all models (3-fold CV)...")
            start_time = time.time()
            model_scores = evaluate_all_models(X_prepared, y_prepared, task_type, cv_folds=3)
            elapsed = time.time() - start_time
            
            # Step 5: Record results
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            
            training_data.append({
                'dataset_id': int(dataset_id),
                'dataset_name': dataset_name,
                'n_rows': int(len(df)),
                'n_cols': int(len(df.columns)),
                'meta_features': meta_features.tolist(),
                'model_scores': model_scores
            })
            
            success_count += 1
            print(f"  ✅ Done in {elapsed:.1f}s | Best: {best_model} ({best_score:.4f})")
            
            # Print all scores
            sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            for model_name, score in sorted_scores:
                marker = "👑" if model_name == best_model else "  "
                print(f"    {marker} {model_name:30s} {score:.4f}")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            fail_count += 1
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Successful: {success_count} | Failed: {fail_count} | Total: {len(dataset_ids)}")
    print(f"{'='*70}")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\n  Saved to: {save_path}")
    
    return training_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect real training data from OpenML')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type')
    parser.add_argument('--n', type=int, default=150,
                       help='Number of datasets to collect')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON path')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"./rl_selector/models/{args.task}_training_data.json"
    
    collect_training_data(
        task_type=args.task,
        n_datasets=args.n,
        save_path=args.output
    )
