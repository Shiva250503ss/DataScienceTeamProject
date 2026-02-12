"""
================================================================================
RL MODEL SELECTOR - GPU REQUIRED (Classification & Regression)
================================================================================

A GPU-accelerated Reinforcement Learning system that learns to select the best 
ML models for ANY dataset - both Classification and Regression tasks.

*** REQUIRES GPU (CUDA) TO RUN - WILL EXIT IF NO GPU FOUND ***

GPU ACCELERATION:
    - PPO Training: Uses CUDA for neural network training
    - XGBoost: Uses gpu_hist tree method
    - LightGBM: Uses GPU device
    - CatBoost: Uses GPU task type

REQUIREMENTS:
    # Core packages
    pip install numpy pandas scikit-learn gymnasium stable-baselines3
    
    # GPU-enabled ML libraries
    pip install xgboost  # GPU support included
    pip install lightgbm  # GPU build required
    pip install catboost  # GPU support included
    
    # PyTorch with CUDA
    pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
    # OR
    pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
    
    # OpenML (optional)
    pip install openml

USAGE:
    # Check GPU availability
    python rl_model_selector_gpu.py --mode check-gpu
    
    # Train (GPU required - will exit if no GPU)
    python rl_model_selector_gpu.py --mode train --task both
    
    # Select models for your dataset
    python rl_model_selector_gpu.py --mode select --data your_data.csv --target target_col

Author: DataPilot AI Pro
Version: 4.0.0 (GPU Required)
================================================================================
"""

import os
import sys
import pickle
import time
import json
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from pathlib import Path
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ==============================================================================
# SECTION 1: GPU DETECTION AND REQUIREMENT
# ==============================================================================

class GPUManager:
    """Manages GPU detection and enforces GPU requirement."""
    
    def __init__(self):
        self.cuda_available = False
        self.gpu_name = None
        self.gpu_memory = None
        self.cuda_version = None
        
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect CUDA GPU availability."""
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.cuda_version = torch.version.cuda
        except ImportError:
            self.cuda_available = False
    
    def require_gpu(self):
        """Exit program if GPU is not available."""
        if not self.cuda_available:
            print("\n" + "=" * 70)
            print(" ERROR: GPU (CUDA) IS REQUIRED BUT NOT AVAILABLE")
            print("=" * 70)
            print("""
This application REQUIRES an NVIDIA GPU with CUDA support to run.

To fix this issue:
  1. Ensure you have an NVIDIA GPU installed
  2. Install NVIDIA drivers (https://www.nvidia.com/drivers)
  3. Install CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
  4. Install PyTorch with CUDA support:
     
     # For CUDA 11.8:
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     
     # For CUDA 12.1:
     pip install torch --index-url https://download.pytorch.org/whl/cu121

To check GPU status:
  python rl_model_selector_gpu.py --mode check-gpu
""")
            print("=" * 70)
            sys.exit(1)
        
        print(f"\n GPU Detected: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
    
    def print_gpu_info(self):
        """Print detailed GPU information."""
        print("\n" + "=" * 60)
        print("GPU STATUS CHECK")
        print("=" * 60)
        
        try:
            import torch
            print(f"\nPyTorch Version: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"\n GPU DETECTED")
                print(f"   Name: {self.gpu_name}")
                print(f"   Memory: {self.gpu_memory:.1f} GB")
                print(f"   CUDA Version: {self.cuda_version}")
                print(f"   Device Count: {torch.cuda.device_count()}")
                
                # Test GPU
                print(f"\n GPU Test...")
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                print(f"   Matrix multiplication: PASSED")
                del x, y
                torch.cuda.empty_cache()
                
                # Check ML libraries
                print(f"\n GPU-Enabled Libraries:")
                
                try:
                    import xgboost as xgb
                    print(f"   [OK] XGBoost {xgb.__version__} (gpu_hist)")
                except:
                    print(f"   [MISSING] XGBoost not installed")
                
                try:
                    import lightgbm as lgb
                    print(f"   [OK] LightGBM {lgb.__version__} (device=gpu)")
                except:
                    print(f"   [MISSING] LightGBM not installed")
                
                try:
                    import catboost
                    print(f"   [OK] CatBoost {catboost.__version__} (task_type=GPU)")
                except:
                    print(f"   [MISSING] CatBoost not installed")
                
                print(f"\n System ready for GPU-accelerated training!")
            else:
                print(f"\n NO GPU DETECTED")
                print(f"\nPlease install PyTorch with CUDA:")
                print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        except ImportError:
            print(f"\n PyTorch not installed")
            print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n" + "=" * 60)


# Global GPU manager
gpu_manager = GPUManager()


# ==============================================================================
# SECTION 2: TASK TYPE
# ==============================================================================

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    
    @staticmethod
    def detect(y: pd.Series) -> 'TaskType':
        if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'bool':
            return TaskType.CLASSIFICATION
        n_unique = y.nunique()
        if n_unique <= 20 and n_unique / len(y) < 0.05:
            return TaskType.CLASSIFICATION
        if y.dtype in ['int64', 'int32'] and n_unique <= 10:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION


# ==============================================================================
# SECTION 3: CONFIGURATION
# ==============================================================================

@dataclass
class RLModelSelectorConfig:
    """Configuration for GPU-accelerated RL Model Selector."""
    
    BASE_DIR: str = "./rl_model_selector_gpu"
    DATA_DIR: str = field(default="")
    MODELS_DIR: str = field(default="")
    LOGS_DIR: str = field(default="")
    
    TRAINING_DATA_CLF_FILE: str = "training_data_clf.pkl"
    TRAINING_DATA_REG_FILE: str = "training_data_reg.pkl"
    PPO_MODEL_CLF_FILE: str = "ppo_clf_gpu"
    PPO_MODEL_REG_FILE: str = "ppo_reg_gpu"
    
    GPU_DEVICE_ID: int = 0
    
    MIN_SAMPLES: int = 100
    MAX_SAMPLES: int = 50000
    MIN_FEATURES: int = 4
    MAX_FEATURES: int = 100
    MIN_CLASSES: int = 2
    MAX_CLASSES: int = 10
    N_DATASETS_CLF: int = 150
    N_DATASETS_REG: int = 150
    
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42
    
    CLF_MODEL_NAMES: List[str] = field(default_factory=lambda: [
        'XGBClassifier_GPU', 'LGBMClassifier_GPU', 'CatBoostClassifier_GPU',
        'RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier',
        'LogisticRegression', 'SVC', 'KNeighborsClassifier', 'GaussianNB'
    ])
    
    REG_MODEL_NAMES: List[str] = field(default_factory=lambda: [
        'XGBRegressor_GPU', 'LGBMRegressor_GPU', 'CatBoostRegressor_GPU',
        'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor',
        'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'KNeighborsRegressor'
    ])
    
    PPO_TOTAL_TIMESTEPS: int = 100000
    PPO_LEARNING_RATE: float = 3e-4
    PPO_N_STEPS: int = 2048
    PPO_BATCH_SIZE: int = 64
    PPO_N_EPOCHS: int = 10
    PPO_GAMMA: float = 0.99
    PPO_NET_ARCH: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    LANDMARK_SUBSAMPLE: int = 1000
    LANDMARK_CV_FOLDS: int = 3
    
    def __post_init__(self):
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.LOGS_DIR = os.path.join(self.BASE_DIR, "logs")
        for d in [self.BASE_DIR, self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)
    
    def get_model_names(self, task_type: TaskType):
        return self.CLF_MODEL_NAMES if task_type == TaskType.CLASSIFICATION else self.REG_MODEL_NAMES
    
    def get_ppo_model_file(self, task_type: TaskType):
        return self.PPO_MODEL_CLF_FILE if task_type == TaskType.CLASSIFICATION else self.PPO_MODEL_REG_FILE


# ==============================================================================
# SECTION 4: LOGGING
# ==============================================================================

def setup_logging(config, name="rl_gpu"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(os.path.join(config.LOGS_DIR, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==============================================================================
# SECTION 5: META-FEATURE EXTRACTOR
# ==============================================================================

class MetaFeatureExtractor:
    FEATURE_NAMES = [
        'n_samples', 'n_features', 'n_numeric', 'n_categorical', 'target_unique', 'dimensionality',
        'missing_ratio', 'cols_with_missing', 'max_missing_ratio',
        'target_stat_1', 'target_stat_2', 'target_stat_3',
        'mean_skewness', 'max_skewness', 'mean_kurtosis', 'max_kurtosis',
        'mean_outlier_ratio', 'max_outlier_ratio', 'mean_correlation', 'max_correlation',
        'high_corr_pairs', 'mean_cv', 'mean_cardinality', 'max_cardinality',
        'high_cardinality_cols', 'pca_95_components', 'pca_50_variance', 'intrinsic_dimensionality',
        'landmark_1', 'landmark_2', 'landmark_3', 'landmark_4'
    ]
    
    def __init__(self, config=None):
        self.config = config or RLModelSelectorConfig()
    
    def extract(self, X, y, task_type=None, compute_landmarks=True):
        if task_type is None:
            task_type = TaskType.detect(y)
        
        features = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        features['n_samples'] = len(X)
        features['n_features'] = len(X.columns)
        features['n_numeric'] = len(numeric_cols)
        features['n_categorical'] = len(categorical_cols)
        features['target_unique'] = y.nunique()
        features['dimensionality'] = len(X.columns) / max(len(X), 1)
        
        missing = X.isnull().sum() / len(X)
        features['missing_ratio'] = X.isnull().sum().sum() / max(X.size, 1)
        features['cols_with_missing'] = (missing > 0).sum() / max(len(X.columns), 1)
        features['max_missing_ratio'] = missing.max() if len(missing) > 0 else 0
        
        if task_type == TaskType.CLASSIFICATION:
            cc = y.value_counts(normalize=True)
            features['target_stat_1'] = cc.max() / max(cc.min(), 1e-10)
            features['target_stat_2'] = cc.min()
            features['target_stat_3'] = cc.max()
        else:
            yn = pd.to_numeric(y, errors='coerce')
            features['target_stat_1'] = abs(yn.skew()) if not yn.isna().all() else 0
            features['target_stat_2'] = abs(yn.kurtosis()) if not yn.isna().all() else 0
            features['target_stat_3'] = (yn.std() / (abs(yn.mean()) + 1e-10)) if not yn.isna().all() else 0
        
        if len(numeric_cols) > 0:
            ndf = X[numeric_cols]
            features['mean_skewness'] = np.abs(ndf.skew()).mean()
            features['max_skewness'] = np.abs(ndf.skew()).max()
            features['mean_kurtosis'] = np.abs(ndf.kurtosis()).mean()
            features['max_kurtosis'] = np.abs(ndf.kurtosis()).max()
            
            ors = []
            for c in numeric_cols:
                Q1, Q3 = ndf[c].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    ors.append(((ndf[c] < Q1-1.5*IQR)|(ndf[c] > Q3+1.5*IQR)).sum()/len(ndf))
                else:
                    ors.append(0)
            features['mean_outlier_ratio'] = np.mean(ors) if ors else 0
            features['max_outlier_ratio'] = np.max(ors) if ors else 0
            
            if len(numeric_cols) > 1:
                corr = ndf.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                vals = upper.stack().values
                features['mean_correlation'] = np.mean(vals) if len(vals) > 0 else 0
                features['max_correlation'] = np.max(vals) if len(vals) > 0 else 0
                features['high_corr_pairs'] = (vals > 0.8).sum() / max(len(vals), 1)
            else:
                features['mean_correlation'] = features['max_correlation'] = features['high_corr_pairs'] = 0
            features['mean_cv'] = (ndf.std() / (ndf.mean().abs() + 1e-10)).abs().mean()
        else:
            for k in ['mean_skewness', 'max_skewness', 'mean_kurtosis', 'max_kurtosis',
                     'mean_outlier_ratio', 'max_outlier_ratio', 'mean_correlation',
                     'max_correlation', 'high_corr_pairs', 'mean_cv']:
                features[k] = 0
        
        if len(categorical_cols) > 0:
            cards = [X[c].nunique() for c in categorical_cols]
            features['mean_cardinality'] = np.mean(cards)
            features['max_cardinality'] = np.max(cards)
            features['high_cardinality_cols'] = sum(1 for c in cards if c > 10) / len(categorical_cols)
        else:
            features['mean_cardinality'] = features['max_cardinality'] = features['high_cardinality_cols'] = 0
        
        if len(numeric_cols) >= 2:
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                ndf = X[numeric_cols].fillna(X[numeric_cols].median())
                scaled = StandardScaler().fit_transform(ndf)
                pca = PCA(n_components=min(len(numeric_cols), 10))
                pca.fit(scaled)
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                features['pca_95_components'] = np.searchsorted(cumsum, 0.95) + 1
                features['pca_50_variance'] = cumsum[min(len(cumsum)//2, len(cumsum)-1)]
                features['intrinsic_dimensionality'] = features['pca_95_components'] / len(numeric_cols)
            except:
                features['pca_95_components'] = len(numeric_cols)
                features['pca_50_variance'] = 0.5
                features['intrinsic_dimensionality'] = 1.0
        else:
            features['pca_95_components'] = max(len(numeric_cols), 1)
            features['pca_50_variance'] = 0.5
            features['intrinsic_dimensionality'] = 1.0
        
        if compute_landmarks:
            lm = self._landmarks(X, y, task_type, numeric_cols, categorical_cols)
            features.update(lm)
        else:
            for i in range(1, 5):
                features[f'landmark_{i}'] = 0.5
        
        arr = np.array([features[n] for n in self.FEATURE_NAMES], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        for i in [0, 1, 2, 3, 4, 22, 23, 25]:
            if i < len(arr) and arr[i] > 0:
                arr[i] = np.log1p(arr[i])
        return np.clip(arr, -100, 100)
    
    def _landmarks(self, X, y, task_type, numeric_cols, categorical_cols):
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        lm = {}
        try:
            Xl = X.copy()
            for c in categorical_cols:
                Xl[c] = LabelEncoder().fit_transform(Xl[c].astype(str))
            Xl = Xl.fillna(Xl.median())
            if len(Xl) > self.config.LANDMARK_SUBSAMPLE:
                idx = np.random.choice(len(Xl), self.config.LANDMARK_SUBSAMPLE, replace=False)
                Xl, yl = Xl.iloc[idx], y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            else:
                yl = y
            cv = self.config.LANDMARK_CV_FOLDS
            
            if task_type == TaskType.CLASSIFICATION:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.linear_model import LogisticRegression
                from sklearn.neighbors import KNeighborsClassifier
                yl = LabelEncoder().fit_transform(yl)
                lm['landmark_1'] = cross_val_score(DecisionTreeClassifier(max_depth=3), Xl, yl, cv=cv).mean()
                lm['landmark_2'] = cross_val_score(GaussianNB(), Xl, yl, cv=cv).mean()
                lm['landmark_3'] = cross_val_score(LogisticRegression(max_iter=200), Xl, yl, cv=cv).mean()
                lm['landmark_4'] = cross_val_score(KNeighborsClassifier(n_neighbors=1), Xl, yl, cv=cv).mean()
            else:
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.linear_model import Ridge, LinearRegression
                from sklearn.neighbors import KNeighborsRegressor
                yl = pd.to_numeric(yl, errors='coerce').fillna(0)
                lm['landmark_1'] = cross_val_score(DecisionTreeRegressor(max_depth=3), Xl, yl, cv=cv, scoring='r2').mean()
                lm['landmark_2'] = cross_val_score(LinearRegression(), Xl, yl, cv=cv, scoring='r2').mean()
                lm['landmark_3'] = cross_val_score(Ridge(), Xl, yl, cv=cv, scoring='r2').mean()
                lm['landmark_4'] = cross_val_score(KNeighborsRegressor(n_neighbors=3), Xl, yl, cv=cv, scoring='r2').mean()
        except:
            lm = {f'landmark_{i}': 0.5 for i in range(1, 5)}
        return lm


# ==============================================================================
# SECTION 6: GPU MODEL FACTORY
# ==============================================================================

class GPUModelFactory:
    @staticmethod
    def get_models(task_type, config):
        if task_type == TaskType.CLASSIFICATION:
            return GPUModelFactory._clf_models(config)
        return GPUModelFactory._reg_models(config)
    
    @staticmethod
    def _clf_models(config):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        
        rs, gpu = config.RANDOM_STATE, config.GPU_DEVICE_ID
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1),
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=rs),
            'LogisticRegression': LogisticRegression(max_iter=500, random_state=rs, n_jobs=-1),
            'SVC': SVC(probability=True, random_state=rs),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'GaussianNB': GaussianNB(),
        }
        try:
            import xgboost as xgb
            models['XGBClassifier_GPU'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=rs, tree_method='gpu_hist',
                gpu_id=gpu, predictor='gpu_predictor', verbosity=0, eval_metric='logloss'
            )
        except: pass
        try:
            import lightgbm as lgb
            models['LGBMClassifier_GPU'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=rs, device='gpu', gpu_device_id=gpu, verbose=-1
            )
        except: pass
        try:
            from catboost import CatBoostClassifier
            models['CatBoostClassifier_GPU'] = CatBoostClassifier(
                n_estimators=100, max_depth=6, random_state=rs, task_type='GPU', devices=str(gpu), verbose=0
            )
        except: pass
        return models
    
    @staticmethod
    def _reg_models(config):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        rs, gpu = config.RANDOM_STATE, config.GPU_DEVICE_ID
        models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1),
            'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=rs),
            'Ridge': Ridge(random_state=rs),
            'Lasso': Lasso(random_state=rs, max_iter=1000),
            'ElasticNet': ElasticNet(random_state=rs, max_iter=1000),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }
        try:
            import xgboost as xgb
            models['XGBRegressor_GPU'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, random_state=rs, tree_method='gpu_hist',
                gpu_id=gpu, predictor='gpu_predictor', verbosity=0
            )
        except: pass
        try:
            import lightgbm as lgb
            models['LGBMRegressor_GPU'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, random_state=rs, device='gpu', gpu_device_id=gpu, verbose=-1
            )
        except: pass
        try:
            from catboost import CatBoostRegressor
            models['CatBoostRegressor_GPU'] = CatBoostRegressor(
                n_estimators=100, max_depth=6, random_state=rs, task_type='GPU', devices=str(gpu), verbose=0
            )
        except: pass
        return models


# ==============================================================================
# SECTION 7: DATA COLLECTOR
# ==============================================================================

class DataCollector:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.meta_extractor = MetaFeatureExtractor(config)
        self.data_clf = []
        self.data_reg = []
    
    def train_models(self, X, y, task_type):
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        pre = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                             ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
        Xp = pre.fit_transform(X)
        
        if task_type == TaskType.CLASSIFICATION:
            yp = LabelEncoder().fit_transform(y)
            scoring = 'roc_auc_ovr' if len(np.unique(yp)) > 2 else 'roc_auc'
            cv = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=42)
        else:
            yp = pd.to_numeric(y, errors='coerce').fillna(0).values
            scoring = 'r2'
            cv = KFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=42)
        
        models = GPUModelFactory.get_models(task_type, self.config)
        names = self.config.get_model_names(task_type)
        
        perfs = {}
        for n in names:
            if n not in models:
                perfs[n] = 0.5 if task_type == TaskType.CLASSIFICATION else 0.0
                continue
            try:
                scores = cross_val_score(models[n], Xp, yp, cv=cv, scoring=scoring, n_jobs=1)
                perfs[n] = float(max(0, scores.mean()))
            except Exception as e:
                self.logger.warning(f"  {n}: {str(e)[:30]}")
                perfs[n] = 0.5 if task_type == TaskType.CLASSIFICATION else 0.0
        return perfs


# ==============================================================================
# SECTION 8: SYNTHETIC DATA GENERATOR
# ==============================================================================

class SyntheticDataGenerator:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.meta_extractor = MetaFeatureExtractor(config)
    
    def generate_clf(self, n=300):
        from sklearn.datasets import make_classification
        self.logger.info(f"Generating {n} CLF datasets...")
        data = []
        for i in range(n):
            ns, nf = np.random.randint(100, 10000), np.random.randint(5, 50)
            nc = np.random.choice([2, 3, 5])
            X, y = make_classification(n_samples=ns, n_features=nf, n_informative=max(2, nf//2),
                                       n_classes=min(nc, nf//2), random_state=i)
            X = pd.DataFrame(X, columns=[f'f{j}' for j in range(nf)])
            y = pd.Series(y)
            mf = self.meta_extractor.extract(X, y, TaskType.CLASSIFICATION, compute_landmarks=False)
            perfs = self._sim_clf(mf)
            data.append({'meta_features': mf.tolist(),
                        'model_performances': [perfs.get(n, 0.5) for n in self.config.CLF_MODEL_NAMES]})
            if (i+1) % 100 == 0: self.logger.info(f"  {i+1}/{n}")
        return data
    
    def generate_reg(self, n=300):
        from sklearn.datasets import make_regression
        self.logger.info(f"Generating {n} REG datasets...")
        data = []
        for i in range(n):
            ns, nf = np.random.randint(100, 10000), np.random.randint(5, 50)
            X, y = make_regression(n_samples=ns, n_features=nf, n_informative=max(2, nf//2),
                                   noise=np.random.uniform(0.1, 10), random_state=i)
            X = pd.DataFrame(X, columns=[f'f{j}' for j in range(nf)])
            y = pd.Series(y)
            mf = self.meta_extractor.extract(X, y, TaskType.REGRESSION, compute_landmarks=False)
            perfs = self._sim_reg(mf)
            data.append({'meta_features': mf.tolist(),
                        'model_performances': [perfs.get(n, 0.5) for n in self.config.REG_MODEL_NAMES]})
            if (i+1) % 100 == 0: self.logger.info(f"  {i+1}/{n}")
        return data
    
    def _sim_clf(self, mf):
        base = 0.7 + np.random.rand() * 0.15
        perfs = {}
        for m in self.config.CLF_MODEL_NAMES:
            p = base
            if 'XGB' in m: p += 0.03
            elif 'LGBM' in m: p += 0.02
            elif 'CatBoost' in m: p += 0.02
            p += np.random.randn() * 0.02
            perfs[m] = np.clip(p, 0.5, 0.99)
        return perfs
    
    def _sim_reg(self, mf):
        base = 0.6 + np.random.rand() * 0.2
        perfs = {}
        for m in self.config.REG_MODEL_NAMES:
            p = base
            if 'XGB' in m: p += 0.04
            elif 'LGBM' in m: p += 0.03
            elif 'CatBoost' in m: p += 0.03
            p += np.random.randn() * 0.03
            perfs[m] = np.clip(p, 0.0, 0.99)
        return perfs


# ==============================================================================
# SECTION 9: GYMNASIUM ENVIRONMENT
# ==============================================================================

class ModelSelectionEnv(gym.Env):
    def __init__(self, data, model_names, task_type):
        self.data = data
        self.model_names = model_names
        self.task_type = task_type
        self.n_models = len(model_names)
        self.idx = 0
        self.rng = np.random.default_rng()
        self.episodes = 0
        self.optimal = 0
        self.total_reward = 0
        
        nf = len(data[0]['meta_features'])
        self.observation_space = spaces.Box(-np.inf, np.inf, (nf,), np.float32)
        self.action_space = spaces.Discrete(self.n_models)
    
    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.idx = self.rng.integers(0, len(self.data))
        return np.array(self.data[self.idx]['meta_features'], dtype=np.float32), {}
    
    def step(self, action):
        perfs = self.data[self.idx]['model_performances']
        if isinstance(perfs, dict):
            perfs = [perfs.get(n, 0.5) for n in self.model_names]
        sel, best = perfs[action], max(perfs)
        reward = sel
        if sel == best:
            reward += 0.1
            self.optimal += 1
        elif sel >= best - 0.02:
            reward += 0.05
        self.episodes += 1
        self.total_reward += reward
        obs = np.array(self.data[self.idx]['meta_features'], dtype=np.float32)
        return obs, reward, True, False, {'is_optimal': sel == best}
    
    def get_stats(self):
        return {'optimal_rate': self.optimal / max(self.episodes, 1),
                'avg_reward': self.total_reward / max(self.episodes, 1)}


# ==============================================================================
# SECTION 10: PPO TRAINER (GPU)
# ==============================================================================

class PPOTrainer:
    def __init__(self, config, task_type, logger=None):
        self.config = config
        self.task_type = task_type
        self.logger = logger or logging.getLogger(__name__)
        self.data = None
        self.env = None
        self.model = None
    
    def prepare(self, data):
        names = self.config.get_model_names(self.task_type)
        self.data = []
        for r in data:
            perfs = r['model_performances']
            if isinstance(perfs, dict):
                perfs = [perfs.get(n, 0.5) for n in names]
            self.data.append({'meta_features': r['meta_features'], 'model_performances': perfs})
    
    def create_env(self):
        names = self.config.get_model_names(self.task_type)
        self.env = ModelSelectionEnv(self.data, names, self.task_type)
    
    def train(self, timesteps=None):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        # REQUIRE GPU
        gpu_manager.require_gpu()
        
        if timesteps is None:
            timesteps = self.config.PPO_TOTAL_TIMESTEPS
        if self.env is None:
            self.create_env()
        
        self.logger.info("=" * 60)
        self.logger.info(f" PPO TRAINING - {self.task_type.value.upper()} (GPU: {gpu_manager.gpu_name})")
        self.logger.info(f"Timesteps: {timesteps}, Device: CUDA")
        
        vec_env = DummyVecEnv([lambda: self.env])
        
        self.model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=self.config.PPO_LEARNING_RATE,
            n_steps=self.config.PPO_N_STEPS,
            batch_size=self.config.PPO_BATCH_SIZE,
            n_epochs=self.config.PPO_N_EPOCHS,
            gamma=self.config.PPO_GAMMA,
            verbose=1,
            device='cuda',  # GPU REQUIRED
            tensorboard_log=os.path.join(self.config.LOGS_DIR, f"tb_{self.task_type.value}"),
            policy_kwargs={'net_arch': self.config.PPO_NET_ARCH}
        )
        
        cb = CheckpointCallback(save_freq=10000, save_path=self.config.MODELS_DIR,
                                name_prefix=f"ppo_gpu_{self.task_type.value}")
        
        self.model.learn(total_timesteps=timesteps, callback=cb, progress_bar=True)
        
        path = os.path.join(self.config.MODELS_DIR, self.config.get_ppo_model_file(self.task_type))
        self.model.save(path)
        self.logger.info(f" Saved: {path}")
        
        stats = self.env.get_stats()
        self.logger.info(f"Optimal rate: {stats['optimal_rate']:.2%}")
        return self.model
    
    def load(self, path=None):
        from stable_baselines3 import PPO
        if path is None:
            path = os.path.join(self.config.MODELS_DIR, self.config.get_ppo_model_file(self.task_type))
        self.model = PPO.load(path, device='cuda')
        return self.model
    
    def evaluate(self, n=1000):
        if self.env is None: self.create_env()
        opt = 0
        for _ in range(n):
            obs, _ = self.env.reset()
            act, _ = self.model.predict(obs, deterministic=True)
            _, _, _, _, info = self.env.step(act)
            if info['is_optimal']: opt += 1
        self.logger.info(f"Eval optimal rate: {opt/n:.2%}")


# ==============================================================================
# SECTION 11: MODEL SELECTOR (INFERENCE)
# ==============================================================================

class RLModelSelector:
    def __init__(self, config=None):
        self.config = config or RLModelSelectorConfig()
        self.meta_extractor = MetaFeatureExtractor(self.config)
        self.ppo_clf = None
        self.ppo_reg = None
    
    def load(self, task_type=None):
        from stable_baselines3 import PPO
        gpu_manager.require_gpu()
        
        if task_type is None or task_type == TaskType.CLASSIFICATION:
            p = os.path.join(self.config.MODELS_DIR, self.config.PPO_MODEL_CLF_FILE)
            if os.path.exists(p + ".zip"):
                self.ppo_clf = PPO.load(p, device='cuda')
                print(" Loaded CLF model (GPU)")
        if task_type is None or task_type == TaskType.REGRESSION:
            p = os.path.join(self.config.MODELS_DIR, self.config.PPO_MODEL_REG_FILE)
            if os.path.exists(p + ".zip"):
                self.ppo_reg = PPO.load(p, device='cuda')
                print("✅ Loaded REG model (GPU)")
    
    def select(self, df, target, task_type=None, top_k=3):
        import torch
        X, y = df.drop(columns=[target]), df[target]
        if task_type is None:
            task_type = TaskType.detect(y)
        
        ppo = self.ppo_clf if task_type == TaskType.CLASSIFICATION else self.ppo_reg
        names = self.config.CLF_MODEL_NAMES if task_type == TaskType.CLASSIFICATION else self.config.REG_MODEL_NAMES
        
        if ppo is None:
            raise ValueError(f"No model for {task_type.value}")
        
        mf = self.meta_extractor.extract(X, y, task_type)
        policy = ppo.policy
        obs = torch.FloatTensor(mf.reshape(1, -1)).to('cuda')
        
        with torch.no_grad():
            feat = policy.extract_features(obs)
            if hasattr(policy, 'mlp_extractor'):
                lat, _ = policy.mlp_extractor(feat)
            else:
                lat = feat
            dist = policy.action_dist.proba_distribution(policy.action_net(lat))
            probs = dist.distribution.probs.cpu().numpy()[0]
        
        top = np.argsort(probs)[::-1][:top_k]
        return {
            'task_type': task_type.value,
            'models': [names[i] for i in top],
            'probs': [float(probs[i]) for i in top],
            'all': dict(zip(names, probs.tolist()))
        }
    
    def print_report(self, df, target, task_type=None):
        r = self.select(df, target, task_type)
        print("\n" + "=" * 60)
        print(f" RL MODEL SELECTOR - {r['task_type'].upper()} (GPU)")
        print("=" * 60)
        print("\n TOP 3 MODELS:")
        for i, (m, p) in enumerate(zip(r['models'], r['probs'])):
            print(f"   {i+1}. {m}: {p:.1%}")
        print("\n ALL PROBABILITIES:")
        for m, p in sorted(r['all'].items(), key=lambda x: -x[1]):
            print(f"   {m:28s} {p:5.1%} {'█'*int(p*30)}")
        print("=" * 60)


# ==============================================================================
# SECTION 12: MAIN
# ==============================================================================

def run_train(source, task, n_datasets, timesteps):
    gpu_manager.require_gpu()
    
    config = RLModelSelectorConfig()
    config.N_DATASETS_CLF = config.N_DATASETS_REG = n_datasets
    config.PPO_TOTAL_TIMESTEPS = timesteps
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info(f" GPU TRAINING - {gpu_manager.gpu_name}")
    logger.info(f"Source: {source}, Task: {task}")
    logger.info("=" * 70)
    
    gen = SyntheticDataGenerator(config, logger)
    
    if task in ['classification', 'both']:
        data = gen.generate_clf(n_datasets)
        t = PPOTrainer(config, TaskType.CLASSIFICATION, logger)
        t.prepare(data)
        t.train(timesteps)
        t.evaluate(500)
    
    if task in ['regression', 'both']:
        data = gen.generate_reg(n_datasets)
        t = PPOTrainer(config, TaskType.REGRESSION, logger)
        t.prepare(data)
        t.train(timesteps)
        t.evaluate(500)
    
    logger.info("\n TRAINING COMPLETE!")


def run_select(data_path, target, task):
    gpu_manager.require_gpu()
    config = RLModelSelectorConfig()
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")
    tt = TaskType.CLASSIFICATION if task == 'classification' else (TaskType.REGRESSION if task == 'regression' else None)
    sel = RLModelSelector(config)
    sel.load(tt)
    sel.print_report(df, target, tt)


def run_demo():
    gpu_manager.require_gpu()
    config = RLModelSelectorConfig()
    logger = setup_logging(config, "demo")
    
    logger.info(f" GPU DEMO - {gpu_manager.gpu_name}")
    
    gen = SyntheticDataGenerator(config, logger)
    
    clf_data = gen.generate_clf(200)
    t = PPOTrainer(config, TaskType.CLASSIFICATION, logger)
    t.prepare(clf_data)
    t.train(10000)
    
    reg_data = gen.generate_reg(200)
    t = PPOTrainer(config, TaskType.REGRESSION, logger)
    t.prepare(reg_data)
    t.train(10000)
    
    sel = RLModelSelector(config)
    sel.load()
    
    from sklearn.datasets import make_classification, make_regression
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
    df['target'] = y
    sel.print_report(df, 'target')
    
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
    df['target'] = y
    sel.print_report(df, 'target')
    
    logger.info(" DEMO COMPLETE!")


def main():
    parser = argparse.ArgumentParser(description='RL Model Selector - GPU REQUIRED')
    parser.add_argument('--mode', choices=['train', 'select', 'demo', 'check-gpu'], default='demo')
    parser.add_argument('--source', choices=['openml', 'synthetic'], default='synthetic')
    parser.add_argument('--task', choices=['classification', 'regression', 'both'], default='both')
    parser.add_argument('--n_datasets', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--data', type=str)
    parser.add_argument('--target', type=str)

    args = parser.parse_args()
    
    if args.mode == 'check-gpu':
        gpu_manager.print_gpu_info()
    elif args.mode == 'train':
        run_train(args.source, args.task, args.n_datasets, args.timesteps)
    elif args.mode == 'select':
        if not args.data or not args.target:
            print("Error: --data and --target required")
            sys.exit(1)
        run_select(args.data, args.target, args.task if args.task != 'both' else None)
    else:
        run_demo()


if __name__ == '__main__':
    main()