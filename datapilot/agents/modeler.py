"""
Modeler Agent - Model Training and Ensemble Creation with RL Selection
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# MODEL NAMES — MUST match dict key order in RL_MODEL_PPO_CORRECT's
# train_rl_model_selector.py so action index N -> same model at inference.
# ============================================================================

CLF_MODEL_NAMES = [
    'LogisticRegression',
    'GaussianNB',
    'KNeighborsClassifier',
    'SVC',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
]

REG_MODEL_NAMES = [
    'Ridge',
    'Lasso',
    'ElasticNet',
    'SVR',
    'KNeighborsRegressor',
    'DecisionTreeRegressor',
    'RandomForestRegressor',
    'ExtraTreesRegressor',
    'GradientBoostingRegressor',
]


def _make_clf_models():
    """Create fresh classification model instances each run."""
    return {
        'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42),
        'GaussianNB':                 GaussianNB(),
        'KNeighborsClassifier':       KNeighborsClassifier(n_neighbors=5),
        'SVC':                        SVC(kernel='rbf', probability=True, random_state=42),
        'DecisionTreeClassifier':     DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTreesClassifier':       ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


def _make_reg_models():
    """Create fresh regression model instances each run."""
    return {
        'Ridge':                     Ridge(alpha=1.0),
        'Lasso':                     Lasso(alpha=1.0),
        'ElasticNet':                ElasticNet(alpha=1.0),
        'SVR':                       SVR(kernel='rbf'),
        'KNeighborsRegressor':       KNeighborsRegressor(n_neighbors=5),
        'DecisionTreeRegressor':     DecisionTreeRegressor(random_state=42),
        'RandomForestRegressor':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTreesRegressor':       ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }


# pkl paths — in project root
_PKL_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CLF_PKL = os.path.join(_PKL_BASE, 'rl_model_selector_classification.pkl')
_REG_PKL = os.path.join(_PKL_BASE, 'rl_model_selector_regression.pkl')

# Paths to the trained pkl files (inside RL_MODEL_PPO_CORRECT)
_PKL_BASE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'RL_MODEL_PPO_CORRECT', 'DataScienceTeamProject'
))
_CLF_PKL = os.path.join(_PKL_BASE, 'rl_model_selector_classification.pkl')
_REG_PKL = os.path.join(_PKL_BASE, 'rl_model_selector_regression.pkl')


class ModelerAgent:
    """Trains all models and creates ensemble with RL-powered model selection."""

    def __init__(self):
        self.trained_models = {}
        self.ensemble = None
        self.rl_model_clf = None
        self.rl_model_reg = None
        self.model_scores = {}
        self.use_rl = False
        self._load_rl_model()

    def _load_rl_model(self):
        """Load pre-trained PPO pkl files."""
        try:
            from stable_baselines3 import PPO

            device = 'cpu'
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
            except ImportError:
                pass

            if os.path.exists(_CLF_PKL):
                self.rl_model_clf = PPO.load(_CLF_PKL, device=device)
                print(f"[OK] Loaded CLF RL model (device={device})")

            if os.path.exists(_REG_PKL):
                self.rl_model_reg = PPO.load(_REG_PKL, device=device)
                print(f"[OK] Loaded REG RL model (device={device})")

            if self.rl_model_clf or self.rl_model_reg:
                self.use_rl = True

        except ImportError:
            print("[INFO] stable-baselines3 not installed — RL disabled")
        except Exception as e:
            print(f"[INFO] RL model not loaded: {e}")

    def _get_rl_recommendations(self, meta_features: np.ndarray, task_type: str) -> List[str]:
        """
        Get top-3 model recommendations using full PPO policy probability
        distribution (not just argmax).
        """
        if not self.use_rl:
            return self._get_default_recommendations(task_type)

        try:
            import torch

            ppo = self.rl_model_clf if task_type == 'classification' else self.rl_model_reg
            names = CLF_MODEL_NAMES if task_type == 'classification' else REG_MODEL_NAMES

            if ppo is None:
                return self._get_default_recommendations(task_type)

            device = next(ppo.policy.parameters()).device
            policy = ppo.policy
            obs = torch.FloatTensor(meta_features.reshape(1, -1)).to(device)

            with torch.no_grad():
                feat = policy.extract_features(obs)
                if hasattr(policy, 'mlp_extractor'):
                    lat, _ = policy.mlp_extractor(feat)
                else:
                    lat = feat
                dist = policy.action_dist.proba_distribution(policy.action_net(lat))
                probs = dist.distribution.probs.cpu().numpy()[0]

            top_indices = np.argsort(probs)[::-1][:3]
            top_models = [names[i] for i in top_indices if i < len(names)]

            print(f"[RL] Recommendations ({task_type}):")
            for idx in top_indices[:3]:
                if idx < len(names):
                    print(f"     {names[idx]}: {probs[idx]:.4f}")
            return top_models

        except Exception as e:
            print(f"[WARN] RL selection failed: {e}")
            return self._get_default_recommendations(task_type)

    def _get_default_recommendations(self, task_type: str) -> List[str]:
        """Fallback when RL is unavailable."""
        if task_type == 'classification':
            return ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']
        return ['RandomForestRegressor', 'GradientBoostingRegressor', 'Ridge']

    def train(self, X: pd.DataFrame, y: pd.Series, task_type: str,
              meta_features: np.ndarray = None) -> Tuple[Any, np.ndarray, Dict]:
        """Train all models and create ensemble."""

        # Reset state
        self.trained_models = {}
        self.model_scores = {}
        self.ensemble = None

        # RL or default recommendations
        recommended = (
            self._get_rl_recommendations(meta_features, task_type)
            if meta_features is not None
            else self._get_default_recommendations(task_type)
        )

        # Fresh model instances
        all_models = _make_clf_models() if task_type == 'classification' else _make_reg_models()

        # Train all with 5-fold CV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"[*] Training {len(all_models)} {task_type} models...")

        for name, model in all_models.items():
            try:
                scoring = 'accuracy' if task_type == 'classification' else 'r2'
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                self.model_scores[name] = float(scores.mean())
                model.fit(X, y)
                self.trained_models[name] = model
                print(f"[OK] {name}: CV = {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                self.model_scores[name] = 0.0

        # Top 3 by CV score
        top_3 = sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_names = [n for n, _ in top_3 if n in self.trained_models]

        print(f"\n[*] Ensemble from top 3:")
        for n, s in top_3:
            print(f"    - {n}: {s:.4f}")

        # Build ensemble with pre-fitted estimators
        estimators = [(n, self.trained_models[n]) for n in top_3_names]

        if task_type == 'classification':
            self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
            self.ensemble.estimators_ = [self.trained_models[n] for n in top_3_names]
            self.ensemble.named_estimators_ = {n: self.trained_models[n] for n in top_3_names}
            le = LabelEncoder()
            le.fit(y)
            self.ensemble.le_ = le
            self.ensemble.classes_ = le.classes_
        else:
            self.ensemble = VotingRegressor(estimators=estimators)
            self.ensemble.estimators_ = [self.trained_models[n] for n in top_3_names]
            self.ensemble.named_estimators_ = {n: self.trained_models[n] for n in top_3_names}

        y_pred = self.ensemble.predict(X)

        # Metrics
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred),
            }

        report = {
            'all_models_trained': list(self.trained_models.keys()),
            'model_scores': self.model_scores,
            'top_3_models': top_3_names,
            'rl_recommended': recommended,
            'ensemble_score': top_3[0][1] if top_3 else 0.0,
            'metrics': metrics,
            'used_rl': self.use_rl,
        }

        return self.ensemble, y_pred, report
