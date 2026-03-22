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
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import os
import sys


# ============================================================================
# MODEL LISTS - must match exactly what the pkl models were trained with
# (train_rl_model_selector.py CLASSIFICATION_MODELS / REGRESSION_MODELS)
# ============================================================================

# 8 classification models (action indices 0-7)
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

# 9 regression models (action indices 0-8)
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

CLASSIFICATION_MODELS = {
    'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42),
    'GaussianNB':                 GaussianNB(),
    'KNeighborsClassifier':       KNeighborsClassifier(n_neighbors=5),
    'SVC':                        SVC(kernel='rbf', probability=True, random_state=42),
    'DecisionTreeClassifier':     DecisionTreeClassifier(random_state=42),
    'RandomForestClassifier':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTreesClassifier':       ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

REGRESSION_MODELS = {
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
        """
        Load the latest pre-trained PPO pkl files from RL_MODEL_PPO_CORRECT.
        Loads separate models for classification and regression.
        """
        try:
            from stable_baselines3 import PPO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if os.path.exists(_CLF_PKL):
                self.rl_model_clf = PPO.load(_CLF_PKL, device=device)
                print(f"[OK] Loaded CLF RL model from {_CLF_PKL} (device={device})")

            if os.path.exists(_REG_PKL):
                self.rl_model_reg = PPO.load(_REG_PKL, device=device)
                print(f"[OK] Loaded REG RL model from {_REG_PKL} (device={device})")

            if self.rl_model_clf is not None or self.rl_model_reg is not None:
                self.use_rl = True

        except Exception as e:
            print(f"[INFO] RL model not loaded: {e}")

    def _get_rl_recommendations(self, meta_features: np.ndarray, task_type: str) -> List[str]:
        """
        Get top-3 model recommendations from the RL agent.

        Uses the full PPO policy probability distribution (as done in
        RL_MODEL_PPO_CORRECT/app_rl_selector.py) - NOT the single argmax action.
        Models are sorted by selection probability, highest first.
        """
        if not self.use_rl:
            return self._get_default_recommendations(task_type)

        try:
            import torch

            if task_type == 'classification':
                ppo = self.rl_model_clf
                names = CLF_MODEL_NAMES
            else:
                ppo = self.rl_model_reg
                names = REG_MODEL_NAMES

            if ppo is None:
                return self._get_default_recommendations(task_type)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

            # Rank all models by probability (highest first), return top 3
            top_indices = np.argsort(probs)[::-1][:3]
            top_models = [names[i] for i in top_indices if i < len(names)]
            print(f"[RL] Top recommendations ({task_type}): {top_models}")
            return top_models

        except Exception as e:
            print(f"[WARN] RL selection failed: {e}")
            return self._get_default_recommendations(task_type)

    def _get_default_recommendations(self, task_type: str) -> List[str]:
        """Get default model recommendations when RL is not available."""
        if task_type == 'classification':
            return ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']
        else:
            return ['RandomForestRegressor', 'GradientBoostingRegressor', 'Ridge']

    def train(self, X: pd.DataFrame, y: pd.Series, task_type: str,
              meta_features: np.ndarray = None) -> Tuple[Any, np.ndarray, Dict]:
        """
        Train all models and create ensemble.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            meta_features: 32 meta-features for RL selection

        Returns:
            (ensemble_model, y_pred, modeling_report)
        """

        # Get RL (or default) recommendations
        if meta_features is not None:
            recommended = self._get_rl_recommendations(meta_features, task_type)
        else:
            recommended = self._get_default_recommendations(task_type)

        # Get all models
        if task_type == 'classification':
            all_models = CLASSIFICATION_MODELS
        else:
            all_models = REGRESSION_MODELS

        # Train all models with cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"[*] Training {len(all_models)} {task_type} models...")

        for model_name, model in all_models.items():
            try:
                scoring = 'accuracy' if task_type == 'classification' else 'r2'
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                self.model_scores[model_name] = scores.mean()
                model.fit(X, y)
                self.trained_models[model_name] = model
                print(f"[OK] {model_name}: CV Score = {scores.mean():.4f}")
            except Exception as e:
                print(f"[ERROR] {model_name}: {str(e)}")
                self.model_scores[model_name] = 0.0

        # Create ensemble from top 3 models by CV score
        top_3 = sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_models = {
            name: self.trained_models[name]
            for name, _ in top_3
            if name in self.trained_models
        }

        print(f"\n[*] Creating ensemble from top 3 models:")
        for name, score in top_3:
            print(f"    • {name}: {score:.4f}")

        if task_type == 'classification':
            self.ensemble = VotingClassifier(
                estimators=list(top_3_models.items()),
                voting='soft'
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=list(top_3_models.items())
            )

        self.ensemble.fit(X, y)
        y_pred = self.ensemble.predict(X)

        # Compute metrics
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
            'top_3_models': [name for name, _ in top_3],
            'rl_recommended': recommended,
            'ensemble_score': top_3[0][1],
            'metrics': metrics,
            'used_rl': self.use_rl,
        }

        return self.ensemble, y_pred, report
