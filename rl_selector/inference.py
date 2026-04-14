# rl_selector/inference.py

"""
RL-based model selection at inference time.

Loads the pre-trained PPO models (trained on 40 meta-features) and uses them
to recommend the best ML models for a new dataset.

The model name lists MUST match the exact order used during PPO training in:
    RL_MODEL_PPO_CORRECT/DataScienceTeamProject/train_rl_model_selector.py
"""

import os
import numpy as np
from typing import List, Tuple

# ============================================================================
# MODEL NAMES — MUST match dict key order in the PPO training script
# so action index N maps to the same model at inference.
# ============================================================================

CLASSIFICATION_MODELS = [
    'LogisticRegression',
    'GaussianNB',
    'KNeighborsClassifier',
    'SVC',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
]

REGRESSION_MODELS = [
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

# pkl paths — in project root (same level as ui/, agents/, etc.)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_CLF_PKL = os.path.join(_PROJECT_ROOT, 'rl_model_selector_classification.pkl')
_REG_PKL = os.path.join(_PROJECT_ROOT, 'rl_model_selector_regression.pkl')


class RLModelSelector:
    """
    RL-based model selection at inference time.

    How it works:
      1. Takes 40 meta-features from the ProfilerAgent
      2. Feeds them into the trained PPO policy network
      3. The policy outputs a probability distribution over all models
      4. Returns top-k models sorted by probability (confidence)

    Fallback: If no trained PPO model exists, returns sensible defaults.
    """

    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self.use_rl = False
        self._load_models()

    def _load_models(self):
        """Load pre-trained PPO pkl files from project root."""
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
                self.clf_model = PPO.load(_CLF_PKL, device=device)
                print(f"[RLModelSelector] Loaded classification PPO model (device={device})")

            if os.path.exists(_REG_PKL):
                self.reg_model = PPO.load(_REG_PKL, device=device)
                print(f"[RLModelSelector] Loaded regression PPO model (device={device})")

            if self.clf_model or self.reg_model:
                self.use_rl = True

        except ImportError:
            print("[RLModelSelector] stable-baselines3 not installed — RL disabled, using defaults")
        except Exception as e:
            print(f"[RLModelSelector] RL model not loaded: {e}")

    def recommend(self, meta_features: np.ndarray,
                  task_type: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k model recommendations with confidence scores.

        Args:
            meta_features: 40-element numpy array of dataset meta-features
                           (produced by ProfilerAgent via extract_meta_features)
            task_type:     'classification' or 'regression'
            top_k:         Number of model recommendations to return

        Returns:
            List of (model_name, confidence) tuples, sorted by confidence descending.
        """
        if task_type == 'classification':
            model = self.clf_model
            model_names = CLASSIFICATION_MODELS
        else:
            model = self.reg_model
            model_names = REGRESSION_MODELS

        if model is None:
            return self._default_recommendations(task_type)

        # Get full probability distribution from PPO policy
        try:
            import torch

            device = next(model.policy.parameters()).device
            policy = model.policy
            obs = torch.FloatTensor(meta_features.reshape(1, -1)).to(device)

            with torch.no_grad():
                feat = policy.extract_features(obs)
                if hasattr(policy, 'mlp_extractor'):
                    lat, _ = policy.mlp_extractor(feat)
                else:
                    lat = feat
                dist = policy.action_dist.proba_distribution(policy.action_net(lat))
                action_probs = dist.distribution.probs.cpu().numpy()[0]

        except Exception as e:
            print(f"[RLModelSelector] Warning: Could not get probabilities ({e})")
            try:
                obs = meta_features.reshape(1, -1).astype(np.float32)
                action, _ = model.predict(obs, deterministic=True)
                action_probs = np.ones(len(model_names)) * 0.05
                action_probs[action[0]] = 0.5
            except Exception:
                return self._default_recommendations(task_type)

        # Sort models by probability (descending)
        sorted_indices = np.argsort(action_probs)[::-1]

        recommendations = []
        for i in range(min(top_k, len(model_names))):
            idx = sorted_indices[i]
            if idx < len(model_names):
                recommendations.append((model_names[idx], float(action_probs[idx])))

        print(f"[RLModelSelector] Recommendations ({task_type}):")
        for name, conf in recommendations:
            print(f"  -{name}: {conf:.4f}")

        return recommendations

    def _default_recommendations(self, task_type: str) -> List[Tuple[str, float]]:
        """Fallback when RL is unavailable."""
        print(f"[RLModelSelector] Using default recommendations (no trained model)")
        if task_type == 'classification':
            return [
                ('RandomForestClassifier', 0.40),
                ('GradientBoostingClassifier', 0.35),
                ('LogisticRegression', 0.25),
            ]
        return [
            ('RandomForestRegressor', 0.40),
            ('GradientBoostingRegressor', 0.35),
            ('Ridge', 0.25),
        ]
