# rl_selector/environment.py

"""
Gymnasium environment for RL-based model selection.

RECONCILED (2026-07): this environment previously used a legacy 32-feature
observation space and a model list (incl. XGBoost/LightGBM/CatBoost) that the
shipped production policy was never trained on. It now matches the production
system exactly:

  - Observation: the 40 meta-features from the shared `meta_features.py`
    (the same vector ProfilerAgent computes and `inference.py` consumes)
  - Actions: the exact sklearn model lists from `rl_selector/inference.py`,
    in the same frozen order (action index N must mean the same model at
    training and inference time)

Episode structure (contextual bandit):
  - reset(): observe a random dataset's 40 meta-features
  - step(a): reward = model a's REAL pre-computed CV score on that dataset,
             +0.1 bonus for picking within 0.01 of the best; episode ends.
Training data comes from data_collection.py, which evaluates every candidate
model on real datasets so the environment can reward any action instantly.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple

from meta_features import N_META_FEATURES
from rl_selector.inference import CLASSIFICATION_MODELS, REGRESSION_MODELS


class ModelSelectionEnv(gym.Env):
    """One-step model-selection environment over real dataset evaluations."""

    def __init__(self, task_type: str = 'classification'):
        super().__init__()

        self.task_type = task_type

        # Single source of truth for the action space: the SAME frozen lists
        # inference.py uses to decode policy outputs (8 clf / 9 reg models).
        if task_type == 'classification':
            self.models = list(CLASSIFICATION_MODELS)
        else:
            self.models = list(REGRESSION_MODELS)

        # Observation: 40 normalized meta-features in [0, 1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N_META_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.models))

        # Training data: list of {meta_features: [40 floats], model_scores: {name: score}}
        self.training_data = []
        self.current_idx = 0

    def load_training_data(self, data: List[Dict]):
        """
        Load pre-computed training data (from data_collection.py).

        Each entry must have:
          - 'meta_features': list of 40 floats (shared extractor output)
          - 'model_scores':  dict mapping model name -> real CV score
        """
        # Validate up front — a 32-feature legacy file would silently
        # distribution-shift the policy, so fail loudly instead.
        for i, entry in enumerate(data):
            n = len(entry.get('meta_features', []))
            if n != N_META_FEATURES:
                raise ValueError(
                    f"Training entry {i} has {n} meta-features, expected "
                    f"{N_META_FEATURES}. Re-collect with the current "
                    f"data_collection.py (legacy 32-feature files are not "
                    f"compatible)."
                )
        self.training_data = data
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        """Observe a random dataset's meta-features."""
        super().reset(seed=seed)

        if len(self.training_data) == 0:
            raise RuntimeError(
                "No training data loaded. Call load_training_data() first "
                "(collect it via: python -m rl_selector.data_collection)."
            )

        self.current_idx = int(self.np_random.integers(0, len(self.training_data)))
        data = self.training_data[self.current_idx]
        return np.array(data['meta_features'], dtype=np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Reward the selected model with its real CV score (+ near-best bonus).
        Episodes are single-step, so the returned observation is a zero
        vector (never used by the agent — done=True ends the episode).
        """
        data = self.training_data[self.current_idx]
        model_scores = data['model_scores']

        selected_model = self.models[action]
        # A model absent from the scores dict means it failed during
        # collection — treat as a poor (but defined) outcome.
        selected_score = model_scores.get(selected_model, 0.0)

        reward = selected_score
        best_score = max(model_scores.values())
        if selected_score >= best_score - 0.01:
            reward += 0.1

        info = {
            'selected_model': selected_model,
            'selected_score': selected_score,
            'best_model': max(model_scores, key=model_scores.get),
            'best_score': best_score,
            'regret': best_score - selected_score,
        }

        # Terminal observation: zeros (defined, deterministic — the previous
        # implementation returned random noise here, which polluted nothing
        # functionally but was misleading dummy data).
        terminal_obs = np.zeros(N_META_FEATURES, dtype=np.float32)
        return terminal_obs, reward, True, False, info
