# rl_selector/environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple


class ModelSelectionEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based model selection.
    
    The PPO agent learns which ML model works best for a given dataset
    by observing 32 meta-features and choosing a model action.
    
    How it works:
      - Observation: 32 meta-features describing a dataset (from ProfilerAgent)
      - Action:      Select one ML model from the available models
      - Reward:      The selected model's cross-validation score
                     + bonus if it picked the best model
      - Episode:     Single step — observe dataset, pick model, get reward, done.
    
    The environment is trained on many datasets (from OpenML or pre-collected)
    so the PPO agent learns patterns like:
      "High-dimensional sparse data → XGBoost tends to win"
      "Small dataset with few features → LogisticRegression is competitive"
    
    Owner: Manohar
    """
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__()
        
        self.task_type = task_type
        
        # Define available models per task type
        if task_type == 'classification':
            self.models = [
                'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
                'RandomForestClassifier', 'ExtraTreesClassifier',
                'GradientBoostingClassifier', 'LogisticRegression',
                'SVC', 'KNeighborsClassifier', 'GaussianNB'
            ]
        else:
            self.models = [
                'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
                'RandomForestRegressor', 'ExtraTreesRegressor',
                'GradientBoostingRegressor', 'Ridge', 'Lasso',
                'ElasticNet', 'SVR', 'KNeighborsRegressor'
            ]
        
        # Observation space: 32 normalized meta-features (all between 0 and 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(32,), dtype=np.float32
        )
        
        # Action space: pick one model
        self.action_space = spaces.Discrete(len(self.models))
        
        # Training data: list of {meta_features, model_scores}
        self.training_data = []
        self.current_idx = 0
    
    def load_training_data(self, data: List[Dict]):
        """
        Load pre-computed training data.
        
        Each entry should have:
          - 'meta_features': list of 32 floats
          - 'model_scores': dict mapping model name → CV score
        
        Args:
            data: List of dataset records with meta-features and model scores
        """
        self.training_data = data
        self.current_idx = 0
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to a random dataset from training data.
        
        Returns:
            observation: 32 meta-features of the selected dataset
            info: empty dict (Gymnasium requirement)
        """
        super().reset(seed=seed)
        
        if len(self.training_data) == 0:
            # Return random observation if no training data loaded
            return np.random.rand(32).astype(np.float32), {}
        
        # Pick a random dataset
        self.current_idx = np.random.randint(0, len(self.training_data))
        data = self.training_data[self.current_idx]
        
        return np.array(data['meta_features'], dtype=np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute the model selection action and return reward.
        
        Args:
            action: Index of the selected model
        
        Returns:
            observation: Next state (random, since episode ends)
            reward:      Selected model's score + bonus for best pick
            terminated:  Always True (single-step episode)
            truncated:   Always False
            info:        Details about selected vs best model
        """
        if len(self.training_data) == 0:
            return np.random.rand(32).astype(np.float32), 0.0, True, False, {}
        
        data = self.training_data[self.current_idx]
        model_scores = data['model_scores']
        
        # Get score for the model the agent selected
        selected_model = self.models[action]
        selected_score = model_scores.get(selected_model, 0.5)
        
        # Base reward = model's CV score
        reward = selected_score
        
        # Bonus +0.1 if the agent picked the best (or near-best) model
        best_score = max(model_scores.values())
        if selected_score >= best_score - 0.01:
            reward += 0.1
        
        # Episode ends after one step
        done = True
        
        info = {
            'selected_model': selected_model,
            'selected_score': selected_score,
            'best_model': max(model_scores, key=model_scores.get),
            'best_score': best_score,
            'regret': best_score - selected_score
        }
        
        # Return dummy next observation (episode is done anyway)
        return np.random.rand(32).astype(np.float32), reward, done, False, info
