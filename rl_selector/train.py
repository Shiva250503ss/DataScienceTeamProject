# rl_selector/train.py

"""
PPO Training Script for the RL Model Selector.

This script trains a PPO agent that learns to recommend the best ML model
for a given dataset, based on 32 meta-features extracted from the data.

Training pipeline:
  1. Collect REAL training data from OpenML (via data_collection.py)
     - Downloads diverse real-world datasets
     - Extracts 32 meta-features per dataset
     - Trains all 10-11 candidate models and records real CV scores
  2. Load training data into the Gymnasium environment
  3. Train PPO agent (3-layer MLP: 256→128→64) for N timesteps
  4. Save trained model to disk for inference

Usage:
    # Collect data first (one-time, takes ~30-60 min):
    python -m rl_selector.data_collection --task classification --n 30

    # Then train the PPO agent:
    python -m rl_selector.train

    # Or do both at once (collect + train):
    python -m rl_selector.train --collect --n_datasets 30
"""

import os
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rl_selector.environment import ModelSelectionEnv
from rl_selector.data_collection import collect_training_data
from utils.config import config


def train_rl_selector(task_type: str = 'classification',
                      total_timesteps: int = 100000,
                      collect_new: bool = False,
                      n_datasets: int = 30):
    """
    Train a PPO agent to select the best ML model for a dataset.
    
    Training process:
      1. Load existing real training data (or collect new from OpenML)
      2. Create Gymnasium environment
      3. Initialize PPO with 3-layer MLP policy (256→128→64)
      4. Train for specified timesteps
      5. Save trained model to disk
    
    The trained model can then be loaded by RLModelSelector for inference.
    
    Args:
        task_type: 'classification' or 'regression'
        total_timesteps: Number of training steps (more = better but slower)
        collect_new: If True, collect fresh data from OpenML before training
        n_datasets: Number of OpenML datasets to collect (if collect_new=True)
    """
    print(f"\n{'='*60}")
    print(f"Training RL Model Selector for {task_type}")
    print(f"{'='*60}")
    
    # Step 1: Load or collect REAL training data from OpenML
    data_path = f"{config.PPO_MODEL_PATH}/{task_type}_training_data.json"
    
    if collect_new or not os.path.exists(data_path):
        print(f"\n📡 Collecting REAL training data from OpenML...")
        print(f"   This downloads {n_datasets} real datasets, extracts meta-features,")
        print(f"   and trains all candidate models to record real CV scores.")
        print(f"   First run may take 30-60 minutes. Results are cached for reuse.\n")
        
        training_data = collect_training_data(
            task_type=task_type,
            n_datasets=n_datasets,
            save_path=data_path
        )
        
        if len(training_data) < 5:
            print(f"\n⚠ Only collected {len(training_data)} datasets.")
            print(f"   The PPO agent needs at least 5 datasets to train effectively.")
            print(f"   Try again with more datasets or check your internet connection.")
            return None
    else:
        print(f"Loading existing real training data from {data_path}...")
        with open(data_path, 'r') as f:
            training_data = json.load(f)
    
    print(f"\n📊 Training data: {len(training_data)} real-world datasets")
    
    # Show data summary
    if training_data:
        all_scores = []
        for entry in training_data:
            scores = list(entry['model_scores'].values())
            all_scores.append(max(scores))
        print(f"   Best achievable score range: {min(all_scores):.4f} — {max(all_scores):.4f}")
        print(f"   Average best score: {np.mean(all_scores):.4f}")
    
    # Step 2: Create environment
    env = ModelSelectionEnv(task_type=task_type)
    env.load_training_data(training_data)
    
    # Step 3: Create PPO model with 3-layer neural network
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 128, 64]},
        verbose=1,
        device="auto"  # Use GPU if available, else CPU
    )
    
    # Step 4: Train
    print(f"\n🏋️ Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Step 5: Save trained model
    model_path = f"{config.PPO_MODEL_PATH}/ppo_{task_type[:3]}.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Quick evaluation
    print("\n📈 Quick evaluation on training data:")
    correct = 0
    total_eval = min(50, len(training_data))
    regrets = []
    for _ in range(total_eval):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        regrets.append(info['regret'])
        if info['regret'] < 0.02:
            correct += 1
    
    print(f"  Near-optimal selections: {correct}/{total_eval} ({correct/total_eval*100:.0f}%)")
    print(f"  Mean regret: {np.mean(regrets):.4f}")
    print(f"  Max regret:  {np.max(regrets):.4f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL Model Selector')
    parser.add_argument('--task', type=str, default='both',
                       choices=['classification', 'regression', 'both'],
                       help='Which task type to train for')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of PPO training timesteps')
    parser.add_argument('--collect', action='store_true',
                       help='Collect fresh data from OpenML before training')
    parser.add_argument('--n_datasets', type=int, default=30,
                       help='Number of OpenML datasets to collect')
    
    args = parser.parse_args()
    
    if args.task in ('classification', 'both'):
        train_rl_selector(
            'classification',
            total_timesteps=args.timesteps,
            collect_new=args.collect,
            n_datasets=args.n_datasets
        )
    
    if args.task in ('regression', 'both'):
        train_rl_selector(
            'regression',
            total_timesteps=args.timesteps,
            collect_new=args.collect,
            n_datasets=args.n_datasets
        )
