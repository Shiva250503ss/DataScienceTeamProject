"""
Quick PPO training run with CSV reward logging.

Trains the model selector on the synthetic dataset pool only (no OpenML
download), logs per-episode reward to a CSV, and saves a learning-curve
plot. Used to produce the training reward figure for the paper without
the multi-hour cost of the full 500-dataset training run.
"""

import os
import sys
import csv

# Allow running from project root or from RL_MODEL_PPO_CORRECT/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from train_rl_model_selector import (
    CLASSIFICATION_MODELS,
    ModelSelectionEnv,
    generate_synthetic_classification_datasets,
)


PROJECT_ROOT = os.path.dirname(_THIS_DIR)
CSV_PATH = os.path.join(_THIS_DIR, "training_reward_log.csv")
FIG_PATH = os.path.join(PROJECT_ROOT, "figures", "rl_training_reward.jpeg")


class CSVRewardLogger(BaseCallback):
    """Append (timestep, reward) for every completed episode to a CSV."""

    def __init__(self, csv_path: str):
        super().__init__(verbose=0)
        self.csv_path = csv_path
        self.episode = 0
        self._fh = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["episode", "timestep", "reward"])

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        rewards = self.locals.get("rewards", [0.0])
        for done, reward in zip(dones, rewards):
            if done:
                self.episode += 1
                self._writer.writerow(
                    [self.episode, int(self.num_timesteps), float(reward)]
                )
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()


def smooth(values, window: int = 200):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_curve(csv_path: str, fig_path: str) -> None:
    timesteps, rewards = [], []
    with open(csv_path) as fh:
        next(fh)  # skip header
        for line in fh:
            _, t, r = line.strip().split(",")
            timesteps.append(int(t))
            rewards.append(float(r))

    rewards_arr = np.asarray(rewards)
    timesteps_arr = np.asarray(timesteps)
    smoothed = smooth(rewards_arr, window=200)
    smoothed_t = timesteps_arr[len(rewards_arr) - len(smoothed) :]

    plt.figure(figsize=(8.5, 4.5))
    plt.plot(
        timesteps_arr,
        rewards_arr,
        color="#9ec5fe",
        alpha=0.4,
        linewidth=0.8,
        label="Per-episode reward",
    )
    plt.plot(
        smoothed_t,
        smoothed,
        color="#1f6feb",
        linewidth=2.2,
        label="Smoothed (window=200)",
    )
    plt.axhline(
        0.5,
        color="#999999",
        linestyle="--",
        linewidth=1,
        label="Random baseline (0.5)",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward (rank-normalized)")
    plt.title("PPO model selector training reward")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, format="jpeg")
    plt.close()
    print(f"[OK] Saved figure -> {fig_path}")


def main():
    total_timesteps = int(os.environ.get("PPO_TIMESTEPS", "30000"))

    print("[*] Loading synthetic classification datasets (no OpenML download)")
    datasets = generate_synthetic_classification_datasets()
    print(f"[OK] {len(datasets)} synthetic datasets ready")

    env = ModelSelectionEnv(datasets, CLASSIFICATION_MODELS, "classification")

    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        max_grad_norm=0.5,
        normalize_advantage=True,
        ent_coef=0.3,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        device="cpu",
    )

    callback = CSVRewardLogger(CSV_PATH)
    print(f"[*] Training PPO for {total_timesteps:,} timesteps")
    ppo.learn(total_timesteps=total_timesteps, callback=callback)
    print(f"[OK] Training complete -- {callback.episode} episodes logged")
    print(f"[OK] Reward CSV -> {CSV_PATH}")

    plot_curve(CSV_PATH, FIG_PATH)


if __name__ == "__main__":
    main()
