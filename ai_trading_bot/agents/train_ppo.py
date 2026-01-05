import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "NIFTY50.csv"
df = pd.read_csv(DATA_PATH)


def make_env():
    return TradingEnv(df)


env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)

print("🚀 Training started...")
model.learn(total_timesteps=20_000)   # reduced for safety
print("✅ Training finished")

MODEL_PATH = PROJECT_ROOT / "agents" / "ppo_trading_agent"
model.save(MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)
