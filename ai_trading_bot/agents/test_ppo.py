import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import TradingEnv


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "NIFTY50.csv"
MODEL_PATH = PROJECT_ROOT / "agents" / "ppo_trading_agent"


df = pd.read_csv(DATA_PATH)

env = TradingEnv(df)
model = PPO.load(MODEL_PATH)

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

env.render()
print("✅ PPO agent evaluation completed")
